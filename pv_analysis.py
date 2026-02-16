import collections
import pandas as pd
import numpy as np
import pvlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import fmi_pv_forecaster as pvfc
from datetime import timedelta, datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.ndimage import uniform_filter
import pytz
import tqdm

def find_clear_days(file_path, print_results=True, threshold=0.8):
    counts = collections.Counter()
    with open(file_path, 'r') as f:
        for line in f:
            day = line[:10].strip()
            if day: counts[day] += 1

    df = pd.DataFrame(list(counts.items()), columns=['Date', 'LineCount'])
    Q1, Q3 = df['LineCount'].quantile(0.25), df['LineCount'].quantile(0.75)
    upper_bound = Q3 + threshold * (Q3 - Q1)
    
    significant_days = df[df['LineCount'] > upper_bound].copy().sort_values(by='LineCount', ascending=False)
    significant_days['Date'] = pd.to_datetime(significant_days['Date'], format='%Y %m %d').dt.date
    return significant_days

def get_extra_data(weather_path, radiation_path, target_date, interval='5min', cams_email=None):
    def load_fmi(path):
        df = pd.read_csv(path, na_values='-')
        df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day']].assign(
            hour=df['Time [UTC]'].str.split(':').str[0], minute=df['Time [UTC]'].str.split(':').str[1]
        )).dt.tz_localize('UTC')
        return df.set_index('timestamp')

    df_ground = pd.concat([load_fmi(weather_path), load_fmi(radiation_path)], axis=1)
    df_ground = df_ground.loc[:, ~df_ground.columns.duplicated()]
    df_day = df_ground.resample(interval).mean(numeric_only=True)
    df_day = df_day[df_day.index.date == target_date].copy()

    if cams_email and not df_day.empty:
        cams_data, _ = pvlib.iotools.get_cams(
            latitude=62.8924, longitude=27.6770, start=df_day.index.min(), 
            end=df_day.index.max(), email=cams_email, identifier='cams_radiation', integrated=True
        )
        cams_aligned = cams_data.reindex(df_day.index, method='ffill')
        
        df_final = pd.DataFrame(index=df_day.index)
        df_final['ghi'] = df_day['Global radiation [W/m2]'].fillna(cams_aligned['ghi'])
        df_final['dhi'] = df_day['Diffuse radiation [W/m2]'].fillna(cams_aligned['dhi'])
        df_final['dni'] = cams_aligned['dni']
        df_final['T'] = df_day['Air temperature [°C]']
        df_final['wind'] = df_day['Wind speed [m/s]']
        df_final['albedo'] = 0.7 if df_final['T'].mean() < 0 else 0.2
        return df_final[['dni', 'dhi', 'ghi', 'T', 'wind', 'albedo']]
    return pd.DataFrame()

def apply_shadows_with_window(forecast_df, shadow_matrix, lat, lon, window_size=(3, 3)):
    df = forecast_df.copy()
    
    clean_matrix = np.nan_to_num(shadow_matrix, nan=0.0)
    solpos = pvlib.solarposition.get_solarposition(df.index, lat, lon)
    
    df['altitude'] = (90 - solpos['apparent_zenith']).round().fillna(0).astype(int).clip(lower=1, upper=clean_matrix.shape[0])
    df['azimuth'] = solpos['azimuth'].round().fillna(0).astype(int).clip(lower=0, upper=clean_matrix.shape[1] - 1)
    
    filter_shape = (int(window_size[1]), int(window_size[0])) if isinstance(window_size, (tuple, list)) else (int(window_size), int(window_size))
    smoothed_matrix = uniform_filter(clean_matrix, size=filter_shape, mode='nearest')

    df['shadow_factor'] = smoothed_matrix[df['altitude'].values - 1, df['azimuth'].values]
    df['output_shaded'] = df['output'] * (1 - df['shadow_factor'])
    
    return df

def pv_analysis(target_date, shadow_matrix_df, excel_df, df_extra, window_size=(3, 3), system_efficiency=0.85, plot=True):
    local_tz = pytz.timezone('Europe/Helsinki')
    aware_dt = local_tz.localize(datetime.combine(pd.to_datetime(target_date).date(), datetime.min.time()))
    fc_shift = int(aware_dt.utcoffset().total_seconds() / 3600)
    inv_shift = fc_shift - 2 

    target_date_obj = pd.to_datetime(target_date).date()
    day_data = excel_df[excel_df['Timestamp'].dt.date == target_date_obj].copy().set_index('Timestamp')
    day_data.index += timedelta(hours=inv_shift)
    
    full_day_index = pd.date_range(start=f"{target_date_obj} 00:00:00", periods=288, freq='5min')
    day_data = day_data.reindex(full_day_index, fill_value=0.0)
    
    pvfc.set_angles(12, 170)
    pvfc.set_location(62.979849, 27.648656)
    pvfc.set_nominal_power_kw(3.96)
    
    forecast_base = pvfc.process_radiation_df(df_extra)
    forecast_windowed = apply_shadows_with_window(forecast_base, shadow_matrix_df.values, 62.979849, 27.648656, window_size=window_size)
    
    forecast_base.index += timedelta(hours=fc_shift)
    forecast_windowed.index += timedelta(hours=fc_shift)
    
    forecast_base.index = forecast_base.index.tz_localize(None)
    forecast_windowed.index = forecast_windowed.index.tz_localize(None)
    
    fb_n = forecast_base[['output']].reindex(full_day_index, fill_value=0.0)
    fw_n = forecast_windowed[['output_shaded']].reindex(full_day_index, fill_value=0.0)

    # --- NEW: Apply System Electrical Efficiency ---
    fb_n['output'] = fb_n['output'] * system_efficiency
    fw_n['output_shaded'] = fw_n['output_shaded'] * system_efficiency

    if plot:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(day_data.index, day_data['Power_W'], label='Real Power Output', color='#2ecc71', lw=1.5)
        ax.plot(fb_n.index, fb_n['output'], label='FMI PV Forecast (No Shadows)', color='#3498db', linestyle='--')
        ax.plot(fw_n.index, fw_n['output_shaded'], label=f'Forecast with Windowed Shadows', color='#e67e22', linestyle='-')
        
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_xlim(full_day_index[0], full_day_index[-1])
        
        plt.title(f"PV Comparison & Shadow Impact: {target_date_obj}")
        plt.xlabel("Time (Helsinki Local)")
        plt.ylabel("Power (W)")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return day_data, fb_n, fw_n

def compute_metrics(day_data, forecast_base, forecast_windowed):
    real = day_data['Power_W'].fillna(0.0)
    base = forecast_base['output'].fillna(0.0)
    shaded = forecast_windowed['output_shaded'].fillna(0.0)

    return {
        'RMSE_Base': np.sqrt(mean_squared_error(real, base)),
        'RMSE_Shaded': np.sqrt(mean_squared_error(real, shaded)),
        'MAE_Base': mean_absolute_error(real, base),
        'MAE_Shaded': mean_absolute_error(real, shaded),
        'MBE_Base': np.mean(base - real),
        'MBE_Shaded': np.mean(shaded - real),
        'Real_Wh': real.sum() * (5/60),
        'Base_Wh': base.sum() * (5/60),
        'Shaded_Wh': shaded.sum() * (5/60)
    }

# def plot_real_vs_predicted_scatter(all_real, all_pred):
#     """Generates a highly academic scatter plot with an R-squared trendline."""
#     real_arr = np.array(all_real)
#     pred_arr = np.array(all_pred)
    
#     # Filter out pure night-time zeros to not skew the R2 with trivial data
#     mask = (real_arr > 50) | (pred_arr > 50)
#     real_filtered = real_arr[mask]
#     pred_filtered = pred_arr[mask]
    
#     r2 = r2_score(real_filtered, pred_filtered)
    
#     plt.figure(figsize=(8, 8))
#     plt.scatter(real_filtered, pred_filtered, alpha=0.2, color='#3498db', edgecolors='none')
    
#     # Perfect alignment line (1:1)
#     max_val = max(real_filtered.max(), pred_filtered.max())
#     plt.plot([0, max_val], [0, max_val], 'k--', label='1:1 Perfect Prediction', lw=2)
    
#     # Linear Regression Trendline
#     z = np.polyfit(real_filtered, pred_filtered, 1)
#     p = np.poly1d(z)
#     plt.plot(real_filtered, p(real_filtered), '#e74c3c', lw=2, label=f'Trendline ($R^2$ = {r2:.3f})')
    
#     plt.title('Real vs. Predicted Power Output (All Clear Days)')
#     plt.xlabel('Real Power Output (W)')
#     plt.ylabel('LiDAR Shaded Forecast (W)')
#     plt.xlim(0, max_val * 1.05)
#     plt.ylim(0, max_val * 1.05)
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()

def evaluate_performance(significant_days_df, shadow_matrix_df, excel_df, weather_path, radiation_path, cams_email, system_efficiency=0.85):
    daily_stats = []
    all_real_power = []
    all_pred_power = []

    for date_obj in tqdm.tqdm(significant_days_df['Date'].tolist(), desc="Evaluating Days"):
        df_extra = get_extra_data(weather_path, radiation_path, date_obj, '5min', cams_email)
        if df_extra.empty: continue
            
        day_data, fb, fw = pv_analysis(date_obj, shadow_matrix_df, excel_df, df_extra, window_size=(3, 3), system_efficiency=system_efficiency, plot=False)
        metrics = compute_metrics(day_data, fb, fw)
        
        # Accumulate 5-minute data arrays for the scatter plot
        all_real_power.extend(day_data['Power_W'].fillna(0.0).values)
        all_pred_power.extend(fw['output_shaded'].fillna(0.0).values)
        
        daily_stats.append({
            'Date': date_obj.strftime('%Y-%m-%d'), 
            'RMSE_Base': metrics['RMSE_Base'], 
            'RMSE_Shaded': metrics['RMSE_Shaded'],
            'MAE_Base': metrics['MAE_Base'],
            'MAE_Shaded': metrics['MAE_Shaded'],
            'MBE_Base': metrics['MBE_Base'],
            'MBE_Shaded': metrics['MBE_Shaded'],
            'Real_Wh': metrics['Real_Wh'], 'Base_Wh': metrics['Base_Wh'], 'Shaded_Wh': metrics['Shaded_Wh']
        })

    results_df = pd.DataFrame(daily_stats)
    
    # Generate the scatter plot
    print("\nGenerating Scatter Plot Analysis...")
    plot_real_vs_predicted_scatter(all_real_power, all_pred_power)
    
    return results_df