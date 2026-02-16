import collections
import pandas as pd
import numpy as np
import pvlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import fmi_pv_forecaster as pvfc
from datetime import timedelta, datetime
from sklearn.metrics import mean_squared_error
import tqdm

def find_clear_days(file_path, print_results=True, threshold=0.8):
    # 1. Efficiently count lines per day
    counts = collections.Counter()
    
    with open(file_path, 'r') as f:
        for line in f:
            # Slicing "YYYY MM DD" from the start of the line
            day = line[:10].strip()
            if day:
                counts[day] += 1

    # 2. Convert to DataFrame for statistical analysis
    df = pd.DataFrame(list(counts.items()), columns=['Date', 'LineCount'])
    
    # 3. Calculate IQR (Interquartile Range)
    Q1 = df['LineCount'].quantile(0.25)
    Q3 = df['LineCount'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define the "High Outlier" threshold
    # 1.5 is standard; use 3.0 for "Extreme" outliers
    upper_bound = Q3 + threshold * IQR
    
    # 4. Filter for statistically significant days
    significant_days = df[df['LineCount'] > upper_bound].copy()
    
    # Sort by count to get the most significant first
    significant_days = significant_days.sort_values(by='LineCount', ascending=False)
    significant_days['Date'] = pd.to_datetime(significant_days['Date'], format='%Y %m %d').dt.date
    
    # 5. Output Results
    if print_results:
        print(f"Total days analyzed: {len(df)}")
        print(f"Statistical Upper Bound: {upper_bound:.2f} lines/day")
        print(f"Number of statistically significant days: {len(significant_days)}")
        print("-" * 30)
        print("Top Statistically Significant Days:")
        print(significant_days.head(20).to_string(index=False))

    return significant_days


def get_extra_data(weather_path, radiation_path, target_date, interval='5T', cams_email=None):
    # 1. Load and Standardize FMI Ground Data
    def load_fmi(path):
        # Use na_values='-' to handle the missing data in your radiation CSV
        df = pd.read_csv(path, na_values='-')
        df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day']].assign(
            hour=df['Time [UTC]'].str.split(':').str[0],
            minute=df['Time [UTC]'].str.split(':').str[1]
        )).dt.tz_localize('UTC')
        return df.set_index('timestamp')

    # Merge ground weather and radiation
    df_ground = pd.concat([load_fmi(weather_path), load_fmi(radiation_path)], axis=1)
    df_ground = df_ground.loc[:, ~df_ground.columns.duplicated()]
    
    # Resample to the requested interval (e.g., 5 min)
    df_resampled = df_ground.resample(interval).mean(numeric_only=True)
    
    # Slice for the target day
    # target_dt = pd.to_datetime(target_date).date()
    df_day = df_resampled[df_resampled.index.date == target_date].copy()

    # 2. Integrate CAMS All-Sky for DNI only
    if cams_email and not df_day.empty:
        cams_data, metadata = pvlib.iotools.get_cams(
            latitude=62.8924, longitude=27.6770, 
            start=df_day.index.min(), end=df_day.index.max(), 
            email=cams_email, identifier='cams_radiation', integrated=True
        )
        cams_aligned = cams_data.reindex(df_day.index, method='ffill')

        # 3. Create Standardized Final DataFrame
        df_final = pd.DataFrame(index=df_day.index)
        
        # PRIORITIZE GROUND TRUTH (FMI)
        df_final['ghi'] = df_day['Global radiation [W/m2]']
        df_final['dhi'] = df_day['Diffuse radiation [W/m2]']
        
        # USE SATELLITE (CAMS) FOR MISSING DNI
        df_final['dni'] = cams_aligned['dni']
        
        # WEATHER PARAMETERS
        df_final['T'] = df_day['Air temperature [°C]']
        df_final['wind'] = df_day['Wind speed [m/s]']
        
        # ALBEDO HEURISTIC (April in Kuopio often has snow)
        # Using 0.7 if frozen, 0.2 if above freezing
        df_final['albedo'] = 0.7 if df_final['T'].mean() < 0 else 0.2
        
        # Cleanup: In case FMI has gaps, we can backfill GHI/DHI with CAMS as a fallback
        df_final['ghi'] = df_final['ghi'].fillna(cams_aligned['ghi'])
        df_final['dhi'] = df_final['dhi'].fillna(cams_aligned['dhi'])
        
        return df_final[['dni', 'dhi', 'ghi', 'T', 'wind', 'albedo']]

    return pd.DataFrame()

# Usage
# df = get_hybrid_kuopio_dataset('weather.csv', 'radiation.csv', '2021-04-01', '5T', 'haoyang.dong@uef.fi')

def apply_shadows(forecast_df, shadow_matrix, lat, lon):
    # Use .copy() to solve the SettingWithCopyWarning
    df = forecast_df.copy()
    
    # 1. Calculate Solar Position
    solpos = pvlib.solarposition.get_solarposition(df.index, lat, lon)
    
    # Altitude = 90 - Zenith. 
    # Use .fillna(0) and cast to int to prevent IndexError
    df['altitude'] = (90 - solpos['apparent_zenith']).round().fillna(0).astype(int)
    df['azimuth'] = solpos['azimuth'].round().fillna(0).astype(int)
    
    # 2. Bound the angles to the matrix dimensions (Alt: 1-90, Az: 0-360)
    df['altitude'] = df['altitude'].clip(lower=1, upper=90)
    df['azimuth'] = df['azimuth'].clip(lower=0, upper=360)
    
    # 3. Vectorized Lookup (Much faster than .apply)
    # shadow_matrix[altitude_index, azimuth_index]
    # We subtract 1 from altitude because index 0 = Altitude_1
    alt_idx = df['altitude'].values - 1
    az_idx = df['azimuth'].values
    
    # Extract the factors directly using NumPy indexing
    df['shadow_factor'] = 1-shadow_matrix[alt_idx, az_idx]
    
    # 4. Final Power Calculation
    df['output_shaded'] = df['output'] * df['shadow_factor']
    
    return df

def apply_shadows_with_window(forecast_df, shadow_matrix, lat, lon, window_size=(5, 3)):
    df = forecast_df.copy()
    
    # 1. Calculate Solar Position
    solpos = pvlib.solarposition.get_solarposition(df.index, lat, lon)
    df['altitude'] = (90 - solpos['apparent_zenith']).round().fillna(0).astype(int)
    df['azimuth'] = solpos['azimuth'].round().fillna(0).astype(int)
    
    # 2. Define rectangular half-window offsets.
    # window_size can be int (square) or tuple/list: (azimuth_width, altitude_width)
    if isinstance(window_size, (tuple, list)) and len(window_size) == 2:
        az_window, alt_window = int(window_size[0]), int(window_size[1])
    else:
        az_window = alt_window = int(window_size)

    az_offset = int(az_window // 2)
    alt_offset = int(alt_window // 2)
    
    def get_windowed_mean(row):
        # Altitude matrix index: Altitude_1 is at index 0
        alt_center = int(row['altitude'] - 1)
        az_center = int(row['azimuth'])
        
        # Calculate slice boundaries and force integer types
        alt_min = int(max(0, alt_center - alt_offset))
        alt_max = int(min(shadow_matrix.shape[0], alt_center + alt_offset + 1))
        
        az_min = int(max(0, az_center - az_offset))
        az_max = int(min(shadow_matrix.shape[1], az_center + az_offset + 1))
        
        # Extract the sub-grid
        window = shadow_matrix[alt_min:alt_max, az_min:az_max]
        
        # Return mean of the window
        return np.mean(window) if window.size > 0 else 0.0

    # 3. Apply the windowed lookup
    df['shadow_factor'] = df.apply(get_windowed_mean, axis=1)
    
    # 4. Final Power Calculation using (1 - shadow_factor)
    # This assumes shadow_factor 1.0 = Total Shade, 0.0 = Clear Sky
    df['output_shaded'] = df['output'] * (1 - df['shadow_factor'])
    
    return df

def pv_analysis(target_date, shadow_matrix_df, excel_df, df_extra, window_size=(5, 3), plot=True):
    """
    Plots real power vs. forecast (with and without shadows) for a specific date.
    target_date_str: String in 'YYYY-MM-DD' format.
    """
    # 1. Date and Season Logic
    # target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
    # Summer Time (EEST) logic: April to October (simplified)
    is_summer = 3 < target_date.month < 11 
    
    # Define shifts
    inv_shift = 1 if is_summer else 0
    fc_shift = 3 if is_summer else 2
    
    # 2. Process Inverter Data (Real Power)
    # Filter for the day, set index, and apply shift
    day_data = excel_df[excel_df['Timestamp'].dt.date == target_date].copy()
    day_data = day_data.set_index('Timestamp')
    day_data.index = day_data.index + timedelta(hours=inv_shift)

    day_data.loc[:, 'Power_W'] = day_data['Power_W'].fillna(0.0)

    full_day_index = pd.date_range(
        start=f"{target_date} 00:00:00", 
        periods=288, 
        freq='5min'
        )
    day_data = day_data.reindex(full_day_index, fill_value=0.0)
    
    # 3. Fetch Environmental and Forecast Data
    # pvfc settings
    pvfc.set_angles(12, 170)
    pvfc.set_location(62.979849, 27.648656)
    pvfc.set_nominal_power_kw(3.76)
    
    # Helper to get external radiation/weather data
    # df_extra = get_extra_data(
    #     'data/pvdata/Kuopio Savilahti 1.4.2021 - 1.10.2021_temp_wind.csv', 
    #     'data/pvdata/Kuopio Savilahti 1.4.2021 - 1.10.2021_radcsv', 
    #     target_date, '5min', 'haoyang.dong@uef.fi'
    # )
    
    # Generate baseline forecast
    forecast_base = pvfc.process_radiation_df(df_extra)
    
    # Apply Shadows with Windowed Mean
    # Note: shadow_matrix_df.values is passed to handle the lookup
    forecast_windowed = apply_shadows_with_window(
        forecast_base, shadow_matrix_df.values, 
        62.979849, 27.648656, window_size=window_size
    )
    
    # 4. Apply Shifts to Forecast Data
    # Move forecast from UTC to Local Helsinki time
    # Ensure forecast_base and forecast_windowed are copies
    forecast_base = forecast_base.copy()
    forecast_windowed = forecast_windowed.copy()

    forecast_base.index = forecast_base.index + timedelta(hours=fc_shift)
    forecast_windowed.index = forecast_windowed.index + timedelta(hours=fc_shift)
    
    # Use .loc to explicitly set values and handle NaNs
    forecast_base.loc[:, 'output'] = forecast_base['output'].fillna(0.0)
    forecast_windowed.loc[:, 'output_shaded'] = forecast_windowed['output_shaded'].fillna(0.0)

    forecast_base_clean = forecast_base[['output']].copy()
    forecast_windowed_clean = forecast_windowed[['output_shaded']].copy()

    day_start = forecast_base_clean.index[0].normalize()
    full_day_idx_n = pd.date_range(start=day_start, periods=288, freq='5min')

    # 3. Reindex to add missing timestamps and pad with 0.0
    forecast_base_n = forecast_base_clean.reindex(full_day_idx_n, fill_value=0.0)
    forecast_windowed_n = forecast_windowed_clean.reindex(full_day_idx_n, fill_value=0.0)
    # forecast_base_clean.index.name = 'Timestamp'
    # forecast_windowed_clean.index.name = 'Timestamp'

    # full_day_index = pd.date_range(
    #     start=f"{target_date} 00:00:00", 
    #     periods=288, 
    #     freq='5min'
    #     )
    
    # # Reindex all dataframes to the full day index
    # forecast_base_clean = forecast_base_clean.reindex(full_day_index, fill_value=0.0)
    # forecast_windowed_clean = forecast_windowed_clean.reindex(full_day_index, fill_value=0.0)
    # day_data = day_data.reindex(full_day_index, fill_value=0.0)

    # 5. Plotting
    if plot:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Real Data
        ax.plot(day_data.index, day_data['Power_W'], 
                label='Real Power Output', color='#2ecc71', lw=1.5)
    
        # Baseline Model
        ax.plot(forecast_base_n.index, forecast_base_n['output'], 
                label='FMI PV Forecast (No Shadows)', color='#3498db', linestyle='--')
        
        # Shaded Model (Our LiDAR refinement)
        ax.plot(forecast_windowed_n.index, forecast_windowed_n['output_shaded'], 
                label='Forecast with Windowed Shadows', color='#e67e22', linestyle='-')
        
        # X-Axis Styling (Hourly, restricted to current day)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_xlim(pd.Timestamp(target_date), pd.Timestamp(target_date) + timedelta(hours=23, minutes=59))
        
        plt.title(f"PV Comparison & Shadow Impact: {target_date}")
        plt.xlabel("Time (Helsinki Local)")
        plt.ylabel("Power (W)")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    return day_data, forecast_base_n, forecast_windowed_n


# evaluation function to compare forecast with and without shadows against real data (RMSE and energy error)
def compute_metrics(day_data, forecast_base, forecast_windowed):

    real = day_data['Power_W']
    base = forecast_base['output']
    shaded = forecast_windowed['output_shaded']

    # Calculate RMSE
    rmse_b = np.sqrt(mean_squared_error(real, base))
    rmse_s = np.sqrt(mean_squared_error(real, shaded))

    # Calculate Energy (Watt-hours)
    d_real_wh = real.sum() * (5/60)  # Convert from W to Wh
    d_base_wh = base.sum() * (5/60)
    d_shaded_wh = shaded.sum() * (5/60)

    return {
        'RMSE_Base': rmse_b,
        'RMSE_Shaded': rmse_s,
        'Real_Wh': d_real_wh,
        'Base_Wh': d_base_wh,
        'Shaded_Wh': d_shaded_wh
    }


def evaluate_performance(significant_days_df, shadow_matrix_df, excel_df):
    daily_stats = []
    
    # Global energy accumulators (Wh)
    total_real_wh = 0
    total_base_wh = 0
    total_shaded_wh = 0

    # Extract dates from your clear sky outlier detection
    dates = significant_days_df['Date'].tolist()

    for date_str in tqdm.tqdm(dates, desc="Evaluating Days"):
        # Standardize date format for the pv_analysis function
        target_date_obj = datetime.strptime(date_str, '%Y %m %d')
        formatted_date = target_date_obj.strftime('%Y-%m-%d')
        
        # 1. Get cleaned data from your optimized function
        # This handles shifts (0/1h for inv, 2/3h for fc) and NaN filling
        day_data, forecast_base, forecast_shaded = pv_analysis(
            formatted_date, shadow_matrix_df, excel_df, window_size=(5, 3), plot=False
        )
        
        # 2. Reindex to 24h range to ensure exact matching for RMSE
        full_day_idx = pd.date_range(
            start=pd.Timestamp(target_date_obj.date()), 
            periods=288, freq='5min'
        )
        
        real = day_data['Power_W'].reindex(full_day_idx, fill_value=0.0)
        base = forecast_base['output'].reindex(full_day_idx, fill_value=0.0)
        shaded = forecast_shaded['output_shaded'].reindex(full_day_idx, fill_value=0.0)

        # 3. Calculate Daily RMSE (Watts)
        rmse_b = np.sqrt(mean_squared_error(real, base))
        rmse_s = np.sqrt(mean_squared_error(real, shaded))
        
        # 4. Energy Calculations (Watt-hours)
        d_real_wh = real.sum() * (5/60)
        d_base_wh = base.sum() * (5/60)
        d_shaded_wh = shaded.sum() * (5/60)

        # Update global accumulators
        total_real_wh += d_real_wh
        total_base_wh += d_base_wh
        total_shaded_wh += d_shaded_wh

        daily_stats.append({
            'Date': formatted_date,
            'RMSE_Base': rmse_b,
            'RMSE_Shaded': rmse_s,
            'RMSE_Improvement_%': ((rmse_b - rmse_s) / rmse_b * 100) if rmse_b > 0 else 0,
            'Real_Wh': d_real_wh,
            'Base_Wh': d_base_wh,
            'Shaded_Wh': d_shaded_wh
        })

    # 5. Aggregate Global Metrics
    results_df = pd.DataFrame(daily_stats)
    
    # Calculate Energy Errors (Absolute difference in Wh)
    base_error = abs(total_real_wh - total_base_wh)
    shaded_error = abs(total_real_wh - total_shaded_wh)
    
    metrics = {
        'total_real_Wh': total_real_wh.round(2),
        'total_base_Wh': total_base_wh.round(2),
        'total_shaded_Wh': total_shaded_wh.round(2),
        'base_error': base_error.round(2),
        'shaded_error': shaded_error.round(2),
        'mean_RMSE_forecast': results_df['RMSE_Base'].mean(),
        'mean_RMSE_forecast_with_shadow': results_df['RMSE_Shaded'].mean(),
        'RMSE_improvement': results_df['RMSE_Improvement_%'].mean(),
        'energy_improvement_pct': ((base_error - shaded_error) / base_error * 100) if base_error > 0 else 0
    }

    return results_df, metrics
