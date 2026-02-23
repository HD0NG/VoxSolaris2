"""
main_analysis.py — Main notebook script for PV shadow analysis.

Uses the refactored pv_analysis module with:
  - SiteConfig with inverter_utc_offset_hours=3 (confirmed fixed UTC+3)
  - Pre-smoothed shadow matrix (done once)
  - Beam/diffuse separated shadow attenuation
  - Robust timezone handling (no magic-number offsets)
  - No duplicate evaluation loops
  - Validated inverter data loading with gap reporting
"""

# %% --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pv_analysis import (
    SiteConfig,
    find_clear_days,
    load_and_smooth_shadow_matrix,
    load_extra_data_csv,
    load_inverter_data,
    pv_analysis,
    compute_metrics,
    evaluate_performance,
    print_performance_summary,
    plot_real_vs_predicted_scatter,
    calculate_api_grid,
    plot_api_map,
)

# Project modules (unchanged)
from shadow_matrix_simulation import create_shadow_matrix

try:
    from visual_utils import plot_shadow_matrix_with_sunpaths
except ImportError:
    plot_shadow_matrix_with_sunpaths = None


# %% --- Configuration: Single Source of Truth ---
cfg = SiteConfig(
    latitude=62.979849,
    longitude=27.648656,
    tilt_deg=12.0,
    azimuth_deg=170.0,
    nominal_power_kw=3.96,
    system_efficiency=0.95,
    local_tz="Europe/Helsinki",
    inverter_utc_offset_hours=3,  # Fronius Symo logs in fixed UTC+3
    window_size=(2, 2),
    interval="5min",
    interval_minutes=5.0,
)

# --- File Paths ---
RAD_FILE       = "data/pvdata/Kuopio Savilahti 1.4.2021 - 1.10.2021_rad.csv"
TEMP_WIND_FILE = "data/pvdata/Kuopio Savilahti 1.4.2021 - 1.10.2021_temp_wind.csv"
CLEAR_MINUTES  = "data/Clear_sky_minutes_kuopio_RH16.txt"
PV_EXCEL       = "data/pvdata/pv_21.xlsx"

LIDAR_FILE     = "output/recovered.laz"
SHADOW_DIR     = "results/shadow_matrix_results_SE_pro"
SHADOW_FN      = "shadow_attenuation_matrix_conecasting_SE_v1.csv"
SHADOW_CSV     = f"{SHADOW_DIR}/{SHADOW_FN}"
EXTRA_DATA_DIR = "output"


# %% --- 1. Find Clear Days ---
clear_days = find_clear_days(CLEAR_MINUTES, threshold=0.8)
print(clear_days.head(10))


# %% --- 2. Load & Inspect Extra Data for First Clear Day ---
first_day = clear_days["Date"].iloc[0]
extra_data_df = load_extra_data_csv(f"{EXTRA_DATA_DIR}/extra_data_{first_day}.csv", cfg=cfg)
print(extra_data_df.head())


# %% --- 3. Create Shadow Matrix (ray-tracing — slow, run once) ---
# Uncomment to regenerate:
# shadow_matrix_raw = create_shadow_matrix(
#     lidar_file_path=LIDAR_FILE, voxel_size=2.0,
#     output_dir=SHADOW_DIR, output_fn=SHADOW_FN,
# )


# %% --- 4. Load & Pre-smooth Shadow Matrix (done ONCE) ---
shadow_matrix = load_and_smooth_shadow_matrix(SHADOW_CSV, window_size=cfg.window_size)
print(f"Shadow matrix shape: {shadow_matrix.shape}  (altitude x azimuth)")


# %% --- 5. Visualise Shadow Matrix ---
if plot_shadow_matrix_with_sunpaths is not None:
    plot_shadow_matrix_with_sunpaths(SHADOW_CSV)


# %% --- 6. Load Inverter Data ---
pv_df = load_inverter_data(PV_EXCEL, expected_interval_min=cfg.interval_minutes)
print(f"Range: {pv_df['Timestamp'].min()} -> {pv_df['Timestamp'].max()}")
print(pv_df.head())


# %% --- 7. Single-Day Analysis (with plot) ---
day_data, forecast_base, forecast_windowed = pv_analysis(
    target_date=first_day,
    shadow_matrix=shadow_matrix,
    excel_df=pv_df,
    df_extra=extra_data_df,
    cfg=cfg,
    plot=True,
)


# %% --- 8. Single-Day Metrics ---
metrics_single = compute_metrics(day_data, forecast_base, forecast_windowed, cfg.interval_minutes)
print(f"\nMetrics for {first_day}:")
for k, v in metrics_single.items():
    if isinstance(v, float):
        print(f"  {k:15s}: {v:10.2f}")


# %% --- 9. Scatter Plots for Single Day ---
plot_real_vs_predicted_scatter(
    day_data["Power_W"].values,
    forecast_base["output"].values,
    title=f"FMI Physical Model vs. Real ({first_day})",
    ylabel="Model Forecast (W)",
)

plot_real_vs_predicted_scatter(
    day_data["Power_W"].values,
    forecast_windowed["output_shaded"].values,
    title=f"Shadow-Corrected vs. Real ({first_day})",
    ylabel="Shaded Forecast (W)",
)


# %% --- 10. Multi-Day Batch Evaluation ---
def _load_cached_extra(date_obj):
    """Load pre-cached extra_data CSV for a date, recomputing albedo."""
    try:
        return load_extra_data_csv(
            f"{EXTRA_DATA_DIR}/extra_data_{date_obj}.csv", cfg=cfg
        )
    except FileNotFoundError:
        print(f"  Warning: No extra data for {date_obj}, skipping.")
        return None


results_df, all_real, all_pred = evaluate_performance(
    significant_days_df=clear_days,
    shadow_matrix=shadow_matrix,
    excel_df=pv_df,
    extra_data_loader=_load_cached_extra,
    cfg=cfg,
)

print_performance_summary(results_df)


# %% --- 11. Energy Yield Bar Chart ---
def plot_energy_bar_chart(results_df: pd.DataFrame):
    """Comparative bar chart: clearsky vs corrected vs real energy yield."""
    totals = results_df[["Real_Wh", "Base_Wh", "Shaded_Wh"]].sum() / 1000

    labels = ["Clearsky Model", "Shadow-Corrected", "Real Output"]
    values = [totals["Base_Wh"], totals["Shaded_Wh"], totals["Real_Wh"]]
    target = values[2]
    errors = [v - target for v in values[:2]]
    imp_pct = ((errors[0] - errors[1]) / errors[0]) * 100 if errors[0] != 0 else 0

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#e74c3c", "#3498db", "#34495e"]
    bars = ax.bar(labels, values, color=colors, width=0.6, zorder=3)

    ax.axhline(y=target, color="#2c3e50", ls="--", lw=2, label="Real", zorder=4)

    for i, bar in enumerate(bars[:2]):
        h = bar.get_height()
        err = values[i] - target
        cx = bar.get_x() + bar.get_width() / 2
        ax.annotate("", xy=(cx, target), xytext=(cx, h),
                     arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
        ax.text(cx, target + err / 2, f"+{err:.1f}\nError",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white",
                bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"))

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + max(values) * 0.02,
                f"{yval:,.1f} kWh", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_title("Shadow Correction Impact on Energy Forecast",
                 fontsize=16, pad=25, fontweight="bold")
    ax.set_ylabel("Total Energy [kWh]", fontsize=14)
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", alpha=0.3)

    info = (f"Error Reduction: {errors[0] - errors[1]:.1f} kWh\n"
            f"Improvement: {imp_pct:.1f}%")
    ax.text(0.95, 0.95, info, transform=ax.transAxes, fontsize=12,
            va="top", ha="right", fontweight="bold", color="#2980b9",
            bbox=dict(boxstyle="round", facecolor="white",
                      edgecolor="#3498db", alpha=0.9))

    plt.tight_layout()
    plt.show()


if not results_df.empty:
    plot_energy_bar_chart(results_df)


# %% --- 12. RMSE Bar Chart ---
def plot_rmse_comparison(results_df: pd.DataFrame):
    """RMSE comparison bar chart."""
    rmse_b = results_df["RMSE_Base"].mean()
    rmse_s = results_df["RMSE_Shaded"].mean()
    imp = rmse_b - rmse_s
    imp_pct = (imp / rmse_b) * 100 if rmse_b != 0 else 0

    labels = ["Clearsky Model", "Shadow-Corrected"]
    values = [rmse_b, rmse_s]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(labels, values, color=["#e74c3c", "#3498db"], width=0.5, zorder=3)

    ax.annotate("", xy=(1, values[1]), xytext=(0, values[0]),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2.5, ls="--"),
                zorder=4)

    mid_y = (values[0] + values[1]) / 2
    ax.text(0.5, mid_y + 20, f"-{imp:.1f} W\n({imp_pct:.1f}% Reduction)",
            ha="center", va="bottom", fontsize=12, fontweight="bold", color="#2c3e50",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none",
                      boxstyle="round,pad=0.3"))

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 10, f"{yval:.1f} W",
                ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_title("RMSE: Clearsky vs. Shadow-Corrected",
                 fontsize=16, pad=25, fontweight="bold")
    ax.set_ylabel("RMSE [W]", fontsize=14)
    ax.set_ylim(0, max(values) * 1.3)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("rmse_comparison_plot.png", dpi=150, bbox_inches="tight")
    plt.show()


if not results_df.empty:
    plot_rmse_comparison(results_df)


# %% --- 13. PV Array Diagram ---
def generate_array_diagram(row_config=(5, 4, 3), pw=1.0, ph=1.6):
    """Top-down footprint of the PV array."""
    import matplotlib.patches as patches

    total_h = len(row_config) * ph
    max_w = max(row_config) * pw

    fig, ax = plt.subplots(figsize=(8, 8))
    y = total_h / 2
    for r_idx, n in enumerate(row_config):
        y -= ph
        for p_idx in range(n):
            x = -max_w / 2 + p_idx * pw
            rect = patches.Rectangle(
                (x, y), pw, ph, lw=1.5,
                edgecolor="#2c3e50", facecolor="#3498db", alpha=0.8)
            ax.add_patch(rect)
            ax.text(x + pw / 2, y + ph / 2, f"R{r_idx+1}-P{p_idx+1}",
                    color="white", weight="bold", fontsize=9, ha="center", va="center")

    ax.set_xlim(-max_w / 2 - 1, max_w / 2 + 1)
    ax.set_ylim(-total_h / 2 - 1, total_h / 2 + 1)
    ax.set_aspect("equal")
    ax.set_title(
        f"PV Array: {'-'.join(map(str, row_config))}\n"
        f"Azimuth: {cfg.azimuth_deg} deg | Tilt: {cfg.tilt_deg} deg",
        fontsize=14, pad=20)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("pv_array_layout.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


generate_array_diagram()


# %% --- 14. API Map ---
api_grid, extent = calculate_api_grid(LIDAR_FILE, grid_size=2.0)
plot_api_map(api_grid, extent)