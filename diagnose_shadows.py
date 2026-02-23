"""
Shadow Application Diagnostic
===============================
Pinpoints why the shadow-corrected forecast diverges from reality.

Checks:
  1. Timezone alignment: are solar positions computed at the right UTC times?
  2. Array placement: are panel points inside/near building or vegetation voxels?
  3. Shadow lookup: do the (altitude, azimuth) indices map correctly?
  4. Morning over-attenuation: is the shadow factor too high at specific hours?
  5. Beam/diffuse split: are the fractions reasonable?

Run after loading your data:
    python diagnose_shadows.py
"""

import numpy as np
import pandas as pd
import pvlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pv_analysis_re import (
    SiteConfig, load_and_smooth_shadow_matrix, load_extra_data_csv,
    load_inverter_data, compute_hemisphere_shadow_factor, estimate_albedo,
)

# ============================================================================
# CONFIG — match your main_analysis.py exactly
# ============================================================================

cfg = SiteConfig(
    latitude=62.979849,
    longitude=27.648656,
    tilt_deg=12.0,
    azimuth_deg=170.0,
    nominal_power_kw=3.96,
    system_efficiency=0.95,
    local_tz="Europe/Helsinki",
    inverter_utc_offset_hours=3,
    window_size=(2, 2),
    interval="5min",
    interval_minutes=5.0,
)

SHADOW_CSV = "results/shadow_matrix_results_re_SE_new/shadow_attenuation_matrix_conecasting_re_SE_n2.csv"
PV_EXCEL = "data/pvdata/pv_21.xlsx"
TARGET_DATE = "2021-07-04"
EXTRA_DATA_DIR = "output"


# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("=" * 70)
print("  SHADOW APPLICATION DIAGNOSTIC")
print("=" * 70)

# Shadow matrix (raw, no smoothing — to see exact values)
raw_df = pd.read_csv(SHADOW_CSV, index_col=0)
raw_matrix = np.nan_to_num(raw_df.values, nan=0.0)
smoothed_matrix = load_and_smooth_shadow_matrix(SHADOW_CSV, window_size=cfg.window_size)

print(f"\nShadow matrix shape: {raw_matrix.shape}")
print(f"  Row labels (first 5): {list(raw_df.index[:5])}")
print(f"  Col labels (first 5): {list(raw_df.columns[:5])}")
print(f"  Value range: [{raw_matrix.min():.3f}, {raw_matrix.max():.3f}]")
print(f"  Mean shadow: {raw_matrix.mean():.3f}")
print(f"  Hemisphere-integrated diffuse SF: {compute_hemisphere_shadow_factor(smoothed_matrix):.3f}")

n_alt, n_azi = smoothed_matrix.shape
print(f"  n_alt={n_alt}, n_azi={n_azi}")


# ============================================================================
# 2. SOLAR POSITION CHECK
# ============================================================================

target_date_obj = pd.to_datetime(TARGET_DATE).date()

# UTC times for the full day at 5-min resolution
utc_times = pd.date_range(
    f"{target_date_obj} 00:00", periods=288, freq="5min", tz="UTC"
)

solpos = pvlib.solarposition.get_solarposition(utc_times, cfg.latitude, cfg.longitude)
altitude = (90.0 - solpos["apparent_zenith"]).values
azimuth = solpos["azimuth"].values

# Convert to local for display
import pytz
local_tz = pytz.timezone(cfg.local_tz)
local_times = utc_times.tz_convert(local_tz).tz_localize(None)

print(f"\n--- SOLAR POSITION for {TARGET_DATE} ---")
print(f"  Sunrise (UTC): {utc_times[altitude > 0][0] if (altitude > 0).any() else 'N/A'}")
print(f"  Sunrise (local): {local_times[altitude > 0][0] if (altitude > 0).any() else 'N/A'}")
print(f"  Solar noon altitude: {altitude.max():.1f}°")
print(f"  Solar noon azimuth: {azimuth[altitude.argmax()]:.1f}°")

# Key hours to inspect (local time)
key_hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
print(f"\n  {'Local':>7s}  {'UTC':>7s}  {'Alt':>6s}  {'Azi':>6s}  {'ShadowRaw':>10s}  {'ShadowSmooth':>12s}")
print("  " + "-" * 60)

for h in key_hours:
    # Find the index closest to this local hour
    target_local = pd.Timestamp(f"{target_date_obj} {h:02d}:00")
    idx = np.argmin(np.abs((local_times - target_local).total_seconds()))

    alt = altitude[idx]
    azi = azimuth[idx]

    if alt > 0.5:
        alt_i = int(np.clip(np.round(alt), 0, n_alt - 1))
        azi_i = int(np.clip(np.round(azi), 0, n_azi - 1))
        sf_raw = raw_matrix[alt_i, azi_i]
        sf_smooth = smoothed_matrix[alt_i, azi_i]
    else:
        alt_i, azi_i = -1, -1
        sf_raw = 0.0
        sf_smooth = 0.0

    utc_str = utc_times[idx].strftime("%H:%M")
    print(f"  {h:02d}:00    {utc_str}    {alt:6.1f}  {azi:6.1f}  "
          f"{sf_raw:10.3f}    {sf_smooth:12.3f}  "
          f"[idx: alt={alt_i}, azi={azi_i}]")


# ============================================================================
# 3. SHADOW MATRIX INDEX MAPPING CHECK
# ============================================================================

print(f"\n--- INDEX MAPPING VERIFICATION ---")
print(f"  Matrix row 0 label: '{raw_df.index[0]}' → should be Altitude_0")
print(f"  Matrix row 90 label: '{raw_df.index[-1]}' → should be Altitude_90")
print(f"  Matrix col 0 label: '{raw_df.columns[0]}' → should be Azimuth_0")

# Check if row labels are Altitude_0, Altitude_1, ...
# If they're Altitude_1 to Altitude_90, the indexing is off by one
first_label = raw_df.index[0]
if "1" in first_label and "0" not in first_label:
    print(f"  ⚠ WARNING: First row is '{first_label}', not 'Altitude_0'!")
    print(f"    This means altitude 0° has NO row → indexing is shifted by 1.")
    print(f"    A shadow lookup at altitude N maps to row N, which is actually")
    print(f"    altitude N+1's shadow. This OVER-shadows low-angle sun!")

# Check if there's an Azimuth_360 column (duplicate of Azimuth_0)
if raw_df.shape[1] == 361:
    print(f"  Note: Matrix has 361 columns (0-360, with wraparound)")
    print(f"    Azimuth index 360 maps to column 360 (= column 0)")
elif raw_df.shape[1] == 360:
    print(f"  Note: Matrix has 360 columns (0-359)")


# ============================================================================
# 4. DETAILED MORNING ANALYSIS (where over-attenuation occurs)
# ============================================================================

print(f"\n--- MORNING OVER-ATTENUATION ANALYSIS (06:00-11:00 local) ---")
print(f"  Checking shadow values at the sun's actual position each 5 min:\n")

morning_local = pd.date_range(f"{target_date_obj} 06:00", f"{target_date_obj} 11:00", freq="5min")
high_shadow_count = 0

for t_local in morning_local:
    idx = np.argmin(np.abs((local_times - t_local).total_seconds()))
    alt = altitude[idx]
    azi = azimuth[idx]

    if alt > 0.5:
        alt_i = int(np.clip(np.round(alt), 0, n_alt - 1))
        azi_i = int(np.clip(np.round(azi), 0, n_azi - 1))
        sf = smoothed_matrix[alt_i, azi_i]

        if sf > 0.3:
            high_shadow_count += 1
            if high_shadow_count <= 15:  # print first 15
                print(f"  {t_local.strftime('%H:%M')}  alt={alt:5.1f}°  azi={azi:5.1f}°  "
                      f"shadow={sf:.3f}  [idx: {alt_i},{azi_i}]")

print(f"\n  Total morning 5-min steps with shadow > 0.3: {high_shadow_count}")


# ============================================================================
# 5. BEAM/DIFFUSE FRACTION CHECK
# ============================================================================

print(f"\n--- BEAM/DIFFUSE FRACTION CHECK ---")
try:
    extra_df = load_extra_data_csv(
        f"{EXTRA_DATA_DIR}/extra_data_{target_date_obj}.csv", cfg=cfg
    )

    # Ensure index is naive
    if extra_df.index.tz is not None:
        extra_df.index = extra_df.index.tz_localize(None)

    print(f"  Extra data index tz: {extra_df.index.tz}")
    print(f"  Extra data range: {extra_df.index[0]} → {extra_df.index[-1]}")
    print(f"  Columns: {list(extra_df.columns)}")
    print()

    # Check at various UTC hours
    for h in [3, 5, 7, 9, 11, 13, 15, 17]:
        target_naive = pd.Timestamp(f"{target_date_obj} {h:02d}:00")
        deltas = (extra_df.index - target_naive).total_seconds().values
        idx = np.argmin(np.abs(deltas))
        row = extra_df.iloc[idx]
        ghi = max(float(row.get("ghi", 0)), 1e-6)
        dni = float(row.get("dni", 0))
        dhi = float(row.get("dhi", 0))

        # Solar position at this UTC time
        sp = pvlib.solarposition.get_solarposition(
            pd.DatetimeIndex([target_naive], tz="UTC"),
            cfg.latitude, cfg.longitude,
        )
        cos_z = max(0.0, float(np.cos(np.deg2rad(sp["apparent_zenith"].iloc[0]))))
        alt_deg = 90.0 - float(sp["apparent_zenith"].iloc[0])

        beam_frac = min(1.0, max(0.0, (dni * cos_z) / ghi))
        diff_frac = min(1.0, max(0.0, dhi / ghi))
        other = max(0.0, 1.0 - beam_frac - diff_frac)

        local_h = h + 3  # UTC+3 for Helsinki in summer

        print(f"  {h:02d}:00 UTC ({local_h:02d}:00 local)  "
              f"alt={alt_deg:5.1f}°  "
              f"GHI={ghi:6.0f}  DNI={dni:6.0f}  DHI={dhi:6.0f}  "
              f"beam={beam_frac:.2f}  diff={diff_frac:.2f}  other={other:.2f}")
except Exception as e:
    import traceback
    print(f"  Could not load extra data: {e}")
    traceback.print_exc()


# ============================================================================
# 6. DIAGNOSTIC PLOTS
# ============================================================================

fig, axes = plt.subplots(3, 1, figsize=(16, 15))

# Plot 1: Solar altitude & azimuth over the day
ax = axes[0]
sun_up_mask = altitude > 0
ax.plot(local_times[sun_up_mask], altitude[sun_up_mask], "orange", lw=2, label="Altitude (°)")
ax2 = ax.twinx()
ax2.plot(local_times[sun_up_mask], azimuth[sun_up_mask], "steelblue", lw=2, label="Azimuth (°)")
ax.set_ylabel("Solar Altitude (°)", color="orange")
ax2.set_ylabel("Solar Azimuth (°)", color="steelblue")
ax.set_title(f"Solar Position — {TARGET_DATE} (local time)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.grid(True, alpha=0.3)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

# Plot 2: Shadow factor over the day
ax = axes[1]
shadow_over_day = np.zeros(len(local_times))
for i in range(len(local_times)):
    if altitude[i] > 0.5:
        ai = int(np.clip(np.round(altitude[i]), 0, n_alt - 1))
        zi = int(np.clip(np.round(azimuth[i]), 0, n_azi - 1))
        shadow_over_day[i] = smoothed_matrix[ai, zi]

ax.fill_between(local_times, shadow_over_day, alpha=0.3, color="red", label="Shadow factor (beam)")
ax.axhline(y=compute_hemisphere_shadow_factor(smoothed_matrix), color="blue",
           ls="--", label=f"Diffuse SF = {compute_hemisphere_shadow_factor(smoothed_matrix):.3f}")
ax.set_ylabel("Shadow Factor (0=clear, 1=blocked)")
ax.set_title("Shadow Factor Applied Over the Day")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# Plot 3: Shadow matrix cross-section at the sun's azimuth track
ax = axes[2]
# For each hour, extract the shadow profile at that azimuth
for h in [7, 9, 12, 15, 18]:
    target_local = pd.Timestamp(f"{target_date_obj} {h:02d}:00")
    idx = np.argmin(np.abs((local_times - target_local).total_seconds()))
    azi = azimuth[idx]
    azi_i = int(np.clip(np.round(azi), 0, n_azi - 1))

    alt_range = np.arange(n_alt)
    profile = smoothed_matrix[:, azi_i]
    ax.plot(alt_range, profile, lw=2, label=f"{h:02d}:00 (azi={azi:.0f}°)")

ax.set_xlabel("Altitude index (°)")
ax.set_ylabel("Shadow intensity")
ax.set_title("Shadow Profile at Sun's Azimuth (by hour)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("shadow_diagnostic.png", dpi=150, bbox_inches="tight")
print(f"\nSaved diagnostic plot to shadow_diagnostic.png")
plt.show()

print(f"\n{'='*70}")
print("  DIAGNOSTIC COMPLETE — check the table and plots above.")
print("  Key things to look for:")
print("    - High shadow values (>0.3) during morning hours when real output is high")
print("    - Index mapping errors (Altitude_1 vs Altitude_0 as first row)")
print("    - Beam fraction near 1.0 on clear days (if <0.5, diffuse SF dominates)")
print(f"{'='*70}")