"""
Timezone Chain Diagnostic
==========================
Prints timestamps at every stage of the pipeline to find
where the latency/offset is introduced.

Usage: python diagnose_tz_chain.py
"""

import pandas as pd
import numpy as np
import pvlib
import pytz
from datetime import timedelta

# Adjust these imports to match your project
from pv_analysis_re import (
    SiteConfig, load_extra_data_csv, load_inverter_data,
    _shift_index_to_local, _inverter_to_local_naive,
    _utc_offset_hours,
)

try:
    import fmi_pv_forecaster as pvfc
except ImportError:
    pvfc = None
    print("WARNING: fmi_pv_forecaster not available")

# ============================================================================
# CONFIG
# ============================================================================

cfg = SiteConfig(inverter_utc_offset_hours=3, interval_minutes=5.0)
TARGET_DATE = "2021-07-04"
EXTRA_DATA_DIR = "output"
PV_EXCEL = "data/pvdata/pv_21.xlsx"

target_date_obj = pd.to_datetime(TARGET_DATE).date()

print("=" * 70)
print("  TIMEZONE CHAIN DIAGNOSTIC")
print("=" * 70)

# --- 1. Extra data timestamps ---
print("\n--- 1. EXTRA DATA (df_extra) ---")
extra_path = f"{EXTRA_DATA_DIR}/extra_data_{target_date_obj}.csv"
df_extra = load_extra_data_csv(extra_path, cfg=cfg)

print(f"  Index tz: {df_extra.index.tz}")
print(f"  First 3 timestamps: {list(df_extra.index[:3])}")
print(f"  Last 3 timestamps:  {list(df_extra.index[-3:])}")
print(f"  Index range: {df_extra.index[0]} → {df_extra.index[-1]}")

# Check: what GHI value is at midnight? If high, timestamps are local
midnight_ghi = df_extra.loc[df_extra.index.hour == 0, "ghi"]
noon_ghi = df_extra.loc[df_extra.index.hour == 12, "ghi"]
print(f"\n  GHI at hour=0:  {midnight_ghi.mean():.1f} W/m² (expect ~0 if UTC, "
      f"or daytime if local)")
print(f"  GHI at hour=12: {noon_ghi.mean():.1f} W/m² (expect high if UTC, "
      f"or past-peak if local)")

# At what hour does GHI first exceed 100?
above100 = df_extra[df_extra["ghi"] > 100]
if len(above100) > 0:
    first_sig = above100.index[0]
    print(f"  GHI first > 100 W/m² at: {first_sig} "
          f"(expect ~03:00 if UTC / ~06:00 if local)")

# --- 2. Solar position at df_extra timestamps ---
print("\n--- 2. SOLAR POSITION (pvlib) ---")
solpos = pvlib.solarposition.get_solarposition(
    df_extra.index, cfg.latitude, cfg.longitude
)
alt = 90.0 - solpos["apparent_zenith"]

# When does altitude first go positive?
sunrise_mask = alt > 0.5
if sunrise_mask.any():
    sunrise_idx = alt[sunrise_mask].index[0]
    print(f"  Sunrise (alt > 0.5°): {sunrise_idx}")
    print(f"    Alt={alt.loc[sunrise_idx]:.1f}°")
    print(f"    (Real sunrise Helsinki Jul 4: ~03:25 UTC = ~06:25 EEST)")

# Solar noon
noon_idx = alt.idxmax()
print(f"  Solar noon: {noon_idx}  Alt={alt.loc[noon_idx]:.1f}°")
print(f"    (Real solar noon Helsinki Jul 4: ~09:40 UTC = ~12:40 EEST)")

# --- 3. pvfc forecast ---
print("\n--- 3. FMI PV FORECASTER OUTPUT ---")
if pvfc:
    pvfc.set_angles(cfg.tilt_deg, cfg.azimuth_deg)
    pvfc.set_location(cfg.latitude, cfg.longitude)
    pvfc.set_nominal_power_kw(cfg.nominal_power_kw)

    forecast_base = pvfc.process_radiation_df(df_extra)
    print(f"  Output index tz: {forecast_base.index.tz}")
    print(f"  First 3 timestamps: {list(forecast_base.index[:3])}")
    print(f"  Index range: {forecast_base.index[0]} → {forecast_base.index[-1]}")

    # When does forecast output first exceed 100W?
    above100_fc = forecast_base[forecast_base["output"] > 100]
    if len(above100_fc) > 0:
        print(f"  Forecast first > 100W at: {above100_fc.index[0]}")

    # Forecast peak
    peak_idx = forecast_base["output"].idxmax()
    print(f"  Forecast peak: {peak_idx}  "
          f"output={forecast_base['output'].loc[peak_idx]:.0f}W")
else:
    print("  SKIPPED (pvfc not imported)")

# --- 4. Shift to local ---
print("\n--- 4. UTC → LOCAL SHIFT ---")
local_offset = _utc_offset_hours(target_date_obj, cfg.tz)
print(f"  Local offset: UTC+{local_offset}")
print(f"  Inverter offset: UTC+{cfg.inverter_utc_offset_hours}")
print(f"  Correction: {local_offset - cfg.inverter_utc_offset_hours} hours")

if pvfc:
    fc_shifted = _shift_index_to_local(forecast_base.copy(), target_date_obj, cfg.tz)
    print(f"  After shift — first 3: {list(fc_shifted.index[:3])}")
    print(f"  After shift — range: {fc_shifted.index[0]} → {fc_shifted.index[-1]}")

    above100_sh = fc_shifted[fc_shifted["output"] > 100]
    if len(above100_sh) > 0:
        print(f"  Shifted forecast first > 100W at: {above100_sh.index[0]}")

# --- 5. Inverter data ---
print("\n--- 5. INVERTER DATA ---")
pv_df = load_inverter_data(PV_EXCEL, expected_interval_min=cfg.interval_minutes)
day_data = pv_df[pv_df["Timestamp"].dt.date == target_date_obj].copy()
day_data = day_data.set_index("Timestamp")
print(f"  Raw index range: {day_data.index[0]} → {day_data.index[-1]}")

# Before shift
above100_inv = day_data[day_data["Power_W"] > 100]
if len(above100_inv) > 0:
    print(f"  Power first > 100W (raw): {above100_inv.index[0]}")

# After shift
day_shifted = _inverter_to_local_naive(day_data.copy(), target_date_obj, cfg)
print(f"  After local shift range: {day_shifted.index[0]} → {day_shifted.index[-1]}")
above100_inv2 = day_shifted[day_shifted["Power_W"] > 100]
if len(above100_inv2) > 0:
    print(f"  Power first > 100W (shifted): {above100_inv2.index[0]}")

# --- Summary ---
print(f"\n{'='*70}")
print("  SUMMARY — Check these for alignment:")
print(f"{'='*70}")
print(f"  Real sunrise Helsinki Jul 4:  ~03:25 UTC = ~06:25 EEST")
print(f"  Extra data GHI > 100:         {first_sig if len(above100) > 0 else 'N/A'}")
print(f"  Solar position sunrise:       {sunrise_idx if sunrise_mask.any() else 'N/A'}")
if pvfc:
    print(f"  Forecast > 100W (raw):        {above100_fc.index[0] if len(above100_fc) > 0 else 'N/A'}")
    print(f"  Forecast > 100W (shifted):    {above100_sh.index[0] if len(above100_sh) > 0 else 'N/A'}")
print(f"  Inverter > 100W (raw):        {above100_inv.index[0] if len(above100_inv) > 0 else 'N/A'}")
print(f"  Inverter > 100W (shifted):    {above100_inv2.index[0] if len(above100_inv2) > 0 else 'N/A'}")
print(f"\n  ALL timestamps above should align to ~06:00-06:30 local")
print(f"  If forecast is offset, the timezone chain has a bug.")
