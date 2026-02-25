"""
pv_analysis.py — Refined PV shadow analysis module.

Key features:
  1. Separated beam vs. diffuse irradiance attenuation (Beer-Lambert).
  2. Unified site configuration via SiteConfig dataclass.
  3. Robust timezone handling: inverter confirmed as fixed UTC+3.
  4. Shadow matrix smoothed once at load time.
  5. Proper nighttime handling — no shadow lookup when sun is below horizon.
  6. Sigmoid-based albedo estimation (not binary).
  7. R² included in compute_metrics (daytime-filtered).
  8. Missing-record-aware inverter loading with gap reporting.
  9. Guarded beam fraction (no negative cos(zenith)).
"""

from __future__ import annotations

import collections
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
import pytz
import tqdm
from scipy.ndimage import uniform_filter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import fmi_pv_forecaster as pvfc
except ImportError:
    pvfc = None
    warnings.warn("fmi_pv_forecaster not installed — pvfc functions will fail.")


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

@dataclass
class SiteConfig:
    """All site- and system-level parameters in one place."""

    latitude: float = 62.979849
    longitude: float = 27.648656

    tilt_deg: float = 12.0
    azimuth_deg: float = 170.0
    nominal_power_kw: float = 3.96
    system_efficiency: float = 0.95

    # Timezone of LOCAL civil time (for plotting, sun position, etc.)
    local_tz: str = "Europe/Helsinki"

    # Inverter logger timezone — FIXED UTC+3 (confirmed via sunrise analysis)
    inverter_utc_offset_hours: int = 3

    # Shadow-matrix smoothing (applied once at load time)
    window_size: Tuple[int, int] = (2, 2)

    # Data interval
    interval: str = "5min"
    interval_minutes: float = 5.0

    # Albedo: sigmoid transition parameters
    albedo_snow: float = 0.65
    albedo_bare: float = 0.20
    albedo_transition_center_c: float = 1.0
    albedo_transition_width_c: float = 3.0

    # Empirical forecast time shift (minutes, positive = shift forecast later).
    # Compensates for the spatial offset between the FMI weather station
    # and the actual PV site (~10+ km apart), which causes irradiance
    # patterns to arrive at slightly different times.
    forecast_shift_minutes: float = 0.0

    @property
    def tz(self) -> pytz.BaseTzInfo:
        return pytz.timezone(self.local_tz)

    @property
    def records_per_day(self) -> int:
        return int(24 * 60 / self.interval_minutes)


DEFAULT_CFG = SiteConfig()


# ============================================================================
# 2. SHADOW MATRIX UTILITIES
# ============================================================================

def load_and_smooth_shadow_matrix(
    path: Union[str, Path],
    window_size: Tuple[int, int] = (2, 2),
) -> np.ndarray:
    """
    Load shadow attenuation matrix CSV, apply a single smoothing pass.

    Smoothing in angular space represents the penumbra / angular uncertainty
    of shadow boundaries. Done ONCE here so downstream never repeats it.

    Parameters
    ----------
    window_size : (azimuth_window, elevation_window) in degrees.

    Returns
    -------
    np.ndarray, shape (n_alt, n_azi)
        Smoothed shadow intensity (0 = no shadow, 1 = full shadow).
    """
    df = pd.read_csv(path, index_col=0)
    raw = np.nan_to_num(df.values, nan=0.0)
    smoothed = uniform_filter(
        raw,
        size=(int(window_size[1]), int(window_size[0])),
        mode="nearest",
    )
    # Clip to [0, 1] — uniform_filter can produce small negatives at edges
    return np.clip(smoothed, 0.0, 1.0)


def compute_hemisphere_shadow_factor(shadow_matrix: np.ndarray) -> float:
    """
    Hemisphere-integrated shadow factor for isotropic diffuse irradiance.

    Weights each (altitude, azimuth) bin by cos(zenith) * sin(zenith)
    — the standard solid-angle weighting for a uniform sky.
    """
    n_alt, n_azi = shadow_matrix.shape
    altitudes_deg = np.arange(1, n_alt + 1, dtype=np.float64)
    zeniths_rad = np.deg2rad(90.0 - altitudes_deg)
    weights = np.cos(zeniths_rad) * np.sin(zeniths_rad)
    weights_2d = np.broadcast_to(weights[:, np.newaxis], shadow_matrix.shape)
    total_weight = weights_2d.sum()
    if total_weight == 0:
        return 0.0
    return float(np.sum(shadow_matrix * weights_2d) / total_weight)


# ============================================================================
# 3. CLEAR-DAY DETECTION
# ============================================================================

def find_clear_days(
    file_path: Union[str, Path],
    threshold: float = 0.8,
) -> pd.DataFrame:
    """
    Identify clear days from a clear-sky-minutes file.

    Heuristic: days with line counts above Q3 + threshold * IQR.
    threshold=0.8 is intentionally permissive (standard outlier = 1.5).
    """
    counts: dict[str, int] = collections.Counter()
    with open(file_path, "r") as f:
        for line in f:
            day = line[:10].strip()
            if day:
                counts[day] += 1

    df = pd.DataFrame(list(counts.items()), columns=["Date", "LineCount"])
    q1, q3 = df["LineCount"].quantile(0.25), df["LineCount"].quantile(0.75)
    upper_bound = q3 + threshold * (q3 - q1)

    sig = df[df["LineCount"] > upper_bound].copy()
    sig = sig.sort_values("LineCount", ascending=False)
    sig["Date"] = pd.to_datetime(sig["Date"], format="%Y %m %d").dt.date
    print(f"Found {len(sig)} clear days (threshold: Q3 + {threshold} * IQR "
          f"= {upper_bound:.0f} minutes)")
    return sig


# ============================================================================
# 4. METEOROLOGICAL DATA
# ============================================================================

def estimate_albedo(temperature_series: pd.Series, cfg: SiteConfig = DEFAULT_CFG) -> float:
    """
    Sigmoid-based albedo from daily mean temperature.

    Smoothly transitions between snow albedo and bare-ground albedo,
    more realistic than a binary switch for boreal spring/autumn.
    """
    t_mean = temperature_series.mean()
    if np.isnan(t_mean):
        return cfg.albedo_bare

    tc = cfg.albedo_transition_center_c
    tw = cfg.albedo_transition_width_c
    snow_frac = 1.0 / (1.0 + np.exp((t_mean - tc) / (tw / 4.0)))
    return round(float(snow_frac * cfg.albedo_snow + (1 - snow_frac) * cfg.albedo_bare), 3)


def _load_fmi_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load a single FMI observation CSV with UTC DatetimeIndex."""
    df = pd.read_csv(path, na_values="-")
    df["timestamp"] = pd.to_datetime(
        df[["Year", "Month", "Day"]].assign(
            hour=df["Time [UTC]"].str.split(":").str[0],
            minute=df["Time [UTC]"].str.split(":").str[1],
        )
    ).dt.tz_localize("UTC")
    return df.set_index("timestamp")


def get_extra_data(
    weather_path: Union[str, Path],
    radiation_path: Union[str, Path],
    target_date,
    cfg: SiteConfig = DEFAULT_CFG,
    cams_email: Optional[str] = None,
) -> pd.DataFrame:
    """Merge FMI ground observations with optional CAMS data for one day."""
    try:
        df_ground = pd.concat(
            [_load_fmi_csv(weather_path), _load_fmi_csv(radiation_path)], axis=1
        )
    except Exception as e:
        warnings.warn(f"Failed to load FMI data: {e}")
        return pd.DataFrame()

    df_ground = df_ground.loc[:, ~df_ground.columns.duplicated()]
    df_day = df_ground.resample(cfg.interval).mean(numeric_only=True)
    df_day = df_day[df_day.index.date == target_date].copy()

    if df_day.empty:
        warnings.warn(f"No ground data for {target_date}")
        return pd.DataFrame()

    if cams_email:
        try:
            cams_data, _ = pvlib.iotools.get_cams(
                latitude=cfg.latitude, longitude=cfg.longitude,
                start=df_day.index.min(), end=df_day.index.max(),
                email=cams_email, identifier="cams_radiation", integrated=True,
            )
            cams_aligned = cams_data.reindex(df_day.index, method="ffill")
        except Exception as e:
            warnings.warn(f"CAMS retrieval failed: {e}")
            cams_aligned = pd.DataFrame(index=df_day.index)

        df_final = pd.DataFrame(index=df_day.index)
        df_final["ghi"] = df_day.get(
            "Global radiation [W/m2]", pd.Series(dtype=float)
        ).fillna(cams_aligned.get("ghi", 0))
        df_final["dhi"] = df_day.get(
            "Diffuse radiation [W/m2]", pd.Series(dtype=float)
        ).fillna(cams_aligned.get("dhi", 0))
        df_final["dni"] = cams_aligned.get("dni", pd.Series(0.0, index=df_day.index))
        df_final["T"] = df_day.get("Air temperature [degC]", np.nan)
        df_final["wind"] = df_day.get("Wind speed [m/s]", np.nan)
        df_final["albedo"] = estimate_albedo(df_final["T"], cfg)
        return df_final[["dni", "dhi", "ghi", "T", "wind", "albedo"]]

    warnings.warn("No CAMS email provided; returning empty DataFrame.")
    return pd.DataFrame()


def load_extra_data_csv(
    path: Union[str, Path],
    cfg: SiteConfig = DEFAULT_CFG,
    recompute_albedo: bool = True,
) -> pd.DataFrame:
    """
    Load a pre-cached extra_data CSV.

    Parameters
    ----------
    recompute_albedo : bool
        If True (default), overwrite the cached albedo column with a fresh
        estimate from the sigmoid model using the temperature data in the file.
        The cached CSVs were generated with the old binary albedo (0.7 / 0.2),
        so recomputing gives the improved sigmoid-based values.
    """
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df.index.name = "time"
    df = df[["dni", "dhi", "ghi", "T", "wind", "albedo"]].round(2)

    if recompute_albedo and "T" in df.columns:
        df["albedo"] = estimate_albedo(df["T"], cfg)

    return df


# ============================================================================
# 5. INVERTER DATA LOADING
# ============================================================================

def load_inverter_data(
    excel_path: Union[str, Path],
    expected_interval_min: float = 5.0,
    tolerance_min: float = 1.0,
    center_timestamps: bool = False,
    energy_column: Optional[Union[int, str]] = None,
) -> pd.DataFrame:
    """
    Load Fronius inverter energy data and convert Wh -> W.

    File format: Fronius Symo Finnish export
      - Row 0: Finnish headers (Paivamaara ja aika, Energia, ...)
      - Row 1: Units ([dd.MM.yyyy HH:mm], [Wh], ...)
      - Data from row 2, timestamps FIXED UTC+3

    Parameters
    ----------
    energy_column : int or str, optional
        Which column contains the energy data.
        - int: column index (0-based). Default is 2 (Bank 1 / MPP1).
        - str: partial column name match from row 0 headers
          (e.g., 'MPP2' to select Bank 2).
    center_timestamps : bool
        If True, shift timestamps back by half the interval.
    """
    if energy_column is None:
        energy_column = 2

    if isinstance(energy_column, str):
        # Read header row to find column by name
        headers = pd.read_excel(excel_path, nrows=1, header=None).iloc[0]
        matches = [i for i, h in enumerate(headers) if energy_column in str(h)]
        if not matches:
            raise ValueError(
                f"No column matching '{energy_column}' found. "
                f"Available: {list(headers.values)}"
            )
        col_idx = matches[0]
        print(f"  Energy column: '{headers[col_idx]}' (index {col_idx})")
    else:
        col_idx = energy_column

    df = pd.read_excel(excel_path, usecols=[0, col_idx], skiprows=[0, 1], header=None)
    df.columns = ["Timestamp", "Energy_Wh"]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d.%m.%Y %H:%M")

    # Clean energy column
    if df["Energy_Wh"].dtype == object:
        df["Energy_Wh"] = pd.to_numeric(
            df["Energy_Wh"].astype(str).str.strip(), errors="coerce"
        )

    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    if len(df) > 1:
        deltas_min = df["Timestamp"].diff().dt.total_seconds() / 60
        dt_hours = deltas_min / 60
        median_dt_min = deltas_min.median()

        n_normal = (np.abs(deltas_min - expected_interval_min) < tolerance_min).sum()
        n_gaps = (deltas_min > expected_interval_min + tolerance_min).sum()
        n_total = len(deltas_min.dropna())

        print(f"Inverter data: {len(df):,} records, "
              f"{n_normal}/{n_total} normal intervals, {n_gaps} gaps")

        if n_gaps > 0:
            large = deltas_min[deltas_min > 60]
            if len(large) > 0:
                print(f"  Largest gaps: {large.nlargest(5).values} min")

        if abs(median_dt_min - expected_interval_min) > tolerance_min:
            warnings.warn(
                f"Median interval ({median_dt_min:.1f} min) differs from "
                f"expected ({expected_interval_min} min)."
            )

        df["Power_W"] = df["Energy_Wh"] / dt_hours
        median_dt_h = median_dt_min / 60
        df.loc[df.index[0], "Power_W"] = df["Energy_Wh"].iloc[0] / median_dt_h

        # Cap unreasonable power from tiny gaps
        max_reasonable_w = df["Energy_Wh"].max() * (60 / expected_interval_min) * 2
        df["Power_W"] = df["Power_W"].clip(upper=max_reasonable_w)
    else:
        df["Power_W"] = df["Energy_Wh"] * (60.0 / expected_interval_min)

    df["Power_W"] = df["Power_W"].fillna(0.0)

    # Center timestamps: shift back by half the interval
    if center_timestamps:
        half_interval = timedelta(minutes=expected_interval_min / 2.0)
        df["Timestamp"] = df["Timestamp"] - half_interval
        print(f"  Timestamps centered: shifted back by {expected_interval_min/2:.1f} min")

    return df


# ============================================================================
# 6. SHADOW APPLICATION — beam / diffuse separation
# ============================================================================

def apply_shadows(
    forecast_df: pd.DataFrame,
    shadow_matrix: np.ndarray,
    cfg: SiteConfig = DEFAULT_CFG,
    solpos: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Apply directional shadow attenuation with beam/diffuse separation.

    Physics:
      - Beam (DNI): attenuated by shadow factor at solar (altitude, azimuth).
        Only looked up when sun is above horizon (altitude > 0.5 deg).
      - Diffuse (DHI): attenuated by hemisphere-integrated shadow factor.
      - Ground-reflected: attenuated by diffuse shadow factor (conservative
        estimate — ground-reflected light must first pass through the canopy
        as downwelling radiation, so it sees similar obstruction as diffuse).
      - Nighttime: shadow_factor = 0, output follows forecast (which is 0).

    Critical invariant: output_shaded <= output (shadows can only reduce power).
    """
    df = forecast_df.copy()

    if solpos is None:
        solpos = pvlib.solarposition.get_solarposition(
            df.index, cfg.latitude, cfg.longitude
        )

    n_alt, n_azi = shadow_matrix.shape
    altitude = (90.0 - solpos["apparent_zenith"]).values
    azimuth = solpos["azimuth"].values

    # --- Beam shadow: only when sun is above horizon ---
    sun_up = altitude > 0.5
    beam_sf = np.zeros(len(df), dtype=np.float64)

    if sun_up.any():
        alt_idx = np.clip(np.round(altitude[sun_up]).astype(int), 0, n_alt - 1)
        azi_idx = np.clip(np.round(azimuth[sun_up]).astype(int), 0, n_azi - 1)
        beam_sf[sun_up] = np.clip(shadow_matrix[alt_idx, azi_idx], 0.0, 1.0)

    df["shadow_factor_beam"] = beam_sf

    # --- Diffuse shadow: hemisphere-integrated (constant) ---
    diffuse_sf = np.clip(compute_hemisphere_shadow_factor(shadow_matrix), 0.0, 1.0)
    df["shadow_factor_diffuse"] = diffuse_sf

    # --- Irradiance-weighted attenuation ---
    has_irradiance = all(c in df.columns for c in ("dni", "dhi", "ghi"))

    if has_irradiance:
        ghi = df["ghi"].clip(lower=1e-6)
        # Guard cos(zenith) against negative (nighttime)
        cos_zenith = np.cos(np.deg2rad(solpos["apparent_zenith"])).clip(lower=0)

        beam_fraction = ((df["dni"] * cos_zenith) / ghi).clip(0, 1).fillna(0)
        diffuse_fraction = (df["dhi"] / ghi).clip(0, 1).fillna(0)
        other_fraction = (1.0 - beam_fraction - diffuse_fraction).clip(0, 1)

        effective_transmission = (
            beam_fraction * (1.0 - beam_sf)
            + diffuse_fraction * (1.0 - diffuse_sf)
            + other_fraction * (1.0 - diffuse_sf)  # ground-reflected also obstructed
        )
        # CRITICAL: transmission can never exceed 1.0 (shadows only reduce power)
        effective_transmission = np.clip(effective_transmission, 0.0, 1.0)

        df["output_shaded"] = df["output"] * effective_transmission
    else:
        warnings.warn(
            "Irradiance columns not found — falling back to beam-only shadow."
        )
        df["output_shaded"] = df["output"] * np.clip(1.0 - beam_sf, 0.0, 1.0)

    return df


# ============================================================================
# 7. TIMEZONE HELPERS
# ============================================================================

def _utc_offset_hours(dt_naive, tz: pytz.BaseTzInfo) -> int:
    """UTC offset in whole hours for a naive date in a timezone."""
    if not isinstance(dt_naive, datetime):
        dt_naive = datetime.combine(dt_naive, datetime.min.time())
    aware = tz.localize(dt_naive)
    return int(aware.utcoffset().total_seconds() / 3600)


def _shift_index_to_local(
    df: pd.DataFrame, target_date, tz: pytz.BaseTzInfo
) -> pd.DataFrame:
    """Shift a UTC-indexed DataFrame to local naive timestamps."""
    offset = _utc_offset_hours(pd.to_datetime(target_date).date(), tz)
    df = df.copy()
    df.index = df.index + timedelta(hours=offset)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _inverter_to_local_naive(
    df: pd.DataFrame,
    target_date,
    cfg: SiteConfig,
) -> pd.DataFrame:
    """
    Convert inverter timestamps (fixed UTC+3) to local naive time.

    The inverter always logs in UTC+3 regardless of DST.
    Helsinki civil time is UTC+2 (winter) or UTC+3 (summer).

    Correction: local_naive = inverter_time + (local_offset - 3)
      - Summer (local=UTC+3): no shift
      - Winter (local=UTC+2): shift -1 hour
    """
    local_offset = _utc_offset_hours(pd.to_datetime(target_date).date(), cfg.tz)
    correction_hours = local_offset - cfg.inverter_utc_offset_hours

    df = df.copy()
    if correction_hours != 0:
        df.index = df.index + timedelta(hours=correction_hours)
    return df


# ============================================================================
# 8. MAIN PV ANALYSIS
# ============================================================================

def pv_analysis(
    target_date,
    shadow_matrix: np.ndarray,
    excel_df: pd.DataFrame,
    df_extra: pd.DataFrame,
    cfg: SiteConfig = DEFAULT_CFG,
    plot: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run PV forecast for a single day with and without shadow correction.

    Timezone flow:
      - df_extra: UTC (from FMI/CAMS)
      - Inverter data: fixed UTC+3 -> converted to local naive
      - Forecasts: computed in UTC -> shifted to local naive
      - All outputs aligned to a local-time full-day index

    Missing records:
      - Inverter gaps filled with 0.0 (no production assumed)
      - Forecast computed for full day regardless
    """
    if pvfc is None:
        raise ImportError("fmi_pv_forecaster is required.")

    target_date_obj = pd.to_datetime(target_date).date()

    # --- Inverter data: UTC+3 -> local naive ---
    day_data = excel_df[excel_df["Timestamp"].dt.date == target_date_obj].copy()

    if day_data.empty:
        warnings.warn(f"No inverter data for {target_date_obj}")

    day_data = day_data.set_index("Timestamp")
    day_data = _inverter_to_local_naive(day_data, target_date_obj, cfg)

    full_day_index = pd.date_range(
        start=f"{target_date_obj} 00:00:00",
        periods=cfg.records_per_day,
        freq=cfg.interval,
    )
    day_data = day_data.reindex(full_day_index, fill_value=0.0)

    # Report missing records
    n_present = (day_data["Power_W"] > 0).sum()
    n_expected_daylight = cfg.records_per_day // 2
    if n_present < n_expected_daylight * 0.5:
        warnings.warn(
            f"{target_date_obj}: only {n_present} nonzero records "
            f"(expected ~{n_expected_daylight} during daylight)"
        )

    # --- PV forecast (no shadows) ---
    pvfc.set_angles(cfg.tilt_deg, cfg.azimuth_deg)
    pvfc.set_location(cfg.latitude, cfg.longitude)
    pvfc.set_nominal_power_kw(cfg.nominal_power_kw)

    forecast_base = pvfc.process_radiation_df(df_extra).copy()

    # Carry irradiance columns into the forecast so apply_shadows can do
    # beam/diffuse separation.  pvfc.process_radiation_df() only outputs
    # 'output' — the irradiance is still in df_extra on the same index.
    for col in ("dni", "dhi", "ghi"):
        if col in df_extra.columns and col not in forecast_base.columns:
            forecast_base[col] = df_extra[col].reindex(forecast_base.index)

    # Solar position (computed ONCE, reused for shadows)
    solpos = pvlib.solarposition.get_solarposition(
        forecast_base.index, cfg.latitude, cfg.longitude
    )

    # --- Apply shadows ---
    forecast_windowed = apply_shadows(
        forecast_base, shadow_matrix, cfg=cfg, solpos=solpos
    )

    # --- Empirical forecast time shift ---
    # Compensates for FMI station being spatially offset from PV site.
    # Applied BEFORE UTC→local to preserve 5-min grid alignment.
    # Rounds to nearest interval to keep timestamps on-grid.
    if cfg.forecast_shift_minutes != 0.0:
        n_periods = round(cfg.forecast_shift_minutes / cfg.interval_minutes)
        actual_shift = timedelta(minutes=n_periods * cfg.interval_minutes)
        forecast_base.index = forecast_base.index + actual_shift
        forecast_windowed.index = forecast_windowed.index + actual_shift

    # --- Shift forecasts UTC -> local ---
    forecast_base = _shift_index_to_local(forecast_base, target_date_obj, cfg.tz)
    forecast_windowed = _shift_index_to_local(forecast_windowed, target_date_obj, cfg.tz)

    # --- Reindex to full day ---
    fb_n = forecast_base[["output"]].reindex(full_day_index, fill_value=0.0)
    fw_n = forecast_windowed[["output_shaded"]].reindex(full_day_index, fill_value=0.0)

    # --- System efficiency ---
    fb_n["output"] *= cfg.system_efficiency
    fw_n["output_shaded"] *= cfg.system_efficiency

    if plot:
        _plot_day_comparison(day_data, fb_n, fw_n, target_date_obj, full_day_index)

    return day_data, fb_n, fw_n


# ============================================================================
# 9. PLOTTING
# ============================================================================

def _plot_day_comparison(day_data, fb_n, fw_n, target_date_obj, full_day_index,
                        save_path=None):
    """Single-day time-series comparison plot."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(day_data.index, day_data["Power_W"],
            label="Real Power Output", color="#2ecc71", lw=1.5)
    ax.plot(fb_n.index, fb_n["output"],
            label="Forecast (No Shadows)", color="#3498db", linestyle="--")
    ax.plot(fw_n.index, fw_n["output_shaded"],
            label="Forecast (Beam+Diffuse Shadows)", color="#e67e22")

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlim(full_day_index[0], full_day_index[-1])
    ax.set_title(f"PV Comparison & Shadow Impact: {target_date_obj}")
    ax.set_xlabel("Time (Local)")
    ax.set_ylabel("Power (W)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def save_all_day_plots(
    significant_days_df: pd.DataFrame,
    shadow_matrix: np.ndarray,
    excel_df: pd.DataFrame,
    extra_data_loader: Callable,
    output_dir: str = "results/daily_plots",
    cfg: SiteConfig = DEFAULT_CFG,
):
    """
    Generate and save the daily comparison plot for every clear day.

    Parameters
    ----------
    significant_days_df : DataFrame with a 'Date' column
    shadow_matrix : loaded shadow attenuation matrix
    excel_df : inverter data
    extra_data_loader : callable(date) -> DataFrame or None
    output_dir : directory to save PNG files
    cfg : SiteConfig
    """
    os.makedirs(output_dir, exist_ok=True)

    dates = significant_days_df["Date"].tolist()
    saved = 0

    for date_obj in tqdm.tqdm(dates, desc="Saving daily plots"):
        df_extra = extra_data_loader(date_obj)
        if df_extra is None or df_extra.empty:
            continue

        day_data, fb, fw = pv_analysis(
            date_obj, shadow_matrix, excel_df, df_extra, cfg=cfg, plot=False
        )

        date_str = (date_obj.strftime("%Y-%m-%d")
                    if hasattr(date_obj, "strftime") else str(date_obj))
        save_path = os.path.join(output_dir, f"pv_comparison_{date_str}.png")

        full_day_index = day_data.index
        _plot_day_comparison(day_data, fb, fw, date_str, full_day_index,
                            save_path=save_path)
        saved += 1

    print(f"\nSaved {saved} daily plots to {output_dir}/")


def plot_real_vs_predicted_scatter(
    all_real,
    all_pred,
    title: str = "Real vs. Predicted Power Output",
    ylabel: str = "Predicted (W)",
    power_threshold: float = 50.0,
    save_path: Optional[str] = None,
):
    """Publication-quality scatter plot with R^2 trendline."""
    real_arr = np.asarray(all_real, dtype=np.float64)
    pred_arr = np.asarray(all_pred, dtype=np.float64)

    mask = (real_arr > power_threshold) | (pred_arr > power_threshold)
    if mask.sum() < 3:
        warnings.warn("Not enough daytime data points for scatter plot.")
        return

    real_f = real_arr[mask]
    pred_f = pred_arr[mask]

    r2 = r2_score(real_f, pred_f)
    z = np.polyfit(real_f, pred_f, 1)
    p = np.poly1d(z)
    sign = "+" if z[1] >= 0 else "-"
    eq_str = f"y = {z[0]:.2f}x {sign} {abs(z[1]):.1f}"

    max_val = max(real_f.max(), pred_f.max())

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(real_f, pred_f, alpha=0.25, color="#2980b9", edgecolors="none", s=30)
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.7, label="1:1 Line", lw=2)
    ax.plot(real_f, p(real_f), "#e74c3c", lw=2.5,
            label=f"Linear Fit: {eq_str}\n($R^2$ = {r2:.3f})")

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Real Power Output (W)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================================
# 10. METRICS
# ============================================================================

def compute_metrics(
    day_data: pd.DataFrame,
    forecast_base: pd.DataFrame,
    forecast_windowed: pd.DataFrame,
    interval_minutes: float = 5.0,
) -> dict:
    """
    Compute RMSE, MAE, MBE, R^2, and energy totals for one day.

    R^2 is computed on daytime-only data (> 10 W) to avoid inflating
    the score with trivial nighttime zeros.
    """
    real = day_data["Power_W"].fillna(0.0).values
    base = forecast_base["output"].fillna(0.0).values
    shaded = forecast_windowed["output_shaded"].fillna(0.0).values

    hours_per_step = interval_minutes / 60.0

    daytime = (real > 10) | (base > 10)
    r2_base = r2_score(real[daytime], base[daytime]) if daytime.sum() > 2 else np.nan
    r2_shaded = r2_score(real[daytime], shaded[daytime]) if daytime.sum() > 2 else np.nan

    return {
        "RMSE_Base": np.sqrt(mean_squared_error(real, base)),
        "RMSE_Shaded": np.sqrt(mean_squared_error(real, shaded)),
        "MAE_Base": mean_absolute_error(real, base),
        "MAE_Shaded": mean_absolute_error(real, shaded),
        "MBE_Base": float(np.mean(base - real)),
        "MBE_Shaded": float(np.mean(shaded - real)),
        "R2_Base": r2_base,
        "R2_Shaded": r2_shaded,
        "Real_Wh": float(real.sum() * hours_per_step),
        "Base_Wh": float(base.sum() * hours_per_step),
        "Shaded_Wh": float(shaded.sum() * hours_per_step),
    }


# ============================================================================
# 11. BATCH EVALUATION
# ============================================================================

def evaluate_performance(
    significant_days_df: pd.DataFrame,
    shadow_matrix: np.ndarray,
    excel_df: pd.DataFrame,
    extra_data_loader: Callable,
    cfg: SiteConfig = DEFAULT_CFG,
) -> Tuple[pd.DataFrame, list, list, list]:
    """
    Evaluate over all clear days.

    Parameters
    ----------
    extra_data_loader : callable(date) -> DataFrame or None

    Returns
    -------
    results_df : DataFrame with per-day metrics
    all_real_power : list of real power values (all days concatenated)
    all_pred_shaded : list of shadow-corrected predictions
    all_pred_base : list of base (no-shadow) predictions
    """
    daily_stats = []
    all_real_power = []
    all_pred_shaded = []
    all_pred_base = []

    for date_obj in tqdm.tqdm(
        significant_days_df["Date"].tolist(), desc="Evaluating"
    ):
        df_extra = extra_data_loader(date_obj)
        if df_extra is None or df_extra.empty:
            continue

        day_data, fb, fw = pv_analysis(
            date_obj, shadow_matrix, excel_df, df_extra, cfg=cfg, plot=False
        )
        metrics = compute_metrics(day_data, fb, fw, cfg.interval_minutes)

        all_real_power.extend(day_data["Power_W"].fillna(0.0).values)
        all_pred_shaded.extend(fw["output_shaded"].fillna(0.0).values)
        all_pred_base.extend(fb["output"].fillna(0.0).values)

        daily_stats.append({
            "Date": (date_obj.strftime("%Y-%m-%d")
                     if hasattr(date_obj, "strftime") else str(date_obj)),
            **metrics,
        })

    results_df = pd.DataFrame(daily_stats)

    if not results_df.empty:
        print("\nGenerating scatter plots...")
        plot_real_vs_predicted_scatter(
            all_real_power, all_pred_base,
            title="Real vs. FMI Base Forecast (All Clear Days)",
            ylabel="FMI Forecast — No Shadows (W)",
        )
        plot_real_vs_predicted_scatter(
            all_real_power, all_pred_shaded,
            title="Real vs. Shadow-Corrected Forecast (All Clear Days)",
            ylabel="Shadow-Corrected Forecast (W)",
        )

    return results_df, all_real_power, all_pred_shaded, all_pred_base


# ============================================================================
# 12. PERFORMANCE SUMMARY
# ============================================================================

def print_performance_summary(results_df: pd.DataFrame):
    """Pretty-print aggregate metrics."""
    if results_df.empty:
        print("No results to summarise.")
        return

    print("\n" + "=" * 60)
    print("  PV FORECAST PERFORMANCE METRICS (CLEAR DAYS)")
    print("=" * 60)

    rmse_b = results_df["RMSE_Base"].mean()
    rmse_s = results_df["RMSE_Shaded"].mean()
    mae_b = results_df["MAE_Base"].mean()
    mae_s = results_df["MAE_Shaded"].mean()
    mbe_b = results_df["MBE_Base"].mean()
    mbe_s = results_df["MBE_Shaded"].mean()

    def _pct(old, new):
        return ((old - new) / old) * 100 if old != 0 else 0.0

    print(f"\n  RMSE  Base:   {rmse_b:>8.2f} W")
    print(f"  RMSE  Shaded: {rmse_s:>8.2f} W  ({_pct(rmse_b, rmse_s):+.1f}%)")
    print(f"\n  MAE   Base:   {mae_b:>8.2f} W")
    print(f"  MAE   Shaded: {mae_s:>8.2f} W  ({_pct(mae_b, mae_s):+.1f}%)")
    print(f"\n  MBE   Base:   {mbe_b:>+8.2f} W")
    print(f"  MBE   Shaded: {mbe_s:>+8.2f} W")

    if "R2_Base" in results_df.columns:
        r2_b = results_df["R2_Base"].mean()
        r2_s = results_df["R2_Shaded"].mean()
        print(f"\n  R2    Base:   {r2_b:>8.3f}")
        print(f"  R2    Shaded: {r2_s:>8.3f}")

    print("\n  --- Energy Yield ---")
    t_real = results_df["Real_Wh"].sum() / 1000
    t_base = results_df["Base_Wh"].sum() / 1000
    t_shad = results_df["Shaded_Wh"].sum() / 1000

    def _ee(est, real):
        return ((est - real) / real) * 100 if real != 0 else 0.0

    print(f"  Real:    {t_real:>8.2f} kWh")
    print(f"  Base:    {t_base:>8.2f} kWh  ({_ee(t_base, t_real):+.1f}%)")
    print(f"  Shaded:  {t_shad:>8.2f} kWh  ({_ee(t_shad, t_real):+.1f}%)")
    print("=" * 60)


# ============================================================================
# 13. API (ALL-ECHO PENETRATION INDEX)
# ============================================================================

def calculate_api_grid(
    file_path: Union[str, Path],
    grid_size: float = 2.0,
) -> Tuple[np.ndarray, list]:
    """Calculate API = ground_returns / total_returns per XY grid cell."""
    import laspy

    print(f"Loading LiDAR data from {file_path}...")
    las = laspy.read(str(file_path))
    points = np.vstack((las.x, las.y)).T
    classes = np.array(las.classification)

    valid_mask = np.isin(classes, [2, 3, 4, 5])
    points = points[valid_mask]
    classes = classes[valid_mask]

    min_x, min_y = points[:, 0].min(), points[:, 1].min()
    max_x, max_y = points[:, 0].max(), points[:, 1].max()

    cols = max(1, int(np.ceil((max_x - min_x) / grid_size)))
    rows = max(1, int(np.ceil((max_y - min_y) / grid_size)))

    col_idx = np.clip(np.floor((points[:, 0] - min_x) / grid_size).astype(int), 0, cols - 1)
    row_idx = np.clip(np.floor((points[:, 1] - min_y) / grid_size).astype(int), 0, rows - 1)
    flat = row_idx * cols + col_idx

    total = np.bincount(flat, minlength=rows * cols)
    ground = np.bincount(flat[classes == 2], minlength=rows * cols)

    api_flat = np.zeros_like(total, dtype=np.float32)
    valid = total > 0
    api_flat[valid] = ground[valid] / total[valid]

    print(f"API grid: {cols} x {rows} at {grid_size} m")
    return api_flat.reshape(rows, cols), [min_x, max_x, min_y, max_y]


def plot_api_map(api_grid: np.ndarray, extent: list):
    """Visualise the API matrix as a heatmap."""
    plt.figure(figsize=(10, 8))
    plt.imshow(api_grid, extent=extent, origin="lower", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="All-echo Penetration Index (API)")
    plt.title("Canopy Gap Fraction (API) Map")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.tight_layout()
    plt.show()