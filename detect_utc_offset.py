"""
Inverter UTC Offset Detector
==============================
Determines the fixed UTC offset of a Fronius inverter logger by comparing
the daily power onset time (first nonzero energy reading) against the
theoretical sunrise time computed from pvlib for the site coordinates.

Method:
  1. For each day, find the first timestamp where power > threshold
  2. Compute true sunrise (solar elevation = 0°) for that day in UTC
  3. The systematic difference (onset_naive - sunrise_utc) reveals the
     UTC offset embedded in the logger timestamps

If the logger is in UTC+2 (EET), onset times will cluster ~2 hours
ahead of UTC sunrise. If UTC+3 (EEST), ~3 hours ahead.

Usage:
  python detect_utc_offset.py pv_21.xlsx
"""

import sys
import pandas as pd
import numpy as np
import pvlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Site coordinates
LAT = 62.9798
LON = 27.6486

# Minimum power to count as "sunrise" (filters noise / standby drain)
POWER_THRESHOLD_WH = 0.05  # 0.25 Wh is a reasonable threshold for production onset


def load_fronius_excel(file_path):
    """Load Fronius Symo Excel export (Finnish locale)."""
    df = pd.read_excel(file_path, skiprows=[1])
    ts_col = df.columns[0]
    df["Timestamp"] = pd.to_datetime(df[ts_col], format="%d.%m.%Y %H:%M")
    df = df.drop(columns=[ts_col])

    # Clean numeric columns
    for col in df.columns:
        if col == "Timestamp":
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip(), errors="coerce"
            )

    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


def detect_utc_offset(file_path, energy_col=None, power_threshold=POWER_THRESHOLD_WH):
    """
    Detect the fixed UTC offset of the inverter logger.

    Parameters
    ----------
    file_path : str
        Path to Fronius Excel file.
    energy_col : str or None
        Column name for energy. If None, auto-detects the first 'Energia' column.
    power_threshold : float
        Minimum Wh value to consider as production onset.

    Returns
    -------
    dict with detected offset and per-day analysis.
    """
    df = load_fronius_excel(file_path)

    # Auto-detect energy column
    if energy_col is None:
        candidates = [c for c in df.columns if "Energia" in c or "Energy" in c]
        # Take the first one (total energy, not MPP1/MPP2)
        candidates_total = [c for c in candidates if "MPP" not in c]
        energy_col = candidates_total[0] if candidates_total else candidates[0]

    print(f"Using energy column: '{energy_col}'")
    print(f"Site: {LAT}°N, {LON}°E")
    print(f"Power threshold: {power_threshold} Wh")

    # --- Per-day analysis ---
    df["date"] = df["Timestamp"].dt.date
    days = sorted(df["date"].unique())

    records = []

    for day in days:
        day_df = df[df["date"] == day].sort_values("Timestamp")

        # Find first timestamp with production above threshold
        producing = day_df[day_df[energy_col] > power_threshold]
        if producing.empty:
            continue  # no production this day (winter / cloudy)

        onset_naive = producing["Timestamp"].iloc[0]

        # Compute sunrise in UTC using pvlib
        # Get solar position for this day at 1-minute resolution in UTC
        day_start_utc = pd.Timestamp(day, tz="UTC")
        times_utc = pd.date_range(
            day_start_utc, day_start_utc + pd.Timedelta(hours=24),
            freq="1min"
        )
        solpos = pvlib.solarposition.get_solarposition(times_utc, LAT, LON)

        # Sunrise = first minute where elevation > 0
        above_horizon = solpos[solpos["elevation"] > 0]
        if above_horizon.empty:
            continue  # polar night

        sunrise_utc = above_horizon.index[0]

        # The onset_naive is in the logger's unknown timezone.
        # Treat it as a naive timestamp and compute the difference:
        #   offset_hours = onset_naive - sunrise_utc
        #
        # If logger is UTC+N, then:
        #   onset_utc_true = onset_naive - N hours
        #   onset_naive - sunrise_utc ≈ N hours + production_delay
        #
        # production_delay is typically 10-30 min after sunrise (inverter startup)

        onset_as_utc = onset_naive.tz_localize(None)
        sunrise_naive = sunrise_utc.tz_localize(None)

        offset_minutes = (onset_as_utc - sunrise_naive).total_seconds() / 60
        offset_hours = offset_minutes / 60

        records.append({
            "date": day,
            "onset_naive": onset_naive,
            "sunrise_utc": sunrise_naive,
            "offset_min": offset_minutes,
            "offset_hours": offset_hours,
            "onset_hour": onset_naive.hour + onset_naive.minute / 60,
            "sunrise_hour_utc": sunrise_naive.hour + sunrise_naive.minute / 60,
        })

    results_df = pd.DataFrame(records)

    if results_df.empty:
        print("No days with production found.")
        return {"offset": None, "results": results_df}

    # --- Statistical analysis ---
    offsets = results_df["offset_hours"]

    # The offset includes inverter startup delay (typically 10-30 min).
    # The true UTC offset is the dominant cluster, rounded to nearest integer.
    # Use the mode of rounded offsets for robustness.
    rounded = offsets.round().astype(int)
    offset_counts = rounded.value_counts().sort_index()

    print(f"\n{'='*60}")
    print(f"  UTC OFFSET DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"  Days analyzed: {len(results_df)}")
    print(f"  Days with production: {len(results_df)}")

    print(f"\n--- OFFSET DISTRIBUTION (onset - sunrise_UTC, rounded to hours) ---")
    for offset_val, count in offset_counts.items():
        pct = 100 * count / len(rounded)
        bar = "█" * int(pct / 2)
        print(f"  UTC+{offset_val}: {count:4d} days ({pct:5.1f}%)  {bar}")

    # The most common rounded offset minus ~0.3h startup delay
    most_common = rounded.mode().iloc[0]

    # Refine: median of the raw offsets for the most common bin
    in_bin = offsets[(rounded == most_common)]
    median_raw = in_bin.median()
    startup_delay_min = (median_raw - most_common) * 60

    print(f"\n--- CONCLUSION ---")
    print(f"  Most frequent offset bin: UTC+{most_common}")
    print(f"  Median raw offset: {median_raw:.2f} hours")
    print(f"  Implied startup delay: ~{startup_delay_min:.0f} min after sunrise")

    if most_common == 2:
        print(f"\n  ✓ Logger timezone: UTC+2 (EET — Eastern European Time)")
        print(f"    No DST switching detected.")
    elif most_common == 3:
        print(f"\n  ✓ Logger timezone: UTC+3 (EEST — Eastern European Summer Time)")
        print(f"    Or possibly Turkey Time (TRT) / Moscow Time (MSK).")
    else:
        print(f"\n  ✓ Logger timezone: UTC+{most_common}")

    # Check for seasonal split (would indicate DST)
    results_df["month"] = pd.to_datetime(
        results_df["date"].apply(str)
    ).dt.month
    winter = results_df[results_df["month"].isin([1, 2, 3, 10, 11, 12])]
    summer = results_df[results_df["month"].isin([4, 5, 6, 7, 8, 9])]

    if not winter.empty and not summer.empty:
        winter_median = winter["offset_hours"].round().median()
        summer_median = summer["offset_hours"].round().median()

        print(f"\n--- SEASONAL CHECK ---")
        print(f"  Winter months median offset: {winter_median:.0f} h")
        print(f"  Summer months median offset: {summer_median:.0f} h")

        if abs(winter_median - summer_median) >= 0.8:
            print(f"  ⚠ SEASONAL SHIFT DETECTED ({summer_median:.0f} vs {winter_median:.0f})")
            print(f"    This contradicts 'fixed UTC' — logger may switch DST!")
        else:
            print(f"  ✓ No seasonal shift — confirms FIXED UTC+{most_common}")

    print(f"{'='*60}")

    # --- Plots ---
    _plot_offset_analysis(results_df, most_common, file_path)

    return {
        "offset": most_common,
        "median_raw": median_raw,
        "startup_delay_min": startup_delay_min,
        "results": results_df,
    }


def _plot_offset_analysis(results_df, detected_offset, file_path):
    """Diagnostic plots for UTC offset detection."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    dates = pd.to_datetime(results_df["date"].apply(str))

    # Plot 1: Raw offset over time
    ax = axes[0, 0]
    ax.scatter(dates, results_df["offset_hours"], s=8, alpha=0.6, c="steelblue")
    ax.axhline(y=detected_offset, color="red", ls="--", lw=2,
               label=f"Detected: UTC+{detected_offset}")
    ax.axhline(y=2, color="orange", ls=":", alpha=0.5, label="UTC+2 (EET)")
    ax.axhline(y=3, color="green", ls=":", alpha=0.5, label="UTC+3 (EEST)")
    ax.set_ylabel("Onset − Sunrise UTC (hours)")
    ax.set_title("Daily Offset: Power Onset vs. UTC Sunrise")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, alpha=0.3)

    # Plot 2: Histogram of offsets
    ax = axes[0, 1]
    ax.hist(results_df["offset_hours"], bins=50, color="steelblue",
            edgecolor="none", alpha=0.8)
    ax.axvline(x=2, color="orange", ls="--", lw=2, label="UTC+2")
    ax.axvline(x=3, color="green", ls="--", lw=2, label="UTC+3")
    ax.set_xlabel("Offset (hours)")
    ax.set_ylabel("Count")
    ax.set_title("Offset Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Onset time vs sunrise (both in their respective frames)
    ax = axes[1, 0]
    ax.scatter(dates, results_df["sunrise_hour_utc"], s=8, alpha=0.5,
               color="orange", label="Sunrise (UTC)")
    ax.scatter(dates, results_df["onset_hour"], s=8, alpha=0.5,
               color="steelblue", label="Power onset (logger time)")

    # Show what onset would be if shifted to UTC
    onset_shifted = results_df["onset_hour"] - detected_offset
    ax.scatter(dates, onset_shifted, s=8, alpha=0.5,
               color="red", label=f"Onset shifted to UTC (−{detected_offset}h)")

    ax.set_ylabel("Hour of day")
    ax.set_title("Sunrise (UTC) vs. Power Onset")
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_ylim(0, 12)
    ax.grid(True, alpha=0.3)

    # Plot 4: Monthly box plot
    ax = axes[1, 1]
    monthly = results_df.copy()
    monthly["month"] = pd.to_datetime(monthly["date"].apply(str)).dt.month
    month_groups = [monthly[monthly["month"] == m]["offset_hours"].values
                    for m in range(1, 13)]
    # Filter empty months
    month_labels = []
    month_data = []
    for m in range(1, 13):
        data = monthly[monthly["month"] == m]["offset_hours"].values
        if len(data) > 0:
            month_data.append(data)
            month_labels.append(
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1]
            )

    if month_data:
        bp = ax.boxplot(month_data, labels=month_labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.6)
        ax.axhline(y=detected_offset, color="red", ls="--", lw=2,
                    label=f"UTC+{detected_offset}")
        ax.set_ylabel("Offset (hours)")
        ax.set_title("Monthly Offset Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"UTC Offset Analysis — {file_path}\n"
        f"Detected: UTC+{detected_offset}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_name = "utc_offset_analysis.png"
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    print(f"\nSaved diagnostic plots to {out_name}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_utc_offset.py <excel_file>")
        sys.exit(1)

    result = detect_utc_offset(sys.argv[1])