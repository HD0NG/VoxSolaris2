"""
Fronius Inverter Timestamp Gap Analyzer
========================================
Tailored for Fronius Symo export files (Finnish locale):
  - Header row 0: Finnish column names (Päivämäärä ja aika, Energia, ...)
  - Row 1: Units row ([dd.MM.yyyy HH:mm], [Wh], ...)
  - Data from row 2 onward
  - 5-minute intervals
  - Timestamps in dd.MM.yyyy HH:mm format

Detects:
  1. 1-hour gaps (spring DST: EET → EEST, clocks jump 02:00 → 03:00)
  2. Duplicate/overlapping hours (autumn DST: EEST → EET, 03:00 → 02:00)
  3. Other irregular intervals, missing data, logger restarts

Usage:
  python investigate_timezone.py pv_21.xlsx
  python investigate_timezone.py pv_21.xlsx pv_22.xlsx pv_23.xlsx
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


EXPECTED_INTERVAL_MIN = 5
EXPECTED_RECORDS_PER_DAY = 288  # 24 * 60 / 5


def load_fronius_excel(file_path):
    """
    Load a Fronius Symo Excel export.

    Skips the units row (row index 1), parses the Finnish date format,
    and strips whitespace from numeric columns.

    Returns
    -------
    pd.DataFrame with DatetimeIndex named 'Timestamp'.
    """
    df = pd.read_excel(file_path, skiprows=[1])  # skip units row

    # First column is the timestamp
    ts_col = df.columns[0]  # 'Päivämäärä ja aika'

    # Parse dd.MM.yyyy HH:mm
    df["Timestamp"] = pd.to_datetime(df[ts_col], format="%d.%m.%Y %H:%M")
    df = df.drop(columns=[ts_col])

    # Clean numeric columns (strip whitespace from string representations)
    for col in df.columns:
        if col == "Timestamp":
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")

    df = df.sort_values("Timestamp").reset_index(drop=True)

    print(f"Loaded {file_path}: {len(df):,} records")
    print(f"  Range: {df['Timestamp'].iloc[0]} → {df['Timestamp'].iloc[-1]}")
    return df


def analyze_timestamps(file_path):
    """
    Full gap analysis for a single Fronius inverter Excel file.

    Returns
    -------
    dict with structured analysis results.
    """
    df = load_fronius_excel(file_path)
    ts = df["Timestamp"].sort_values().reset_index(drop=True)

    # --- Compute all inter-record intervals ---
    deltas_td = ts.diff().iloc[1:]  # timedelta Series
    deltas_min = deltas_td.dt.total_seconds() / 60  # in minutes

    tol = 0.5  # tolerance in minutes

    # --- Classify each interval ---
    normal = np.abs(deltas_min - EXPECTED_INTERVAL_MIN) < tol
    gap_1h = np.abs(deltas_min - 60) < tol
    gap_large = deltas_min > 60 + tol
    gap_other = (~normal) & (~gap_1h) & (~gap_large) & (deltas_min > tol)
    duplicate = np.abs(deltas_min) < tol  # 0-minute gap = duplicate timestamp
    backwards = deltas_min < -tol

    # --- Report ---
    print(f"\n{'='*65}")
    print(f"  TIMESTAMP GAP ANALYSIS: {file_path}")
    print(f"{'='*65}")
    print(f"  Total records:      {len(ts):,}")
    print(f"  First:              {ts.iloc[0]}")
    print(f"  Last:               {ts.iloc[-1]}")
    print(f"  Span:               {(ts.iloc[-1] - ts.iloc[0]).days} days")
    print(f"  Expected interval:  {EXPECTED_INTERVAL_MIN} min")

    print(f"\n--- INTERVAL CLASSIFICATION ---")
    total = len(deltas_min)
    print(f"  Normal (5 min):     {normal.sum():>7,}  ({100*normal.sum()/total:.2f}%)")
    print(f"  Exactly 1-hour gap: {gap_1h.sum():>7}")
    print(f"  Gaps > 1 hour:      {gap_large.sum():>7}")
    print(f"  Other irregular:    {gap_other.sum():>7}")
    print(f"  Duplicates (0 min): {duplicate.sum():>7}")
    print(f"  Backwards jumps:    {backwards.sum():>7}")

    report = {
        "file": file_path,
        "total_records": len(ts),
        "first": ts.iloc[0],
        "last": ts.iloc[-1],
        "one_hour_gaps": [],
        "large_gaps": [],
        "other_gaps": [],
        "backwards_jumps": [],
        "duplicates_count": int(duplicate.sum()),
    }

    # --- Detail: 1-hour gaps ---
    if gap_1h.sum() > 0:
        print(f"\n--- 1-HOUR GAPS ---")
        for idx in deltas_min.index[gap_1h]:
            before = ts.iloc[idx - 1]
            after = ts.iloc[idx]
            gap = deltas_min.loc[idx]
            print(f"  {before}  →  {after}  ({gap:.0f} min)")
            report["one_hour_gaps"].append({
                "before": str(before), "after": str(after),
                "gap_min": float(gap), "date": before.date(),
            })

    # --- Detail: large gaps ---
    if gap_large.sum() > 0:
        print(f"\n--- GAPS > 1 HOUR ---")
        large_sorted = deltas_min[gap_large].sort_values(ascending=False)
        for idx in large_sorted.head(20).index:
            before = ts.iloc[idx - 1]
            after = ts.iloc[idx]
            gap = deltas_min.loc[idx]
            hours = gap / 60
            print(f"  {before}  →  {after}  ({gap:.0f} min = {hours:.1f} h)")
            report["large_gaps"].append({
                "before": str(before), "after": str(after),
                "gap_min": float(gap), "date": before.date(),
            })

    # --- Detail: other irregular ---
    if gap_other.sum() > 0:
        print(f"\n--- OTHER IRREGULAR INTERVALS ---")
        other_sorted = deltas_min[gap_other].sort_values(ascending=False)
        for idx in other_sorted.head(15).index:
            before = ts.iloc[idx - 1]
            after = ts.iloc[idx]
            gap = deltas_min.loc[idx]
            print(f"  {before}  →  {after}  ({gap:.0f} min)")

    # --- Detail: backwards ---
    if backwards.sum() > 0:
        print(f"\n--- BACKWARDS JUMPS (possible autumn fallback) ---")
        for idx in deltas_min.index[backwards]:
            before = ts.iloc[idx - 1]
            after = ts.iloc[idx]
            jump = deltas_min.loc[idx]
            print(f"  {before}  →  {after}  ({jump:.0f} min)")
            report["backwards_jumps"].append({
                "before": str(before), "after": str(after),
                "jump_min": float(jump), "date": before.date(),
            })

    # --- Cross-reference with Finnish DST dates ---
    print(f"\n--- EU/FINNISH DST TRANSITION CHECK ---")
    print(f"  (Finland: EET UTC+2 in winter, EEST UTC+3 in summer)")
    print(f"  Spring forward: last Sunday of March, 03:00 EET → 04:00 EEST")
    print(f"  Fall back:      last Sunday of October, 04:00 EEST → 03:00 EET")

    years = sorted(ts.dt.year.unique())
    for year in years:
        # Last Sunday of March
        mar31 = pd.Timestamp(year, 3, 31)
        spring_date = (mar31 - pd.Timedelta(days=(mar31.weekday() + 1) % 7)).date()

        # Last Sunday of October
        oct31 = pd.Timestamp(year, 10, 31)
        autumn_date = (oct31 - pd.Timedelta(days=(oct31.weekday() + 1) % 7)).date()

        spring_gaps = [g for g in report["one_hour_gaps"] if g["date"] == spring_date]
        autumn_anomalies = (
            [g for g in report["one_hour_gaps"] if g["date"] == autumn_date]
            + [j for j in report["backwards_jumps"] if j["date"] == autumn_date]
        )

        # Also check for duplicate timestamps around autumn transition
        autumn_ts = ts[ts.dt.date == autumn_date]
        autumn_dups = autumn_ts[autumn_ts.duplicated(keep=False)]

        # Check daily record count for autumn date
        autumn_count = len(autumn_ts)

        print(f"\n  {year}:")
        print(f"    Spring forward ({spring_date}):")
        if spring_gaps:
            for g in spring_gaps:
                print(f"      ✓ GAP: {g['before']} → {g['after']}")
                after_hour = pd.Timestamp(g["after"]).hour
                if after_hour in (3, 4):
                    print(f"        → Consistent with DST spring-forward")
        else:
            data_on_date = ts[ts.dt.date == spring_date]
            if len(data_on_date) == 0:
                print(f"      ? No data on this date")
            else:
                print(f"      ✗ No 1-hour gap ({len(data_on_date)} records present)")

        print(f"    Fall back ({autumn_date}):")
        if autumn_anomalies:
            for a in autumn_anomalies:
                print(f"      ✓ ANOMALY: {a.get('before')} → {a.get('after')}")
        elif len(autumn_dups) > 0:
            print(f"      ✓ DUPLICATE timestamps found ({len(autumn_dups)} records)")
            for t in autumn_dups.head(5):
                print(f"        {t}")
        elif autumn_count > EXPECTED_RECORDS_PER_DAY + 2:
            print(f"      ✓ EXTRA records: {autumn_count} "
                  f"(expected {EXPECTED_RECORDS_PER_DAY}) — likely repeated hour")
        else:
            if autumn_count == 0:
                print(f"      ? No data on this date")
            else:
                print(f"      ✗ No anomaly ({autumn_count} records)")

    # --- Daily record count anomalies ---
    daily_counts = ts.dt.date.value_counts().sort_index()
    short_days = daily_counts[daily_counts < EXPECTED_RECORDS_PER_DAY - 2]
    long_days = daily_counts[daily_counts > EXPECTED_RECORDS_PER_DAY + 2]

    if len(short_days) > 0:
        print(f"\n--- DAYS WITH FEWER RECORDS THAN EXPECTED ({EXPECTED_RECORDS_PER_DAY}) ---")
        for date, count in short_days.items():
            deficit = EXPECTED_RECORDS_PER_DAY - count
            print(f"  {date}: {count} records (missing ~{deficit}, ≈{deficit*5} min)")

    if len(long_days) > 0:
        print(f"\n--- DAYS WITH MORE RECORDS THAN EXPECTED ---")
        for date, count in long_days.items():
            excess = count - EXPECTED_RECORDS_PER_DAY
            print(f"  {date}: {count} records (excess ~{excess}, ≈{excess*5} min)")

    # --- Conclusion ---
    print(f"\n{'='*65}")
    print(f"  CONCLUSION")
    print(f"{'='*65}")

    dst_spring_hits = 0
    for year in years:
        mar31 = pd.Timestamp(year, 3, 31)
        expected_spring = (mar31 - pd.Timedelta(days=(mar31.weekday() + 1) % 7)).date()
        dst_spring_hits += sum(1 for g in report["one_hour_gaps"]
                               if g["date"] == expected_spring)

    if dst_spring_hits > 0:
        print(f"  → {dst_spring_hits} spring DST gap(s) on expected dates.")
        print(f"  → The Fronius logger uses LOCAL TIME (Europe/Helsinki)")
        print(f"    with automatic DST switching (EET ↔ EEST).")
        print(f"  → Winter: UTC+2 (EET) | Summer: UTC+3 (EEST)")
    elif len(report["one_hour_gaps"]) > 0:
        print(f"  → 1-hour gaps found but NOT on DST transition dates.")
        print(f"  → Likely logger downtime, not DST switching.")
        print(f"  → The logger probably uses a FIXED offset (UTC+2 or UTC+3).")
    else:
        print(f"  → No 1-hour gaps found anywhere.")
        print(f"  → The logger uses a FIXED UTC offset (no DST).")
        print(f"  → Most likely UTC+2 (EET) or UTC+3 (EEST) year-round.")
        print(f"  → To determine which: compare sunrise energy onset with")
        print(f"    known sunrise times for the site coordinates.")
    print(f"{'='*65}")

    # --- Plots ---
    _plot_diagnostics(ts, deltas_min, daily_counts, file_path)

    return report


def _plot_diagnostics(ts, deltas_min, daily_counts, file_path):
    """Generate diagnostic plots."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 13))

    # Plot 1: Interval time series (color-coded)
    ax = axes[0]
    colors = np.where(
        np.abs(deltas_min - EXPECTED_INTERVAL_MIN) < 0.5, "steelblue",
        np.where(np.abs(deltas_min - 60) < 0.5, "red", "orange")
    )
    ax.scatter(ts.iloc[1:].values, deltas_min.values, s=2, c=colors, alpha=0.5)
    ax.axhline(y=EXPECTED_INTERVAL_MIN, color="green", ls="--", alpha=0.6,
               label=f"Expected ({EXPECTED_INTERVAL_MIN} min)")
    ax.axhline(y=60, color="red", ls="--", alpha=0.6, label="1-hour gap")
    ax.set_ylabel("Interval (min)")
    ax.set_title(f"Timestamp Intervals — {file_path}")
    ax.set_ylim(-5, min(deltas_min.quantile(0.999) * 1.5, 200))
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, alpha=0.3)

    # Plot 2: Histogram (log scale)
    ax = axes[1]
    clip_max = 180
    ax.hist(deltas_min.clip(upper=clip_max), bins=150,
            color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(x=EXPECTED_INTERVAL_MIN, color="green", ls="--", lw=2,
               label=f"{EXPECTED_INTERVAL_MIN} min")
    ax.axvline(x=60, color="red", ls="--", lw=2, label="60 min")
    ax.set_xlabel("Interval (min)")
    ax.set_ylabel("Count (log)")
    ax.set_title("Interval Distribution")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Records per day
    ax = axes[2]
    dates = pd.to_datetime(pd.Series(daily_counts.index))
    ax.bar(dates, daily_counts.values, width=1, color="steelblue", alpha=0.7)
    ax.axhline(y=EXPECTED_RECORDS_PER_DAY, color="green", ls="--",
               label=f"Expected ({EXPECTED_RECORDS_PER_DAY}/day)")
    ax.axhline(y=EXPECTED_RECORDS_PER_DAY - 12, color="orange", ls=":",
               alpha=0.5, label="1-hour deficit (276)")
    ax.axhline(y=EXPECTED_RECORDS_PER_DAY + 12, color="red", ls=":",
               alpha=0.5, label="1-hour excess (300)")
    ax.set_ylabel("Records/day")
    ax.set_title("Daily Data Density")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_name = "timestamp_gap_analysis.png"
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    print(f"\nSaved diagnostic plots to {out_name}")
    plt.show()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python investigate_timezone.py <excel_file> [excel_file2 ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        print(f"\n{'#'*70}")
        print(f"# Analyzing: {path}")
        print(f"{'#'*70}")
        analyze_timestamps(path)