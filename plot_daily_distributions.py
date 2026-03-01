"""
plot_daily_distributions.py — Alternative statistical visualizations for per-day PV metrics.

Three visualization styles:
  1. Raincloud plot   — half-violin + box + strip (publication-quality)
  2. Paired slope     — dumbbell/slope chart (one line per day)
  3. Ridgeline        — stacked KDE curves for multiple metrics

All functions accept the results_df from evaluate_performance().

Usage:
    from plot_daily_distributions import (
        plot_raincloud, plot_paired_slope, plot_ridgeline
    )
    plot_raincloud(results_df, save_path="raincloud.png")
    plot_paired_slope(results_df, save_path="slope.png")
    plot_ridgeline(results_df, save_path="ridgeline.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats


# ── Colour palette ──────────────────────────────────────────────────────────
C_BASE    = "#e74c3c"
C_SHADED  = "#2980b9"
C_IMPROVE = "#27ae60"
C_BG      = "#f8f9fa"
C_GRID    = "#dee2e6"
C_BASE_LIGHT   = "#fadbd8"
C_SHADED_LIGHT = "#d4e6f1"


# ═══════════════════════════════════════════════════════════════════════════
# 1.  RAINCLOUD PLOT
# ═══════════════════════════════════════════════════════════════════════════

def _half_violin(ax, data, center, side="right", width=0.35, color="blue", alpha=0.4):
    """Draw a half-violin (KDE) on one side of center."""
    if len(data) < 3:
        return
    kde = stats.gaussian_kde(data, bw_method="scott")
    y_range = np.linspace(data.min() - 0.1 * np.ptp(data), data.max() + 0.1 * np.ptp(data), 200)
    density = kde(y_range)
    density = density / density.max() * width  # normalise to width

    if side == "right":
        ax.fill_betweenx(y_range, center, center + density, alpha=alpha, color=color, linewidth=0)
        ax.plot(center + density, y_range, color=color, linewidth=1.0, alpha=0.7)
    else:
        ax.fill_betweenx(y_range, center - density, center, alpha=alpha, color=color, linewidth=0)
        ax.plot(center - density, y_range, color=color, linewidth=1.0, alpha=0.7)


def _mini_box(ax, data, center, width=0.08, color="black"):
    """Draw a minimal box (IQR + median line) at center."""
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    lo = max(data.min(), q1 - 1.5 * iqr)
    hi = min(data.max(), q3 + 1.5 * iqr)

    # Whiskers
    ax.plot([center, center], [lo, q1], color=color, linewidth=1.2)
    ax.plot([center, center], [q3, hi], color=color, linewidth=1.2)
    # Box
    rect = mpatches.FancyBboxPatch(
        (center - width / 2, q1), width, iqr,
        boxstyle="round,pad=0.01", facecolor=color, alpha=0.25,
        edgecolor=color, linewidth=1.2,
    )
    ax.add_patch(rect)
    # Median
    ax.plot([center - width / 2, center + width / 2], [med, med],
            color="white", linewidth=2.5, solid_capstyle="round", zorder=6)
    ax.plot([center - width / 2, center + width / 2], [med, med],
            color=color, linewidth=1.5, solid_capstyle="round", zorder=7)


def plot_raincloud(
    results_df: pd.DataFrame,
    metrics: Optional[list] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    figsize: Optional[tuple] = None,
) -> None:
    """
    Raincloud plot: half-violin + miniature box + jittered strip.

    Parameters
    ----------
    results_df : DataFrame
        Output of evaluate_performance().
    metrics : list of str, optional
        Which metrics to plot. Default auto-detects available ones.
    """
    df = results_df.copy()

    # Auto-detect available metric pairs
    if metrics is None:
        metrics = []
        for m in ["RMSE", "MAE", "MBE"]:
            if f"{m}_Base" in df.columns and f"{m}_Shaded" in df.columns:
                metrics.append(m)
        if "R2_Base" in df.columns and "R2_Shaded" in df.columns:
            metrics.append("R2")

    n_metrics = len(metrics)
    if figsize is None:
        figsize = (5 * n_metrics, 7)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, sharey=False)
    fig.patch.set_facecolor("white")

    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.set_facecolor(C_BG)

        if metric == "R2":
            base_col, shad_col = "R2_Base", "R2_Shaded"
            ylabel = "R²"
        else:
            base_col = f"{metric}_Base"
            shad_col = f"{metric}_Shaded"
            ylabel = f"{metric} (W)"

        base_data = df[base_col].dropna().values
        shad_data = df[shad_col].dropna().values

        # Positions: baseline at x=1, corrected at x=2
        for pos, data, color in [(1, base_data, C_BASE), (2, shad_data, C_SHADED)]:
            # Half-violin on the right
            _half_violin(ax, data, pos + 0.12, side="right", width=0.35, color=color, alpha=0.35)
            # Mini box at center
            _mini_box(ax, data, pos, width=0.10, color=color)
            # Jittered strip on the left
            jitter = np.random.uniform(-0.08, -0.02, size=len(data))
            ax.scatter(
                pos + jitter, data, color=color, s=28, alpha=0.6,
                edgecolors="white", linewidth=0.4, zorder=5,
            )

        # Paired connecting lines (faint)
        n_common = min(len(base_data), len(shad_data))
        for i in range(n_common):
            ax.plot([1, 2], [base_data[i], shad_data[i]],
                    color="#bbb", linewidth=0.5, alpha=0.35, zorder=1)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Clearsky\nBaseline", "Shadow-\nCorrected"], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(metric, fontsize=14, fontweight="bold", pad=10)
        ax.grid(axis="y", color=C_GRID, linewidth=0.8, alpha=0.7)
        ax.set_xlim(0.3, 2.9)

        if metric == "MBE":
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

        # Mean ± std annotation
        mu_b, sd_b = np.mean(base_data), np.std(base_data, ddof=1)
        mu_s, sd_s = np.mean(shad_data), np.std(shad_data, ddof=1)
        dec = 3 if metric == "R2" else 1
        stats_text = (
            f"Base:  {mu_b:.{dec}f} ± {sd_b:.{dec}f}\n"
            f"Corr:  {mu_s:.{dec}f} ± {sd_s:.{dec}f}"
        )
        ax.text(
            0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor=C_GRID),
        )

    fig.suptitle(
        f"Raincloud Plot — Per-Day Metric Distributions  (n = {len(df)})",
        fontsize=16, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 2.  PAIRED SLOPE / DUMBBELL CHART
# ═══════════════════════════════════════════════════════════════════════════

def plot_paired_slope(
    results_df: pd.DataFrame,
    metric: str = "RMSE",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    figsize: tuple = (10, 9),
) -> None:
    """
    Dumbbell / slope chart: one line per day, sorted by improvement.

    Each row is one day. Left dot = baseline, right dot = corrected.
    Connecting bar coloured by improvement direction (green = better).
    Days sorted by magnitude of improvement (biggest at top).
    """
    df = results_df.copy()

    if metric == "R2":
        base_col, shad_col = "R2_Base", "R2_Shaded"
        xlabel = "R²"
        higher_is_better = True
    else:
        base_col = f"{metric}_Base"
        shad_col = f"{metric}_Shaded"
        xlabel = f"{metric} (W)"
        higher_is_better = False

    if base_col not in df.columns:
        print(f"Column '{base_col}' not found. Available: {list(df.columns)}")
        return

    df = df.dropna(subset=[base_col, shad_col]).copy()

    # Compute improvement
    if higher_is_better:
        df["improvement"] = df[shad_col] - df[base_col]
    else:
        df["improvement"] = df[base_col] - df[shad_col]

    # Sort by improvement (best at top)
    df = df.sort_values("improvement", ascending=True).reset_index(drop=True)
    n = len(df)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(C_BG)

    y_positions = np.arange(n)

    for i, row in df.iterrows():
        b_val = row[base_col]
        s_val = row[shad_col]
        imp = row["improvement"]

        # Bar colour: green if improved, red if worsened
        bar_color = C_IMPROVE if imp > 0 else C_BASE
        bar_alpha = 0.5 + 0.3 * min(abs(imp) / (df["improvement"].abs().max() + 1e-9), 1.0)

        # Connecting bar
        ax.plot([b_val, s_val], [i, i], color=bar_color, linewidth=2.5, alpha=bar_alpha, zorder=2)

        # Dots
        ax.scatter(b_val, i, color=C_BASE, s=70, edgecolors="white", linewidth=0.8, zorder=5)
        ax.scatter(s_val, i, color=C_SHADED, s=70, edgecolors="white", linewidth=0.8, zorder=5)

    # Day labels on y-axis
    date_labels = []
    for _, row in df.iterrows():
        d = row.get("Date", "")
        if isinstance(d, str) and len(d) >= 10:
            date_labels.append(d[5:])  # MM-DD
        else:
            date_labels.append(str(d)[:10] if d else "")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(date_labels, fontsize=9, family="monospace")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_title(
        f"Paired Slope Chart — {metric} per Day  (n = {n})",
        # f"Sorted by improvement (best at bottom)",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.grid(axis="x", color=C_GRID, linewidth=0.8, alpha=0.7)

    # Improvement annotation on right margin
    for i, row in df.iterrows():
        imp = row["improvement"]
        pct = (imp / abs(row[base_col])) * 100 if row[base_col] != 0 else 0
        sign = "+" if imp > 0 else ""
        color = C_IMPROVE if imp > 0 else C_BASE
        ax.text(
            1.02, i, f"{sign}{pct:.0f}%", transform=ax.get_yaxis_transform(),
            fontsize=8, va="center", ha="left", color=color, fontweight="bold",
        )

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_BASE,
                    markersize=10, label="Clearsky baseline"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SHADED,
                    markersize=10, label="Shadow-corrected"),
        plt.Line2D([0], [0], color=C_IMPROVE, linewidth=2.5, label="Improved"),
        plt.Line2D([0], [0], color=C_BASE, linewidth=2.5, label="Worsened"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

    # Summary stats box
    mu_imp = df["improvement"].mean()
    sd_imp = df["improvement"].std(ddof=1)
    n_better = (df["improvement"] > 0).sum()
    stats_text = (
        f"Mean Δ: {mu_imp:.1f} {xlabel.split(' ')[0]}\n"
        f"Std Δ:  {sd_imp:.1f}\n"
        f"Days improved: {n_better}/{n}"
    )
    # ax.text(
    #     0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
    #     va="bottom", ha="left", family="monospace",
    #     bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor=C_GRID),
    # )

    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def plot_energy_slope(
    results_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    figsize: tuple = (11, 10),
    use_absolute: bool = False,
) -> None:
    """
    Paired dumbbell chart of per-day energy forecast error (%).

    Parameters
    ----------
    results_df : DataFrame
        Must contain: Date, Real_Wh, Base_Wh, Shaded_Wh
    use_absolute : bool
        If True, plot |error| so both over- and under-prediction
        appear as positive values (closer to 0 = better).
        If False (default), plot signed error so you see the
        direction of bias.
    """
    df = results_df.copy()

    for col in ["Real_Wh", "Base_Wh", "Shaded_Wh"]:
        if col not in df.columns:
            print(f"Missing column '{col}'. Available: {list(df.columns)}")
            return

    # Per-day energy error (%)
    df["Err_Base_pct"]   = ((df["Base_Wh"]   - df["Real_Wh"]) / df["Real_Wh"]) * 100
    df["Err_Shaded_pct"] = ((df["Shaded_Wh"] - df["Real_Wh"]) / df["Real_Wh"]) * 100

    if use_absolute:
        df["Err_Base_pct"]   = df["Err_Base_pct"].abs()
        df["Err_Shaded_pct"] = df["Err_Shaded_pct"].abs()
        xlabel = "|Energy Error| (%)"
        # Improvement = reduction in absolute error
        df["improvement"] = df["Err_Base_pct"] - df["Err_Shaded_pct"]
    else:
        xlabel = "Energy Error (%)"
        # Improvement = how much closer to 0 the corrected forecast is
        df["improvement"] = df["Err_Base_pct"].abs() - df["Err_Shaded_pct"].abs()

    # Also compute absolute Wh values for annotation
    df["Real_kWh"]   = df["Real_Wh"]   / 1000
    df["Base_kWh"]   = df["Base_Wh"]   / 1000
    df["Shaded_kWh"] = df["Shaded_Wh"] / 1000

    # Sort by improvement (best at bottom)
    df = df.sort_values("improvement", ascending=True).reset_index(drop=True)
    n = len(df)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(C_BG)

    y_positions = np.arange(n)

    for i, row in df.iterrows():
        b_val = row["Err_Base_pct"]
        s_val = row["Err_Shaded_pct"]
        imp   = row["improvement"]

        # Bar colour
        bar_color = C_IMPROVE if imp > 0 else C_BASE
        bar_alpha = 0.45 + 0.35 * min(abs(imp) / (df["improvement"].abs().max() + 1e-9), 1.0)

        # Connecting bar
        ax.plot([b_val, s_val], [i, i], color=bar_color, linewidth=2.5, alpha=bar_alpha, zorder=2)

        # Dots
        ax.scatter(b_val, i, color=C_BASE, s=75, edgecolors="white", linewidth=0.8, zorder=5)
        ax.scatter(s_val, i, color=C_SHADED, s=75, edgecolors="white", linewidth=0.8, zorder=5)

    # Zero line — perfect forecast
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.5, zorder=1, label="Perfect forecast")

    # Light shading for good-accuracy zone
    if use_absolute:
        ax.axvspan(0, 5, alpha=0.08, color=C_IMPROVE, zorder=0)
        ax.text(4.8, n - 0.5, "<5%", fontsize=8, alpha=0.5, ha="right", va="top", style="italic")
        ax.set_xlim(left=-0.5)  # no negative values in absolute mode
    else:
        ax.axvspan(-5, 5, alpha=0.08, color=C_IMPROVE, zorder=0)
        ax.text(4.8, n - 0.5, "±5%", fontsize=8, alpha=0.5, ha="right", va="top", style="italic")

    # Date labels
    date_labels = []
    for _, row in df.iterrows():
        d = row.get("Date", "")
        if isinstance(d, str) and len(d) >= 10:
            date_labels.append(d[5:])  # MM-DD
        else:
            date_labels.append(str(d)[:10] if d else "")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(date_labels, fontsize=9, family="monospace")
    ax.set_xlabel(xlabel, fontsize=13)
    if use_absolute:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    else:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%+.0f%%"))
    ax.set_title(
        f"Per-Day Energy Forecast Error  (n = {n} days)",
        # f"Sorted by improvement (best at bottom)",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.grid(axis="x", color=C_GRID, linewidth=0.8, alpha=0.7)

    # Right-margin annotation: actual kWh values
    for i, row in df.iterrows():
        real = row["Real_kWh"]
        base = row["Base_kWh"]
        shad = row["Shaded_kWh"]
        ax.text(
            1.02, i,
            f"{shad:.1f} / {real:.1f} kWh",
            transform=ax.get_yaxis_transform(),
            fontsize=7.5, va="center", ha="left", color="#555", family="monospace",
        )

    # Left-margin annotation: improvement in percentage points
    for i, row in df.iterrows():
        imp = row["improvement"]
        color = C_IMPROVE if imp > 0 else C_BASE
        sign = "+" if imp > 0 else ""
        ax.text(
            -0.14, i,
            f"{sign}{imp:.1f} pp",
            transform=ax.get_yaxis_transform(),
            fontsize=8, va="center", ha="right", color=color, fontweight="bold",
        )

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_BASE,
                    markersize=10, label="Clearsky baseline"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SHADED,
                    markersize=10, label="Shadow-corrected"),
        plt.Line2D([0], [0], color=C_IMPROVE, linewidth=2.5, label="Improved (closer to 0%)"),
        plt.Line2D([0], [0], color=C_BASE, linewidth=2.5, label="Worsened"),
        plt.Line2D([0], [0], color="black", linewidth=1.5, linestyle="--", label="Perfect forecast (0%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)

    # Summary stats
    mean_base_err = df["Err_Base_pct"].mean()
    mean_shad_err = df["Err_Shaded_pct"].mean()
    mean_abs_base = df["Err_Base_pct"].abs().mean()
    mean_abs_shad = df["Err_Shaded_pct"].abs().mean()
    std_base      = df["Err_Base_pct"].std(ddof=1)
    std_shad      = df["Err_Shaded_pct"].std(ddof=1)
    n_improved    = (df["improvement"] > 0).sum()

    # Total energy
    total_real = df["Real_kWh"].sum()
    total_base = df["Base_kWh"].sum()
    total_shad = df["Shaded_kWh"].sum()
    total_base_err = ((total_base - total_real) / total_real) * 100
    total_shad_err = ((total_shad - total_real) / total_real) * 100

    stats_text = (
        f"Mean error:  Base {mean_base_err:+.1f}%  →  Corr {mean_shad_err:+.1f}%  |  "
        f"|Mean err|:  Base {mean_abs_base:.1f}%  →  Corr {mean_abs_shad:.1f}%  |  "
        f"Std:  Base {std_base:.1f}%  →  Corr {std_shad:.1f}%  |  "
        f"Days improved: {n_improved}/{n}  |  "
        f"Total: {total_real:.1f} kWh real, "
        f"{total_base:.1f} kWh base ({total_base_err:+.1f}%), "
        f"{total_shad:.1f} kWh corr ({total_shad_err:+.1f}%)"
    )

    fig.tight_layout()
    fig.subplots_adjust(left=0.18, right=0.87, bottom=0.12)
    # fig.text(
    #     0.52, 0.02, stats_text, fontsize=7.5,
    #     va="bottom", ha="center", family="monospace", wrap=True,
    #     bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.92, edgecolor=C_GRID),
    # )
    if save_path:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.show()
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# 3.  RIDGELINE / JOY PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_ridgeline(
    results_df: pd.DataFrame,
    metrics: Optional[list] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    figsize: tuple = (12, 10),
) -> None:
    """
    Ridgeline (joy plot): stacked KDE curves, one row per metric,
    overlaying Baseline vs Shadow-Corrected distributions.

    Parameters
    ----------
    metrics : list of str, optional
        Metrics to include. Default auto-detects.
    """
    df = results_df.copy()

    # Auto-detect
    if metrics is None:
        metrics = []
        for m in ["RMSE", "MAE", "MBE"]:
            if f"{m}_Base" in df.columns:
                metrics.append(m)
        if "R2_Base" in df.columns:
            metrics.append("R2")

    # Also add derived metrics
    if "RMSE_Base" in df.columns and "RMSE_Shaded" in df.columns:
        df["RMSE_Improvement"] = df["RMSE_Base"] - df["RMSE_Shaded"]
        if "RMSE_Improvement" not in metrics:
            metrics.append("RMSE_Improvement")

    if "Energy_Err_Base_pct" not in df.columns and "Real_Wh" in df.columns:
        df["Energy_Err_Base_pct"] = ((df["Base_Wh"] - df["Real_Wh"]) / df["Real_Wh"]) * 100
        df["Energy_Err_Shaded_pct"] = ((df["Shaded_Wh"] - df["Real_Wh"]) / df["Real_Wh"]) * 100

    n_rows = len(metrics)
    overlap = 0.55  # How much adjacent rows overlap

    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=False)
    fig.patch.set_facecolor("white")

    if n_rows == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.set_facecolor("white" if idx % 2 == 0 else C_BG)

        # Determine data columns
        if metric == "R2":
            base_data = df["R2_Base"].dropna().values
            shad_data = df["R2_Shaded"].dropna().values
            label = "R²"
            is_paired = True
        elif metric == "RMSE_Improvement":
            imp_data = df["RMSE_Improvement"].dropna().values
            label = "RMSE Improvement (W)"
            is_paired = False
        elif metric.endswith("_pct"):
            base_data = df[f"Energy_Err_Base_pct"].dropna().values
            shad_data = df[f"Energy_Err_Shaded_pct"].dropna().values
            label = "Energy Error (%)"
            is_paired = True
        else:
            base_data = df[f"{metric}_Base"].dropna().values
            shad_data = df[f"{metric}_Shaded"].dropna().values
            label = f"{metric} (W)"
            is_paired = True

        if is_paired:
            # KDE for both distributions
            all_vals = np.concatenate([base_data, shad_data])
            x_range = np.linspace(
                all_vals.min() - 0.15 * np.ptp(all_vals),
                all_vals.max() + 0.15 * np.ptp(all_vals),
                300,
            )

            if len(base_data) >= 3:
                kde_b = stats.gaussian_kde(base_data, bw_method="scott")
                y_base = kde_b(x_range)
                y_base = y_base / max(y_base.max(), 1e-9)
                ax.fill_between(x_range, 0, y_base, alpha=0.35, color=C_BASE, linewidth=0)
                ax.plot(x_range, y_base, color=C_BASE, linewidth=1.5, alpha=0.8)

            if len(shad_data) >= 3:
                kde_s = stats.gaussian_kde(shad_data, bw_method="scott")
                y_shad = kde_s(x_range)
                y_shad = y_shad / max(y_shad.max(), 1e-9)
                ax.fill_between(x_range, 0, y_shad, alpha=0.35, color=C_SHADED, linewidth=0)
                ax.plot(x_range, y_shad, color=C_SHADED, linewidth=1.5, alpha=0.8)

            # Rug marks
            ax.scatter(base_data, [-0.03] * len(base_data), color=C_BASE, s=15, alpha=0.6,
                       marker="|", linewidths=1.5, zorder=5)
            ax.scatter(shad_data, [-0.06] * len(shad_data), color=C_SHADED, s=15, alpha=0.6,
                       marker="|", linewidths=1.5, zorder=5)

            # Mean lines
            ax.axvline(np.mean(base_data), color=C_BASE, linewidth=1.5, linestyle="--", alpha=0.6)
            ax.axvline(np.mean(shad_data), color=C_SHADED, linewidth=1.5, linestyle="--", alpha=0.6)

        else:
            # Single distribution (e.g., improvement)
            x_range = np.linspace(
                imp_data.min() - 0.15 * np.ptp(imp_data),
                imp_data.max() + 0.15 * np.ptp(imp_data),
                300,
            )
            if len(imp_data) >= 3:
                kde = stats.gaussian_kde(imp_data, bw_method="scott")
                y_vals = kde(x_range)
                y_vals = y_vals / max(y_vals.max(), 1e-9)
                ax.fill_between(x_range, 0, y_vals, alpha=0.35, color=C_IMPROVE, linewidth=0)
                ax.plot(x_range, y_vals, color=C_IMPROVE, linewidth=1.5, alpha=0.8)

            ax.scatter(imp_data, [-0.03] * len(imp_data), color=C_IMPROVE, s=15, alpha=0.6,
                       marker="|", linewidths=1.5, zorder=5)
            ax.axvline(np.mean(imp_data), color=C_IMPROVE, linewidth=1.5, linestyle="--", alpha=0.6)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.3)

        # Styling
        ax.set_ylabel(label, fontsize=11, fontweight="bold", rotation=0, labelpad=80, va="center")
        ax.set_ylim(-0.1, 1.15)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        if idx < n_rows - 1:
            ax.set_xticklabels([])
            ax.spines["bottom"].set_alpha(0.2)
        else:
            ax.set_xlabel("Value", fontsize=12)

        ax.grid(axis="x", color=C_GRID, linewidth=0.6, alpha=0.5)

    # Global legend at top
    legend_elements = [
        mpatches.Patch(facecolor=C_BASE, alpha=0.5, label="Clearsky baseline"),
        mpatches.Patch(facecolor=C_SHADED, alpha=0.5, label="Shadow-corrected"),
        mpatches.Patch(facecolor=C_IMPROVE, alpha=0.5, label="Improvement"),
        plt.Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--", label="Mean"),
    ]
    fig.legend(
        handles=legend_elements, loc="upper center", ncol=4,
        fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.98),
    )

    fig.suptitle(
        f"Ridgeline Plot — Metric Distributions  (n = {len(df)})",
        fontsize=16, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)

    if save_path:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Integration snippet
# ═══════════════════════════════════════════════════════════════════════════

INTEGRATION_CODE = """
# %% --- Alternative Daily Distribution Plots ---
from plot_daily_distributions import plot_raincloud, plot_paired_slope, plot_ridgeline

if not results_df.empty:
    plot_raincloud(results_df, save_path="raincloud.png")
    plot_paired_slope(results_df, metric="RMSE", save_path="slope_rmse.png")
    plot_paired_slope(results_df, metric="MAE",  save_path="slope_mae.png")
    plot_ridgeline(results_df, save_path="ridgeline.png")
"""


# ═══════════════════════════════════════════════════════════════════════════
# Demo with synthetic data
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    n = 20
    dates = pd.date_range("2024-04-15", periods=n, freq="5D").strftime("%Y-%m-%d")

    demo_df = pd.DataFrame({
        "Date":         dates,
        "RMSE_Base":    np.random.normal(350, 60, n).clip(150),
        "RMSE_Shaded":  np.random.normal(280, 50, n).clip(120),
        "MAE_Base":     np.random.normal(250, 45, n).clip(100),
        "MAE_Shaded":   np.random.normal(200, 40, n).clip(80),
        "MBE_Base":     np.random.normal(120, 40, n),
        "MBE_Shaded":   np.random.normal(30, 35, n),
        "R2_Base":      np.random.normal(0.82, 0.05, n).clip(0.5, 0.99),
        "R2_Shaded":    np.random.normal(0.88, 0.04, n).clip(0.5, 0.99),
        "Real_Wh":      np.random.normal(18000, 4000, n).clip(5000),
        "Base_Wh":      np.random.normal(21000, 4500, n).clip(6000),
        "Shaded_Wh":    np.random.normal(19000, 4200, n).clip(5500),
    })

    print("Running demo with synthetic data (n=20)...")
    print(f"\nIntegration code:\n{INTEGRATION_CODE}")

    plot_raincloud(demo_df, save_path="/mnt/user-data/outputs/raincloud_demo.png")
    plot_paired_slope(demo_df, metric="RMSE", save_path="/mnt/user-data/outputs/slope_rmse_demo.png")
    plot_paired_slope(demo_df, metric="MAE", save_path="/mnt/user-data/outputs/slope_mae_demo.png")
    plot_ridgeline(demo_df, save_path="/mnt/user-data/outputs/ridgeline_demo.png")
