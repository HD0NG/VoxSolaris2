"""
3D LiDAR Scene Visualizer with Ray Beam
=========================================
Interactive 3D scatter plot showing:
  - Classified LiDAR points (ground, vegetation, building) color-coded
  - PV array panel positions
  - Solar beam direction at a specified local time
  - Ray paths from each panel toward the sun
  - Computed transmittance per panel and mean

Usage:
    python visualize_rays_3d.py --time "2021-07-04 08:00"
    python visualize_rays_3d.py --time "2021-07-04 17:00" --radius 50

Or import:
    from visualize_rays_3d import visualize_scene
    visualize_scene("2021-07-04 08:00", lidar_path="output/recovered.laz")
"""

import argparse
import numpy as np
import pandas as pd
import pvlib
import pytz
import laspy
import plotly.graph_objects as go
from datetime import datetime, timedelta


# ============================================================================
# CONFIG — match your project
# ============================================================================

DEFAULT_LIDAR = "output/reclassified_final_v5.laz"
DEFAULT_SHADOW_CSV = "results/shadow_matrix_results_SE_pro/shadow_attenuation_matrix_conecasting_SE_v1.csv"

LAT = 62.979849
LON = 27.648656
LOCAL_TZ = "Europe/Helsinki"
INVERTER_UTC_OFFSET = 3

# PV array: left-corner anchor
ARRAY_CORNER_XY = np.array([532882.50, 6983507.00])
ROOF_SEARCH_RADIUS = 2.0
OFFSET_FROM_ROOF = 1.5
TILT_DEG = 12
AZ_DEG = 170
PANEL_W = 1.0
PANEL_H = 1.6
ROW_CONFIG = (5, 4, 3)

# Visualization
DEFAULT_RADIUS = 40  # meters around array center to show
RAY_LENGTH = 60  # meters for ray visualization
POINT_SUBSAMPLE = 3  # show every Nth point for performance

# Classification colors
CLASS_COLORS = {
    2: ("Ground", "rgb(139,119,101)"),   # brown
    3: ("Low Veg", "rgb(144,238,144)"),  # light green
    4: ("Med Veg", "rgb(34,139,34)"),    # forest green
    5: ("High Veg", "rgb(0,100,0)"),     # dark green
    6: ("Building", "rgb(220,20,60)"),   # red
}


# ============================================================================
# PV ARRAY GEOMETRY (must match shadow_matrix_simulation.py)
# ============================================================================

def generate_pv_array_points(corner_coords, tilt_deg=TILT_DEG, az_deg=AZ_DEG,
                              panel_w=PANEL_W, panel_h=PANEL_H,
                              row_config=ROW_CONFIG):
    """Generate panel center world coords from left-corner anchor."""
    tilt_rad = np.radians(tilt_deg)
    rot_z_rad = np.radians(180 - az_deg)

    num_rows = len(row_config)
    total_height = (num_rows - 1) * panel_h

    local_points = []
    y_steps = np.linspace(total_height, 0.0, num_rows)

    for i, n_panels in enumerate(row_config):
        y = y_steps[i]
        for p in range(n_panels):
            x = p * panel_w + panel_w / 2
            local_points.append([x, y, 0.0])

    local_points = np.array(local_points)

    R_tilt = np.array([
        [1, 0, 0],
        [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
        [0, np.sin(tilt_rad),  np.cos(tilt_rad)]
    ])
    R_az = np.array([
        [np.cos(rot_z_rad), -np.sin(rot_z_rad), 0],
        [np.sin(rot_z_rad),  np.cos(rot_z_rad), 0],
        [0, 0, 1]
    ])

    rotated = (R_az @ R_tilt @ local_points.T).T
    return rotated + corner_coords


# ============================================================================
# SHADOW MATRIX LOOKUP
# ============================================================================

def load_shadow_matrix(path):
    """Load shadow matrix CSV, return as numpy array."""
    df = pd.read_csv(path, index_col=0)
    return np.clip(np.nan_to_num(df.values, nan=0.0), 0.0, 1.0)


def lookup_transmittance(shadow_matrix, altitude_deg, azimuth_deg):
    """Look up transmittance for a given solar position."""
    n_alt, n_azi = shadow_matrix.shape
    if altitude_deg < 0.5:
        return 0.0  # below horizon
    alt_i = int(np.clip(np.round(altitude_deg), 0, n_alt - 1))
    azi_i = int(np.clip(np.round(azimuth_deg), 0, n_azi - 1))
    shadow = shadow_matrix[alt_i, azi_i]
    return 1.0 - shadow  # transmittance


# ============================================================================
# MAIN VISUALIZATION
# ============================================================================

def visualize_scene(
    local_time_str,
    lidar_path=DEFAULT_LIDAR,
    shadow_csv=DEFAULT_SHADOW_CSV,
    radius=DEFAULT_RADIUS,
    ray_length=RAY_LENGTH,
    subsample=POINT_SUBSAMPLE,
):
    """
    Create an interactive 3D visualization of the LiDAR scene with
    solar beam rays and transmittance at a specific local time.

    Parameters
    ----------
    local_time_str : str
        Local time, e.g. "2021-07-04 08:00"
    lidar_path : str
        Path to .laz/.las file
    shadow_csv : str
        Path to shadow matrix CSV
    radius : float
        Meters around array center to include
    ray_length : float
        Length of ray visualization in meters
    subsample : int
        Show every Nth point (performance)
    """
    local_time = pd.Timestamp(local_time_str)
    print(f"Visualizing scene at {local_time} (local)")

    # --- Convert local → UTC for solar position ---
    tz = pytz.timezone(LOCAL_TZ)
    local_aware = tz.localize(local_time.to_pydatetime())
    utc_time = local_aware.astimezone(pytz.UTC)
    print(f"  UTC: {utc_time}")

    # --- Solar position ---
    solpos = pvlib.solarposition.get_solarposition(
        pd.DatetimeIndex([utc_time]), LAT, LON
    )
    altitude_deg = 90.0 - float(solpos["apparent_zenith"].iloc[0])
    azimuth_deg = float(solpos["azimuth"].iloc[0])
    print(f"  Solar altitude: {altitude_deg:.1f}°")
    print(f"  Solar azimuth:  {azimuth_deg:.1f}°")

    if altitude_deg < 0.5:
        print("  Sun is below horizon — no beam to visualize.")
        return

    # Sun direction vector (in local ENU: x=East, y=North, z=Up)
    el_rad = np.radians(altitude_deg)
    az_rad = np.radians(azimuth_deg)
    sun_dir = np.array([
        np.cos(el_rad) * np.sin(az_rad),  # East component
        np.cos(el_rad) * np.cos(az_rad),  # North component
        np.sin(el_rad),                    # Up component
    ])

    # --- Load LiDAR ---
    print(f"  Loading LiDAR from {lidar_path}...")
    las = laspy.read(lidar_path)
    pts = np.vstack((las.x, las.y, las.z)).T
    cls = np.array(las.classification)

    # Filter to relevant classes
    relevant = np.isin(cls, [2, 3, 4, 5, 6])
    pts = pts[relevant]
    cls = cls[relevant]

    # --- Build PV array ---
    # Find roof height
    dists = np.linalg.norm(pts[:, :2] - ARRAY_CORNER_XY, axis=1)
    roof_mask = (dists < ROOF_SEARCH_RADIUS) & (cls == 6)
    if roof_mask.any():
        target_z = np.max(pts[roof_mask, 2]) + OFFSET_FROM_ROOF
    else:
        target_z = np.median(pts[cls == 2, 2]) + 5.0
        print(f"  WARNING: No building points near array. Using z={target_z:.1f}")

    corner_3d = np.array([ARRAY_CORNER_XY[0], ARRAY_CORNER_XY[1], target_z])
    panel_points = generate_pv_array_points(corner_3d)
    array_center = panel_points.mean(axis=0)

    print(f"  Array center: ({array_center[0]:.1f}, {array_center[1]:.1f}, {array_center[2]:.1f})")
    print(f"  {len(panel_points)} panel points")

    # --- Crop scene around array ---
    dist_to_center = np.linalg.norm(pts[:, :2] - array_center[:2], axis=1)
    crop_mask = dist_to_center < radius
    pts_crop = pts[crop_mask]
    cls_crop = cls[crop_mask]

    # Subsample for performance
    pts_show = pts_crop[::subsample]
    cls_show = cls_crop[::subsample]
    print(f"  Showing {len(pts_show):,} points (of {len(pts_crop):,} within {radius}m)")

    # --- Shadow matrix lookup ---
    shadow_matrix = load_shadow_matrix(shadow_csv)
    transmittance = lookup_transmittance(shadow_matrix, altitude_deg, azimuth_deg)
    print(f"  Matrix transmittance at ({altitude_deg:.1f}°, {azimuth_deg:.1f}°): {transmittance:.3f}")
    print(f"  Shadow factor: {1 - transmittance:.3f}")

    # --- Build Plotly traces ---
    traces = []

    # LiDAR points by class
    for class_id, (label, color) in CLASS_COLORS.items():
        mask = cls_show == class_id
        if mask.sum() == 0:
            continue
        p = pts_show[mask]
        traces.append(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2],
            mode="markers",
            marker=dict(size=1.5, color=color, opacity=0.6),
            name=f"{label} ({mask.sum():,})",
            hovertemplate=f"{label}<br>x=%{{x:.1f}}<br>y=%{{y:.1f}}<br>z=%{{z:.1f}}",
        ))

    # Panel points
    traces.append(go.Scatter3d(
        x=panel_points[:, 0], y=panel_points[:, 1], z=panel_points[:, 2],
        mode="markers",
        marker=dict(size=8, color="rgb(255,215,0)", symbol="diamond",
                    line=dict(width=1, color="black")),
        name=f"PV Panels ({len(panel_points)})",
        hovertemplate="Panel<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}",
    ))

    # Ray lines from each panel toward the sun
    for i, pt in enumerate(panel_points):
        end = pt + sun_dir * ray_length
        # Color by transmittance: green=clear, red=blocked
        ray_color = f"rgb({int(255*(1-transmittance))},{int(255*transmittance)},0)"

        traces.append(go.Scatter3d(
            x=[pt[0], end[0]], y=[pt[1], end[1]], z=[pt[2], end[2]],
            mode="lines",
            line=dict(width=4, color=ray_color),
            name=f"Ray {i+1}" if i == 0 else None,
            showlegend=(i == 0),
            hovertemplate=f"Ray {i+1}<br>T={transmittance:.3f}",
        ))

    # Sun direction indicator (large arrow endpoint)
    sun_marker = array_center + sun_dir * ray_length * 1.2
    traces.append(go.Scatter3d(
        x=[sun_marker[0]], y=[sun_marker[1]], z=[sun_marker[2]],
        mode="markers+text",
        marker=dict(size=15, color="yellow", symbol="diamond",
                    line=dict(width=2, color="orange")),
        text=[f"☀ Alt={altitude_deg:.1f}° Az={azimuth_deg:.1f}°"],
        textposition="top center",
        textfont=dict(size=12, color="orange"),
        name="Sun direction",
    ))

    # --- Layout ---
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(f"3D Scene — {local_time_str} local<br>"
                  f"<sub>Alt={altitude_deg:.1f}° Az={azimuth_deg:.1f}° "
                  f"| Transmittance={transmittance:.3f} "
                  f"| Shadow={1-transmittance:.3f}</sub>"),
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="Easting (m)",
            yaxis_title="Northing (m)",
            zaxis_title="Elevation (m)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.0),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        width=1400,
        height=900,
    )

    # Save and show
    html_path = f"scene_3d_{local_time.strftime('%Y%m%d_%H%M')}.html"
    fig.write_html(html_path)
    print(f"\n  Saved interactive 3D view to {html_path}")
    fig.show()

    return fig


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D LiDAR + Ray Beam Visualizer")
    parser.add_argument("--time", required=True,
                        help="Local time, e.g. '2021-07-04 08:00'")
    parser.add_argument("--lidar", default=DEFAULT_LIDAR,
                        help="Path to LAS/LAZ file")
    parser.add_argument("--shadow", default=DEFAULT_SHADOW_CSV,
                        help="Path to shadow matrix CSV")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS,
                        help="Crop radius in meters (default 40)")
    parser.add_argument("--ray-length", type=float, default=RAY_LENGTH,
                        help="Ray visualization length in meters (default 60)")
    parser.add_argument("--subsample", type=int, default=POINT_SUBSAMPLE,
                        help="Point subsampling factor (default 3)")

    args = parser.parse_args()

    visualize_scene(
        args.time,
        lidar_path=args.lidar,
        shadow_csv=args.shadow,
        radius=args.radius,
        ray_length=args.ray_length,
        subsample=args.subsample,
    )
