import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import laspy
import plotly.express as px
import copy
import pvlib
from sklearn.metrics import r2_score

# import plotly.io as pio
# For standard Jupyter Notebooks
# pio.renderers.default = 'notebook'

CLASS_NAMES = {
    1: "Unclassified", 2: "Ground", 3: "Low Veg", 4: "Med Veg", 5: "High Veg",
    6: "Building", 7: "Noise", 9: "Water", 17: "Bridge Deck",
}
CLASS_COLORS = {
    "Unclassified": "lightgray", "Ground": "brown", "Low Veg": "green",
    "Med Veg": "forestgreen", "High Veg": "darkgreen", "Building": "red",
    "Noise": "black", "Water": "blue", "Bridge Deck": "orange",
}

def create_dataframe(matrix, altitude_range, azimuth_range):
    altitudes = [f"Altitude_{int(alt)}" for alt in altitude_range]
    azimuths = [f"Azimuth_{int(azi)}" for azi in azimuth_range]
    return pd.DataFrame(matrix, index=altitudes, columns=azimuths)

def plot_shadow_polar(matrix, altitude_range, azimuth_range):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    for i, altitude in enumerate(altitude_range):
        for j, azimuth in enumerate(azimuth_range):
            shadow_intensity = matrix[i, j]
            if shadow_intensity > 0:
                color = plt.cm.gray_r(shadow_intensity)
                ax.plot(np.radians(azimuth), 90 - altitude, 'o', color=color, markersize=5)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_title('Shadow Intensity Polar Plot\n(Beer–Lambert Voxel Model)\n\n')

    # Add colorbar indicating shadow intensity
    sm = plt.cm.ScalarMappable(cmap='gray_r', norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=ax, label='Shadow Intensity (1 - Transmission)')
    
    plt.show()

def plot_shadow_polar_refined(matrix_df):
    """
    Refined polar plot for shadow attenuation matrix.
    Uses pcolormesh for a continuous hemispherical representation.
    """
    # 1. Extract and clean degree values from the DataFrame headers
    # Assumes headers are 'Altitude_X' and 'Azimuth_Y'
    altitudes = np.array([int(i.split('_')[1]) for i in matrix_df.index])
    azimuths = np.array([int(c.split('_')[1]) for c in matrix_df.columns])

    # 2. Create a grid for the polar plot
    # Theta (Azimuth) and R (90 - Altitude for zenith-center)
    theta, r = np.meshgrid(np.radians(azimuths), 90 - altitudes)
    
    # fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'}, dpi=100)


    # 3. Use pcolormesh for continuous shading
    # matrix_df values represent (1 - transmittance)
    pc = ax.pcolormesh(theta, r, matrix_df.values, cmap='Greys', shading='auto', vmin=0, vmax=1, edgecolors='face')
    # levels = np.linspace(0, 1, 100)
    # pc = ax.contourf(theta, r, matrix_df.values, levels=levels, cmap='Greys', extend='both')

    # 4. Standardize for Hemispherical Photography (DHP) style
    ax.set_theta_zero_location('N') # North at the top
    ax.set_theta_direction(-1)      # Clockwise rotation
    ax.set_ylim(0, 90)              # Center is Zenith (0), outer ring is Horizon (90)
    
    # Labeling for academic clarity

    # Position y-labels at 0 degrees (North/Up)
    ax.set_rlabel_position(0) 
    
    # Set y-ticks at 10-degree intervals
    tick_positions = np.arange(10, 90, 10)
    tick_labels = [f'{90 - p}°' for p in tick_positions] # Results in 80, 70, ..., 10

    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=9, color='black')

    ax.grid(True, linestyle='--', alpha=0.5)

    # ax.set_yticklabels(['80°', '60°', '40°', '20°', '0°']) # Elevation labels
    ax.set_title('Shadow Matrix Polar Plot\n(Voxel-Based API Model)\n', 
                 va='bottom', fontsize=12)

    # 5. Add professional colorbar
    cbar = fig.colorbar(pc, ax=ax, pad=0.1, shrink=0.6, aspect=20)
    cbar.set_label('Shadow Intensity (1 - Transmittance)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.show()

def plot_shadow_polar_in(matrix, altitude_range, azimuth_range):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    for i, altitude in enumerate(altitude_range):
        for j, azimuth in enumerate(azimuth_range):
            shadow_intensity = matrix[i, j]
            if shadow_intensity > 0:
                color = plt.cm.gray_r(shadow_intensity)
                ax.plot(np.radians(azimuth), 90 - altitude, 'o', color=color, markersize=5)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_title('Shadow Intensity Polar Plot\n(Beer–Lambert Voxel Model)\n\n')

    # Add colorbar indicating shadow intensity
    sm = plt.cm.ScalarMappable(cmap='gray_r', norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=ax, label='Shadow Intensity (1 - Transmission)')
    
    return fig

def _open_las_any(path, prefer="auto"):
    if path.lower().endswith(".laz"):
        # pick a LAZ backend if available
        if prefer in ("auto", "lazrs") and getattr(laspy.LazBackend, "Lazrs", None):
            if laspy.LazBackend.Lazrs.is_available():
                return laspy.open(path, laz_backend=laspy.LazBackend.Lazrs)
        if prefer in ("auto", "laszip") and getattr(laspy.LazBackend, "Laszip", None):
            if laspy.LazBackend.Laszip.is_available():
                return laspy.open(path, laz_backend=laspy.LazBackend.Laszip)
        raise RuntimeError(
            "No LAZ backend available.\n"
            "Install: pip install 'laspy[lazrs]'  (or)  pip install laszip-python"
        )
    return laspy.open(path)

def plot_shadow_matrix_with_sunpaths(matrix_path, lat=62.9798, lon=27.6486):
    print("Loading shadow matrix...")
    df = pd.read_csv(matrix_path, index_col=0)

    azimuths = np.array([float(col.split('_')[1]) for col in df.columns])
    elevations = np.array([float(idx.split('_')[1]) for idx in df.index])

    r_grid = 90.0 - elevations
    theta_grid = np.radians(azimuths)
    Theta, R = np.meshgrid(theta_grid, r_grid)

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')  
    ax.set_theta_direction(-1)       
    ax.set_rlim(0, 90)               

    yticks = range(0, 91, 10)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{90-y}°" for y in yticks], color='black')

    cmap = copy.copy(plt.cm.gray_r)
    cmap.set_bad(color='#1e272e') # Deep slate color for missing data

    c = ax.pcolormesh(Theta, R, df.values, cmap=cmap, vmin=0, vmax=1, shading='auto')
    cbar = fig.colorbar(c, ax=ax, shrink=0.8, pad=0.1)
    cbar.set_label('Shadow Intensity (1 - Transmittance)', rotation=270, labelpad=20)

    print("Calculating seasonal sun paths...")
    tz = 'Europe/Helsinki'
    times_summer = pd.date_range('2021-06-21 00:00', '2021-06-21 23:59', freq='10min', tz=tz)
    times_equinox = pd.date_range('2021-09-21 00:00', '2021-09-21 23:59', freq='10min', tz=tz)
    times_winter = pd.date_range('2021-12-21 00:00', '2021-12-21 23:59', freq='10min', tz=tz)

    sol_summer = pvlib.solarposition.get_solarposition(times_summer, lat, lon)
    sol_equinox = pvlib.solarposition.get_solarposition(times_equinox, lat, lon)
    sol_winter = pvlib.solarposition.get_solarposition(times_winter, lat, lon)

    daylight_summer = sol_summer[sol_summer['elevation'] > 0].sort_values('azimuth')
    daylight_equinox = sol_equinox[sol_equinox['elevation'] > 0]
    daylight_winter = sol_winter[sol_winter['elevation'] > 0]

    # --- DYNAMIC HATCHING FOR UNCOMPUTED ZONE ---
    # Create an array of 360 degrees
    theta_fill_deg = np.linspace(0, 360, 360)
    theta_fill_rad = np.radians(theta_fill_deg)
    
    # Calculate the summer boundary + 2 deg buffer just like the raycaster
    max_el_fill = np.interp(theta_fill_deg, daylight_summer['azimuth'].values, daylight_summer['elevation'].values, left=0, right=0) + 2.0
    r_fill = 90 - max_el_fill # Convert elevation boundary to Zenith radius
    
    # Fill from the center (0) out to the r_fill boundary
    ax.fill_between(theta_fill_rad, 0, r_fill, 
                    color='white', edgecolor='#BFC6C4', hatch='/', 
                    alpha=1.0, label='Uncomputed Zone (Outside Sun Path)')

    # Plot the sun paths
    ax.plot(np.radians(daylight_summer['azimuth']), daylight_summer['apparent_zenith'], 
            color='#f1c40f', label='Summer Solstice (Jun 21)', linewidth=3)
    ax.plot(np.radians(daylight_equinox['azimuth']), daylight_equinox['apparent_zenith'], 
            color='#2ecc71', label='Equinox (Mar/Sep 21)', linewidth=3)
    ax.plot(np.radians(daylight_winter['azimuth']), daylight_winter['apparent_zenith'], 
            color='#3498db', label='Winter Solstice (Dec 21)', linewidth=3)

    plt.title("Shadow Matrix Polar Plot\n", fontsize=14)
    ax.legend(loc='lower left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.show()



def view_lidar_classes(
    filepath: str,
    sample: int = 200_000,
    use_hag: bool = False,
    marker_size: int = 2,
    backend: str = "auto",      # "auto" | "lazrs" | "laszip"
    sampling: str = "stride",   # "stride" | "random"
    show: bool = True,
    title: str = None,
):
    """
    Interactive Plotly viewer for LAS/LAZ with robust sampling.

    - Reads full file via laspy.open(...).read() to avoid API differences.
    - Subsamples with NumPy so x/y/z/class/HAG stay aligned.
    """
    with _open_las_any(filepath, prefer=backend) as r:
        data = r.read()  # LasData (portable across laspy 2.x)
    n = len(data.x)
    if n == 0:
        raise ValueError("File contains 0 points.")

    if sample >= n:
        idx = np.arange(n, dtype=np.int64)
    else:
        if sampling == "random":
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(n, size=sample, replace=False))
        else:  # stride
            step = max(1, n // sample)
            idx = np.arange(0, n, step, dtype=np.int64)

    x = np.asarray(data.x, dtype=np.float64)[idx]
    y = np.asarray(data.y, dtype=np.float64)[idx]
    z = np.asarray(data.z, dtype=np.float64)[idx]

    # Color by HAG if requested and present; else by class
    try:
        # data.point_format.dimension_names_lower
        has_hag = "heightaboveground" in data.point_format.dimension_names_lower
    except Exception:
        has_hag = False
    if use_hag and has_hag:
        color_name = "HAG (m)"
        color = np.asarray(data["HeightAboveGround"], dtype=np.float32)[idx]
        discrete = False
    else:
        color_name = "Class"
        cls_raw = np.asarray(data.classification, dtype=np.int32)[idx]
        color = np.array([CLASS_NAMES.get(int(c), f"Class {c}") for c in cls_raw], dtype=object)
        discrete = True

    # Assemble DataFrame (prevents Plotly arg-shape confusion)
    df = pd.DataFrame({"x": x, "y": y, "z": z, color_name: color})

    if discrete:
        # preserve legend order of appearance
        uniq = list(dict.fromkeys(df[color_name].tolist()))
        color_seq = [CLASS_COLORS.get(u, "gray") for u in uniq]
        fig = px.scatter_3d(
            df, x="x", y="y", z="z",
            color=color_name,
            color_discrete_sequence=color_seq,
            title=f"{title} — {len(df):,} pts (by class)"
        )
    else:
        fig = px.scatter_3d(
            df, x="x", y="y", z="z",
            color=color_name,
            color_continuous_scale="Viridis",
            title=f"{filepath} — {len(df):,} pts (by HAG)"
        )

    fig.update_traces(marker=dict(size=marker_size))
    fig.update_layout(
        scene=dict(aspectmode="data",
                   xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        legend=dict(itemsizing="constant"),
    )
    fig.update_layout(
    scene=dict(
        aspectmode="data",
        xaxis_title="X", yaxis_title="Y", zaxis_title="Z"
    ),
    legend=dict(itemsizing="constant"),
    width=1200,   # <-- make wider
    height=900,   # <-- make taller
    margin=dict(l=0, r=0, b=0, t=40)  # remove whitespace around plot
    )
    if show:
        fig.show()
    return fig

def plot_publication_scatter(all_real, all_pred):
    """
    Generates an academic-grade scatter plot of Real vs. Predicted Power.
    """
    real_arr = np.array(all_real)
    pred_arr = np.array(all_pred)
    
    # Filter out nighttime/zero-value data to prevent artificial R^2 inflation
    mask = (real_arr > 50) | (pred_arr > 50)
    real_filtered = real_arr[mask]
    pred_filtered = pred_arr[mask]
    
    r2 = r2_score(real_filtered, pred_filtered)
    
    # Setup the plot
    plt.figure(figsize=(9, 8))
    
    # The Scatter Data
    plt.scatter(real_filtered, pred_filtered, alpha=0.25, color='#2980b9', edgecolors='none', s=30)
    
    # The 1:1 Perfect Alignment Line
    max_val = max(real_filtered.max(), pred_filtered.max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, label='1:1 Perfect Prediction', lw=2)
    
    # The Linear Regression Trendline
    z = np.polyfit(real_filtered, pred_filtered, 1)
    p = np.poly1d(z)
    
    # Determine trendline equation string
    sign = "+" if z[1] >= 0 else "-"
    eq_str = f"y = {z[0]:.2f}x {sign} {abs(z[1]):.1f}"
    
    plt.plot(real_filtered, p(real_filtered), '#e74c3c', lw=2.5, 
             label=f'Linear Fit: {eq_str}\n($R^2$ = {r2:.3f})')
    
    # Formatting
    plt.title('Voxel-Based LiDAR Forecast vs. Real Power Output (Clear Days)', fontsize=14, pad=15)
    plt.xlabel('Real Power Output (W)', fontsize=12)
    plt.ylabel('LiDAR Shaded Forecast (W)', fontsize=12)
    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)
    
    # Move legend to the upper left to avoid covering high-power data points
    plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()