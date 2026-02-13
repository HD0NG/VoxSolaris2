import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import laspy
import plotly.express as px

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