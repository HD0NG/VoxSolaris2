"""
Voxel-Based Ray-Tracing Shadow Matrix Generator — BANK 2 (West, 260°)
======================================================================
Based on shadow_matrix_simulation_re.py (skip_dist version).

Roof geometry (annotated aerial image):
  The building has two roof sections separated by a hinge line running
  approximately NW–SE.

  BANK 1 — SE-facing (separate script)
    Azimuth: 170°   Tilt: 12°   Panels: 12   Nominal: 3 960 W
    Layout: (5, 4, 3) left-aligned
    Panels: 1-1…1-5 (top), 1-6…1-9 (mid), 1-10…1-12 (bot)

  BANK 2 — West-facing (this script)
    Azimuth: 260°   Tilt: 20°   Panels: 14   Nominal: 4 620 W
    Inverter channel: MPP2

    Part 1 — North sub-array (8 panels)
      Layout: (6, 2) left-aligned
      Anchor corner: (532882.93, 6983518.73) — panel 2-1-1 at (532884.50, 6983518.50)
      Row 0: 2-1-1  2-1-2  2-1-3  2-1-4  2-1-5  2-1-6
      Row 1: 2-1-7  2-1-8

    Part 2 — South sub-array (6 panels)
      Layout: (4, 2) right-aligned
      Anchor corner: (532888.63, 6983503.79) — panel 2-2-1 at (532889.50, 6983507.50)
      Row 0: 2-2-1  2-2-2  2-2-3  2-2-4
      Row 1: 2-2-5  2-2-6

    Note: at 260° azimuth, local +x maps to ≈ north, so "rows" in the
    code appear as vertical columns in the aerial view.

Two separate shadow matrices are generated (scene loaded once):
  - ..._W_north_v1.csv  (panels 2-1-1 … 2-1-8)
  - ..._W_south_v1.csv  (panels 2-2-1 … 2-2-6)
"""

import os
import time
import math
import laspy
import numpy as np
import pandas as pd
from numba import njit, prange

# ============================================================================
# CONFIGURATION (Boreal Calibration)
# ============================================================================

BETA_FINLAND = 2.08
K_BASE_FINLAND = 1.0 / BETA_FINLAND  # ≈ 0.481
OMEGA_S = 0.56  # retained for reference, NOT used

# --- Bank 2 sub-array corners ---
NORTH_CORNER_2D = np.array([532882.93, 6983518.73])  # Part 1 corner — derived from panel 2-1-1 at (532884.50, 6983518.50)
SOUTH_CORNER_2D = np.array([532888.63, 6983503.79])  # Part 2 corner — derived from panel 2-2-1 at (532889.50, 6983507.50)

ROOF_SEARCH_RADIUS = 3.0
OFFSET_FROM_ROOF = -0.5
SKIP_DISTANCE = 3.0  # meters to skip building/ground voxels near origin

GROUND_CLASS = 2
BUILDING_CLASS = 6
VEGETATION_CLASSES = {3, 4, 5}
RELEVANT_CLASSES = {2, 3, 4, 5, 6}
CLASS_PRIORITY = {6: 5, 5: 4, 4: 3, 3: 2, 2: 1, 0: 0}

AZIMUTH_STEPS = 360
ELEVATION_STEPS = 91
SOLAR_ANGULAR_RADIUS_DEG = 0.265
NUM_RAYS_PER_CONE = 6


# ============================================================================
# MODULE 1: DATA INGESTOR
# ============================================================================

def load_and_prepare_lidar(file_path, relevant_classes=None):
    """Load a LAS/LAZ file, filter to relevant classes."""
    print(f"Loading LiDAR data from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    las = laspy.read(file_path)
    points_xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    classifications = np.array(las.classification, dtype=np.int8)

    if relevant_classes:
        mask = np.isin(classifications, list(relevant_classes))
        points_xyz = points_xyz[mask]
        classifications = classifications[mask]

    print(f"  {len(points_xyz):,} points after filtering.")
    return points_xyz, classifications


# ============================================================================
# MODULE 2: VOXELIZER & GEOMETRY
# ============================================================================

def voxelize_scene(points, classifications, voxel_size, beta=BETA_FINLAND):
    """
    Build class_grid (int8) and dens_grid (float32, leaf-area density m²/m³)
    from a classified point cloud.
    """
    scene_min = np.min(points, axis=0)
    scene_max = np.max(points, axis=0)
    grid_dims = np.ceil((scene_max - scene_min) / voxel_size).astype(np.int32)
    grid_dims = np.maximum(grid_dims, 1)
    nx, ny, nz = grid_dims

    voxel_indices = np.clip(
        np.floor((points - scene_min) / voxel_size).astype(np.int32),
        0, grid_dims - 1
    )
    vi_x, vi_y, vi_z = voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]

    class_grid = _assign_class_priority(
        vi_x, vi_y, vi_z, classifications, nx, ny, nz
    )

    ground_mask = classifications == GROUND_CLASS
    veg_mask = (classifications == 3) | (classifications == 4) | (classifications == 5)

    flat_2d = vi_x.astype(np.int64) * ny + vi_y.astype(np.int64)

    total_counts = np.zeros(nx * ny, dtype=np.int32)
    np.add.at(total_counts, flat_2d, 1)
    ground_counts = np.zeros(nx * ny, dtype=np.int32)
    np.add.at(ground_counts, flat_2d[ground_mask], 1)
    veg_counts_2d = np.zeros(nx * ny, dtype=np.int32)
    np.add.at(veg_counts_2d, flat_2d[veg_mask], 1)

    total_counts = total_counts.reshape(nx, ny)
    ground_counts = ground_counts.reshape(nx, ny)
    veg_counts_2d = veg_counts_2d.reshape(nx, ny)

    flat_3d_veg = (
        vi_x[veg_mask].astype(np.int64) * ny * nz
        + vi_y[veg_mask].astype(np.int64) * nz
        + vi_z[veg_mask].astype(np.int64)
    )
    veg_counts_3d = np.zeros(nx * ny * nz, dtype=np.float32)
    np.add.at(veg_counts_3d, flat_3d_veg, 1.0)
    dens_grid = veg_counts_3d.reshape(nx, ny, nz)

    api_grid = np.ones((nx, ny), dtype=np.float32)
    valid = total_counts > 0
    api_grid[valid] = ground_counts[valid] / total_counts[valid]
    api_grid = np.clip(api_grid, 0.01, 1.0)
    column_lai_e = -beta * np.log(api_grid)

    safe_veg = np.where(veg_counts_2d > 0, veg_counts_2d, 1).astype(np.float32)
    for iz_idx in range(nz):
        dens_grid[:, :, iz_idx] = (
            (dens_grid[:, :, iz_idx] / safe_veg) * (column_lai_e / voxel_size)
        )
    no_veg = veg_counts_2d == 0
    dens_grid[no_veg, :] = 0.0

    _fill_below_surface(class_grid, GROUND_CLASS, CLASS_PRIORITY)
    _fill_below_surface(class_grid, BUILDING_CLASS, CLASS_PRIORITY)

    print(f"  Voxel grid: {nx}×{ny}×{nz} = {nx*ny*nz:,} voxels")
    return class_grid, dens_grid, scene_min, grid_dims


@njit
def _assign_class_priority(vi_x, vi_y, vi_z, classifications, nx, ny, nz):
    class_grid = np.zeros((nx, ny, nz), dtype=np.int8)
    for i in range(len(vi_x)):
        ix, iy, iz = vi_x[i], vi_y[i], vi_z[i]
        c = classifications[i]
        existing = class_grid[ix, iy, iz]
        p_new = _class_priority(c)
        p_old = _class_priority(existing)
        if p_new > p_old:
            class_grid[ix, iy, iz] = c
    return class_grid


@njit
def _class_priority(c):
    if c == 6: return 5
    elif c == 5: return 4
    elif c == 4: return 3
    elif c == 3: return 2
    elif c == 2: return 1
    else: return 0


def _fill_below_surface(class_grid, fill_class, priority_map):
    nx, ny, nz = class_grid.shape
    fill_priority = priority_map[fill_class]
    match = class_grid == fill_class
    z_indices = np.arange(nz)[np.newaxis, np.newaxis, :]
    z_where_match = np.where(match, z_indices, -1)
    max_z = np.max(z_where_match, axis=2)

    cols = np.argwhere(max_z >= 0)
    for idx in range(len(cols)):
        ix, iy = cols[idx, 0], cols[idx, 1]
        top_z = max_z[ix, iy]
        for z_level in range(top_z - 1, -1, -1):
            existing = class_grid[ix, iy, z_level]
            if _class_priority_py(existing) < fill_priority:
                class_grid[ix, iy, z_level] = fill_class
            else:
                break


def _class_priority_py(c):
    return {6: 5, 5: 4, 4: 3, 3: 2, 2: 1}.get(c, 0)


# ============================================================================
# PV ARRAY GEOMETRY
# ============================================================================

def generate_pv_array_points(
    corner_coords, tilt_deg=20, az_deg=260,
    panel_width_m=1.0, panel_height_m=1.6,
    row_configuration=(6, 2),
    align="left",
):
    """
    Generate world-coordinate panel center points for a rooftop PV array.

    Parameters
    ----------
    corner_coords : array-like, shape (3,)
        World coordinates of the anchor corner.
    align : str
        'left'  — panels extend rightward from anchor (left corner).
        'right' — panels extend leftward from anchor (right corner).
    """
    tilt_rad = np.radians(tilt_deg)
    rot_z_rad = np.radians(180 - az_deg)

    num_rows = len(row_configuration)
    total_height = (num_rows - 1) * panel_height_m

    local_points = []
    y_steps = np.linspace(total_height, 0.0, num_rows)

    for i, num_panels in enumerate(row_configuration):
        y = y_steps[i]
        row_width = num_panels * panel_width_m
        if align == "right":
            # Right-aligned: rightmost panel edge at x=0, panels extend left
            x_start = -(row_width - panel_width_m / 2)
        else:
            # Left-aligned: leftmost panel edge at x=0
            x_start = panel_width_m / 2
        for p in range(num_panels):
            x = x_start + p * panel_width_m
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

    R_combined = R_az @ R_tilt
    rotated_points = (R_combined @ local_points.T).T
    world_points = rotated_points + np.asarray(corner_coords)

    return world_points


def compute_optical_center(corner_coords, **kwargs):
    """Compute the irradiance-weighted optical center of the PV array."""
    panel_points = generate_pv_array_points(corner_coords, **kwargs)
    return np.mean(panel_points, axis=0)


# ============================================================================
# CONE / SOLAR DISK SAMPLING
# ============================================================================

def precompute_cone_offsets(num_samples):
    indices = np.arange(num_samples, dtype=np.float64)
    r = np.sqrt((indices + 0.5) / num_samples)
    phi = 2.0 * np.pi * 0.618034 * indices
    return np.column_stack([r * np.cos(phi), r * np.sin(phi)])


def build_cone_directions(center_dir, radius_rad, offsets_2d):
    if np.abs(center_dir[2]) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    else:
        up = np.array([0.0, 0.0, 1.0])
    u = np.cross(up, center_dir)
    u /= np.linalg.norm(u)
    v = np.cross(center_dir, u)
    dirs = (
        center_dir[np.newaxis, :]
        + radius_rad * offsets_2d[:, 0:1] * u[np.newaxis, :]
        + radius_rad * offsets_2d[:, 1:2] * v[np.newaxis, :]
    )
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs /= norms
    return dirs


# ============================================================================
# MODULE 3: RAY TRAVERSAL (3D-DDA with Beer-Lambert)
# ============================================================================

@njit(fastmath=True)
def g_function_spherical(cos_zenith):
    return 0.5


@njit(fastmath=True)
def calculate_ray_transmittance(
    origin, direction, scene_min, voxel_size,
    grid_dims, class_grid, dens_grid,
    k_base, g_class, b_class,
    origin_ix, origin_iy, origin_iz,
    buffer_dist=0.1,
    skip_dist=0.0,
):
    """
    Trace a single ray through the voxel grid using 3D-DDA.

    Self-occlusion avoidance: building/ground voxels within skip_dist
    of the ray origin are ignored (Chebyshev + distance check).
    """
    cos_zenith = abs(direction[2])
    g_theta = g_function_spherical(cos_zenith)

    adj_origin = origin + direction * buffer_dist
    ray_pos = (adj_origin - scene_min) / voxel_size
    ix = int(math.floor(ray_pos[0]))
    iy = int(math.floor(ray_pos[1]))
    iz = int(math.floor(ray_pos[2]))

    if not (0 <= ix < grid_dims[0] and 0 <= iy < grid_dims[1] and 0 <= iz < grid_dims[2]):
        return 1.0

    step_x = 1 if direction[0] >= 0 else -1
    step_y = 1 if direction[1] >= 0 else -1
    step_z = 1 if direction[2] >= 0 else -1

    t_delta_x = voxel_size / abs(direction[0]) if direction[0] != 0.0 else 1e30
    t_delta_y = voxel_size / abs(direction[1]) if direction[1] != 0.0 else 1e30
    t_delta_z = voxel_size / abs(direction[2]) if direction[2] != 0.0 else 1e30

    if direction[0] != 0.0:
        t_max_x = ((ix + (1 if step_x > 0 else 0)) * voxel_size + scene_min[0] - adj_origin[0]) / direction[0]
    else:
        t_max_x = 1e30
    if direction[1] != 0.0:
        t_max_y = ((iy + (1 if step_y > 0 else 0)) * voxel_size + scene_min[1] - adj_origin[1]) / direction[1]
    else:
        t_max_y = 1e30
    if direction[2] != 0.0:
        t_max_z = ((iz + (1 if step_z > 0 else 0)) * voxel_size + scene_min[2] - adj_origin[2]) / direction[2]
    else:
        t_max_z = 1e30

    transmittance = 1.0
    current_t = 0.0

    while True:
        if not (0 <= ix < grid_dims[0] and 0 <= iy < grid_dims[1] and 0 <= iz < grid_dims[2]):
            break

        v_class = class_grid[ix, iy, iz]
        next_t = min(t_max_x, t_max_y, t_max_z)
        path_len = next_t - current_t
        if path_len < 0.0:
            path_len = 0.0

        # Solid occlusion (ground / building)
        if v_class == b_class or v_class == g_class:
            dx = abs(ix - origin_ix)
            dy = abs(iy - origin_iy)
            dz = abs(iz - origin_iz)
            if (dx <= 1 and dy <= 1 and dz <= 1) or (current_t + buffer_dist < skip_dist):
                pass  # skip — within safe zone of origin building
            else:
                return 0.0

        # Vegetation attenuation
        elif 3 <= v_class <= 5:
            density = dens_grid[ix, iy, iz]
            if density > 0.0:
                transmittance *= math.exp(-k_base * g_theta * density * path_len)

        if transmittance < 1e-6:
            return 0.0

        current_t = next_t
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                ix += step_x
                t_max_x += t_delta_x
            else:
                iz += step_z
                t_max_z += t_delta_z
        else:
            if t_max_y < t_max_z:
                iy += step_y
                t_max_y += t_delta_y
            else:
                iz += step_z
                t_max_z += t_delta_z

    return transmittance


# ============================================================================
# BATCH RAY TRACING
# ============================================================================

@njit(parallel=True, fastmath=True)
def trace_batch(
    array_points, directions, scene_min, voxel_size,
    grid_dims, class_grid, dens_grid, k_base,
    g_class, b_class, buffer_dist, skip_dist
):
    """Trace all (panel × cone_ray) combinations. Returns mean transmittance."""
    n_panels = array_points.shape[0]
    n_cone = directions.shape[0]
    panel_means = np.empty(n_panels, dtype=np.float64)

    for p in prange(n_panels):
        origin = array_points[p]
        ov = (origin - scene_min) / voxel_size
        o_ix = int(math.floor(ov[0]))
        o_iy = int(math.floor(ov[1]))
        o_iz = int(math.floor(ov[2]))

        acc = 0.0
        for c in range(n_cone):
            acc += calculate_ray_transmittance(
                origin, directions[c],
                scene_min, voxel_size, grid_dims,
                class_grid, dens_grid, k_base,
                g_class, b_class,
                o_ix, o_iy, o_iz,
                buffer_dist, skip_dist
            )
        panel_means[p] = acc / n_cone

    return np.mean(panel_means)


# ============================================================================
# MODULE 4: SIMULATION ORCHESTRATION
# ============================================================================

def _find_roof_z(points, classifications, corner_xy, search_radius,
                 offset_from_roof, scene_min_z):
    """Find roof height near a corner and return panel Z."""
    dists = np.linalg.norm(points[:, :2] - corner_xy, axis=1)
    mask = dists < search_radius
    roof_pts = points[mask & (classifications == BUILDING_CLASS)]
    if len(roof_pts) > 0:
        roof_max = np.max(roof_pts[:, 2])
        target_z = roof_max + offset_from_roof
        print(f"    Roof: {len(roof_pts)} pts, max_z={roof_max:.2f}m, "
              f"offset={offset_from_roof:+.1f}m → panel_z={target_z:.2f}m")
        return target_z
    else:
        print(f"    WARNING: no building points near corner, using scene min Z")
        return scene_min_z


def _run_hemispherical_sweep(
    array_points, scene_min_c, voxel_size, grid_dims_c,
    class_grid_c, dens_grid_c, k_base,
    cone_offsets, solar_radius_rad,
    buf_dist, skip_dist, label="",
):
    """
    Run the full elevation × azimuth sweep for a set of panel points.
    Returns a shadow-intensity DataFrame (91 × 361).
    """
    start = time.time()
    array_points_c = np.ascontiguousarray(array_points)

    elevations_rad = np.linspace(0, np.pi / 2, ELEVATION_STEPS, endpoint=True)
    azimuths_rad = np.linspace(0, 2 * np.pi, AZIMUTH_STEPS, endpoint=False)
    total_tasks = ELEVATION_STEPS * AZIMUTH_STEPS

    print(f"\n--- Sweep: {label} ({len(array_points)} panels) ---")

    results_transmittance = np.zeros((ELEVATION_STEPS, AZIMUTH_STEPS), dtype=np.float64)
    completed = 0

    for ei, el in enumerate(elevations_rad):
        el_deg = np.rad2deg(el)

        if el < 0.001:
            results_transmittance[ei, :] = 0.0
            completed += AZIMUTH_STEPS
            continue

        center_dirs = np.empty((AZIMUTH_STEPS, 3), dtype=np.float64)
        center_dirs[:, 0] = np.cos(el) * np.sin(azimuths_rad)
        center_dirs[:, 1] = np.cos(el) * np.cos(azimuths_rad)
        center_dirs[:, 2] = np.sin(el)

        for ai in range(AZIMUTH_STEPS):
            cone_dirs = build_cone_directions(
                center_dirs[ai], solar_radius_rad, cone_offsets
            )
            t = trace_batch(
                array_points_c, cone_dirs, scene_min_c, voxel_size,
                grid_dims_c, class_grid_c, dens_grid_c, k_base,
                GROUND_CLASS, BUILDING_CLASS, buf_dist, skip_dist
            )
            results_transmittance[ei, ai] = t

        completed += AZIMUTH_STEPS
        if (ei + 1) % 10 == 0 or ei == ELEVATION_STEPS - 1:
            pct = 100.0 * completed / total_tasks
            elapsed = time.time() - start
            print(f"  Elevation {el_deg:5.1f}° done  "
                  f"({completed:,}/{total_tasks:,} = {pct:.1f}%)  "
                  f"[{elapsed:.0f}s elapsed]")

    shadow_matrix = 1.0 - results_transmittance

    elev_labels = [f"Altitude_{i}" for i in range(ELEVATION_STEPS)]
    azim_labels = [f"Azimuth_{i}" for i in range(AZIMUTH_STEPS)]
    matrix_df = pd.DataFrame(shadow_matrix, index=elev_labels, columns=azim_labels)
    matrix_df["Azimuth_360"] = matrix_df["Azimuth_0"]

    elapsed = time.time() - start
    print(f"  {label} completed in {elapsed:.1f}s")
    return matrix_df


def create_shadow_matrices(
    lidar_file_path=None, voxel_size=2.0,
    output_dir=None,
    buf_dist=0.1, skip_dist=SKIP_DISTANCE,
    offset_from_roof=OFFSET_FROM_ROOF,
    file_name_suffix="v1",
):
    """
    Build separate hemispherical shadow matrices for Bank 2 North and South.

    Loads and voxelizes the LiDAR scene ONCE, then runs the ray-tracing
    sweep separately for each sub-array. Produces two CSV files:
      - ..._W_north_v1.csv  (6+2 sub-array)
      - ..._W_south_v1.csv  (4+2 sub-array)

    Returns
    -------
    north_df, south_df : pd.DataFrame
    """
    start_time = time.time()

    # --- Load & voxelize (once) ---
    points, classifications = load_and_prepare_lidar(
        lidar_file_path, relevant_classes=RELEVANT_CLASSES
    )
    class_grid, dens_grid, scene_min, grid_dims = voxelize_scene(
        points, classifications, voxel_size, BETA_FINLAND
    )

    # --- Build panel points ---
    print("\n  --- Sub-array: North (6, 2) — left-aligned ---")
    print("    Panels: 2-1-1…2-1-6 (row 0), 2-1-7…2-1-8 (row 1)")
    north_z = _find_roof_z(points, classifications, NORTH_CORNER_2D,
                           ROOF_SEARCH_RADIUS, offset_from_roof, scene_min[2])
    north_corner = np.array([NORTH_CORNER_2D[0], NORTH_CORNER_2D[1], north_z])
    north_pts = generate_pv_array_points(
        north_corner, tilt_deg=20, az_deg=260,
        panel_width_m=1.0, panel_height_m=1.6,
        row_configuration=(6, 2), align="left",
    )
    print(f"    {len(north_pts)} panel points")

    print("\n  --- Sub-array: South (4, 2) — right-aligned ---")
    print("    Panels: 2-2-1…2-2-4 (row 0), 2-2-5…2-2-6 (row 1)")
    south_z = _find_roof_z(points, classifications, SOUTH_CORNER_2D,
                           ROOF_SEARCH_RADIUS, offset_from_roof, scene_min[2])
    south_corner = np.array([SOUTH_CORNER_2D[0], SOUTH_CORNER_2D[1], south_z])
    south_pts = generate_pv_array_points(
        south_corner, tilt_deg=20, az_deg=260,
        panel_width_m=1.0, panel_height_m=1.6,
        row_configuration=(4, 2), align="right",
    )
    print(f"    {len(south_pts)} panel points")

    # --- Shared setup ---
    cone_offsets = precompute_cone_offsets(NUM_RAYS_PER_CONE)
    solar_radius_rad = np.deg2rad(SOLAR_ANGULAR_RADIUS_DEG)
    k_base = K_BASE_FINLAND

    print(f"\n  Self-occlusion skip distance: {skip_dist:.1f}m "
          f"(voxel_size={voxel_size}m)")

    scene_min_c = np.ascontiguousarray(scene_min)
    grid_dims_c = np.ascontiguousarray(grid_dims)
    class_grid_c = np.ascontiguousarray(class_grid)
    dens_grid_c = np.ascontiguousarray(dens_grid)

    # --- JIT warm-up ---
    print("\n--- JIT warm-up (first trace) ---")
    _wu_dirs = build_cone_directions(
        np.array([0.0, 0.0, 1.0]), solar_radius_rad, cone_offsets
    )
    _ = trace_batch(
        np.ascontiguousarray(north_pts), _wu_dirs, scene_min_c, voxel_size,
        grid_dims_c, class_grid_c, dens_grid_c, k_base,
        GROUND_CLASS, BUILDING_CLASS, buf_dist, skip_dist
    )

    # --- Sweep: North ---
    north_df = _run_hemispherical_sweep(
        north_pts, scene_min_c, voxel_size, grid_dims_c,
        class_grid_c, dens_grid_c, k_base,
        cone_offsets, solar_radius_rad,
        buf_dist, skip_dist, label="North (6+2)",
    )

    # --- Sweep: South ---
    south_df = _run_hemispherical_sweep(
        south_pts, scene_min_c, voxel_size, grid_dims_c,
        class_grid_c, dens_grid_c, k_base,
        cone_offsets, solar_radius_rad,
        buf_dist, skip_dist, label="South (4+2)",
    )

    # --- Save ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        north_path = os.path.join(output_dir,
            f"shadow_attenuation_matrix_conecasting_W_north_{file_name_suffix}.csv")
        north_df.to_csv(north_path, header=True, index=True)
        print(f"\nSaved North matrix to {north_path}")

        south_path = os.path.join(output_dir,
            f"shadow_attenuation_matrix_conecasting_W_south_{file_name_suffix}.csv")
        south_df.to_csv(south_path, header=True, index=True)
        print(f"Saved South matrix to {south_path}")

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f}s "
          f"({total_time / 60:.1f} min)")
    return north_df, south_df


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    LIDAR_FILE_PATH = "output/reclassified_final_v5.laz"
    OUTPUT_DIRECTORY = "results/shadow_matrix_results_W_pro"

    create_shadow_matrices(
        lidar_file_path=LIDAR_FILE_PATH,
        voxel_size=2.0,
        output_dir=OUTPUT_DIRECTORY,
        offset_from_roof=OFFSET_FROM_ROOF,
        skip_dist=SKIP_DISTANCE,
        file_name_suffix="v1",
    )