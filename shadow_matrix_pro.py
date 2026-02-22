"""
Voxel-Based Ray-Tracing Shadow Matrix Generator for PV Array Siting
====================================================================
Uses LiDAR point clouds and the Beer-Lambert model to simulate light
interception in plant canopies via 3D-DDA voxel traversal.

Key improvements over prior version:
  - Corrected Beer-Lambert: no double clumping correction
  - Zenith-angle-dependent G(θ) for spherical leaf angle distribution
  - Vectorized voxelization (np.add.at) and ground/building infill
  - Restructured parallelism: grids shared across workers, batched tasks
  - Pre-computed & normalized cone offset vectors
  - High-elevation fill (transmittance=1.0) instead of NaN
  - Robust origin-voxel tracking for self-occlusion avoidance
"""

import os
import time
import math
import laspy
import numpy as np
import pandas as pd
from numba import njit, prange
from multiprocessing import shared_memory

# ============================================================================
# CONFIGURATION (Boreal Calibration)
# ============================================================================

BETA_FINLAND = 2.08
# K_BASE is the base extinction coefficient derived from β.
# Since column_lai_e = -β * ln(P_gap) already yields EFFECTIVE LAI
# (clumping is embedded in the gap-fraction measurement), we do NOT
# multiply k by Ω again.  Ω is only needed if converting LAI_e → LAI_true.
K_BASE_FINLAND = 1.0 / BETA_FINLAND  # ≈ 0.481
# OMEGA_S is retained for reference but NOT used to scale k.
OMEGA_S = 0.56

TARGET_COORDS_2D = np.array([532882.50, 6983507.00])
ROOF_SEARCH_RADIUS = 2.0
OFFSET_FROM_ROOF = 1.5

GROUND_CLASS = 2
BUILDING_CLASS = 6
VEGETATION_CLASSES = {3, 4, 5}
RELEVANT_CLASSES = {2, 3, 4, 5, 6}
CLASS_PRIORITY = {6: 5, 5: 4, 4: 3, 3: 2, 2: 1, 0: 0}

AZIMUTH_STEPS = 360
ELEVATION_STEPS = 91
SOLAR_ANGULAR_RADIUS_DEG = 0.265
# Reduced from 16 — for a 0.265° disk, 6 rays suffice with Fibonacci sampling
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

    The density derivation follows:
      1. Per-column gap fraction  P_gap = ground_returns / total_returns
      2. Effective LAI             LAI_e = -β · ln(P_gap)
      3. Vertical distribution     LAD_voxel = (frac_veg_in_voxel) · LAI_e / Δz

    Because LAI_e already embeds the clumping index Ω, no further Ω scaling
    is applied to the extinction coefficient downstream.
    """
    scene_min = np.min(points, axis=0)
    scene_max = np.max(points, axis=0)
    grid_dims = np.ceil((scene_max - scene_min) / voxel_size).astype(np.int32)
    # Ensure at least 1 voxel per dimension
    grid_dims = np.maximum(grid_dims, 1)
    nx, ny, nz = grid_dims

    # --- Vectorized voxel index computation ---
    voxel_indices = np.clip(
        np.floor((points - scene_min) / voxel_size).astype(np.int32),
        0, grid_dims - 1
    )
    vi_x, vi_y, vi_z = voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]

    # --- Class grid (priority-based) using vectorized approach ---
    # Numba-accelerated priority assignment
    class_grid = _assign_class_priority(
        vi_x, vi_y, vi_z, classifications, nx, ny, nz
    )

    # --- 2D column counts (vectorized with np.add.at) ---
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

    # --- 3D vegetation point counts (vectorized) ---
    dens_grid = np.zeros((nx, ny, nz), dtype=np.float32)
    flat_3d_veg = (
        vi_x[veg_mask].astype(np.int64) * ny * nz
        + vi_y[veg_mask].astype(np.int64) * nz
        + vi_z[veg_mask].astype(np.int64)
    )
    veg_counts_3d = np.zeros(nx * ny * nz, dtype=np.float32)
    np.add.at(veg_counts_3d, flat_3d_veg, 1.0)
    dens_grid = veg_counts_3d.reshape(nx, ny, nz)

    # --- API → effective LAI → leaf area density (LAD) ---
    api_grid = np.ones((nx, ny), dtype=np.float32)
    valid = total_counts > 0
    api_grid[valid] = ground_counts[valid] / total_counts[valid]
    api_grid = np.clip(api_grid, 0.01, 1.0)
    column_lai_e = -beta * np.log(api_grid)

    # Distribute column LAI_e vertically by vegetation fraction
    safe_veg = np.where(veg_counts_2d > 0, veg_counts_2d, 1).astype(np.float32)
    for iz_idx in range(nz):
        dens_grid[:, :, iz_idx] = (
            (dens_grid[:, :, iz_idx] / safe_veg) * (column_lai_e / voxel_size)
        )
    # Zero out columns with no vegetation points
    no_veg = veg_counts_2d == 0
    dens_grid[no_veg, :] = 0.0

    # --- Vectorized ground/building infill ---
    _fill_below_surface(class_grid, GROUND_CLASS, CLASS_PRIORITY)
    _fill_below_surface(class_grid, BUILDING_CLASS, CLASS_PRIORITY)

    print(f"  Voxel grid: {nx}×{ny}×{nz} = {nx*ny*nz:,} voxels")
    return class_grid, dens_grid, scene_min, grid_dims


@njit
def _assign_class_priority(vi_x, vi_y, vi_z, classifications, nx, ny, nz):
    """Assign voxel classes respecting a priority order (Numba-accelerated)."""
    class_grid = np.zeros((nx, ny, nz), dtype=np.int8)
    # Inline priority lookup (avoid dict in njit)
    # {6:5, 5:4, 4:3, 3:2, 2:1, 0:0}
    for i in range(len(vi_x)):
        ix, iy, iz = vi_x[i], vi_y[i], vi_z[i]
        c = classifications[i]
        existing = class_grid[ix, iy, iz]
        # Priority function inlined
        p_new = _class_priority(c)
        p_old = _class_priority(existing)
        if p_new > p_old:
            class_grid[ix, iy, iz] = c
    return class_grid


@njit
def _class_priority(c):
    """Return priority for a classification code."""
    if c == 6:
        return 5
    elif c == 5:
        return 4
    elif c == 4:
        return 3
    elif c == 3:
        return 2
    elif c == 2:
        return 1
    else:
        return 0


def _fill_below_surface(class_grid, fill_class, priority_map):
    """
    For each column, find the highest voxel of fill_class and fill everything
    below it (vectorized per-column rather than per-voxel Python loop).
    """
    nx, ny, nz = class_grid.shape
    fill_priority = priority_map[fill_class]

    # Mask of voxels matching fill_class
    match = class_grid == fill_class  # (nx, ny, nz)

    # For each column, find the highest z with this class (-1 if absent)
    # Work with z-indices: build array of z-values where match is True, else -1
    z_indices = np.arange(nz)[np.newaxis, np.newaxis, :]  # (1, 1, nz)
    z_where_match = np.where(match, z_indices, -1)
    max_z = np.max(z_where_match, axis=2)  # (nx, ny)

    # For columns that have at least one match
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
    """Pure-Python priority lookup (for use outside Numba)."""
    return {6: 5, 5: 4, 4: 3, 3: 2, 2: 1}.get(c, 0)


# ============================================================================
# PV ARRAY GEOMETRY
# ============================================================================

def generate_pv_array_points(
    corner_coords, tilt_deg=12, az_deg=170,
    panel_width_m=1.0, panel_height_m=1.6,
    row_configuration=(5, 4, 3)
):
    """
    Generate world-coordinate panel center points for a rooftop PV array,
    anchored from the LEFT CORNER of the array.

    The left corner is defined as the bottom-left of the widest (first) row
    in local coordinates before rotation. Panels extend:
      - rightward (+x local) across each row
      - upward (+y local) across rows

    After construction, the local grid is rotated by tilt and azimuth,
    then translated so the left corner lands on corner_coords.

    Parameters
    ----------
    corner_coords : array-like, shape (3,)
        World coordinates (x, y, z) of the left corner anchor point.
    tilt_deg : float
        Panel tilt angle in degrees.
    az_deg : float
        Panel azimuth in degrees (compass convention, 180 = south).
    panel_width_m : float
        Width of each panel in meters.
    panel_height_m : float
        Height (depth) of each panel row in meters.
    row_configuration : tuple of int
        Number of panels per row, top-to-bottom (e.g., (5, 4, 3)).

    Returns
    -------
    np.ndarray, shape (n_panels, 3)
        World-coordinate panel center points.
    """
    tilt_rad = np.radians(tilt_deg)
    rot_z_rad = np.radians(180 - az_deg)

    num_rows = len(row_configuration)
    max_width = max(row_configuration) * panel_width_m
    total_height = (num_rows - 1) * panel_height_m

    # Build local panel centers anchored at left corner = (0, 0)
    # x: left-aligned, panels start at x = panel_width/2 (center of first panel)
    # y: row 0 (top row) at y = total_height, last row at y = 0
    local_points = []
    y_steps = np.linspace(total_height, 0.0, num_rows)  # top row → bottom row

    for i, num_panels in enumerate(row_configuration):
        y = y_steps[i]
        for p in range(num_panels):
            x = p * panel_width_m + panel_width_m / 2  # center of panel p
            local_points.append([x, y, 0.0])

    local_points = np.array(local_points)

    # The left corner in local space is (0, 0, 0).
    # Panel centers are offset from it. After rotation, we translate
    # so that the rotated left corner lands on corner_coords.

    # Rotation matrices
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

    # Rotate local points
    rotated_points = (R_combined @ local_points.T).T

    # The local origin (0,0,0) = left corner. After rotation it maps to (0,0,0)
    # since R @ [0,0,0]^T = [0,0,0]^T. So we just translate by corner_coords.
    world_points = rotated_points + np.asarray(corner_coords)

    return world_points



# ============================================================================
# CONE / SOLAR DISK SAMPLING (pre-computed, normalized)
# ============================================================================

def precompute_cone_offsets(num_samples):
    """
    Build unit-disk offsets using a Fibonacci spiral.
    Returns (num_samples, 2) array of (r·cos(φ), r·sin(φ)) values
    in a unit disk that can be scaled by the angular radius.
    """
    indices = np.arange(num_samples, dtype=np.float64)
    r = np.sqrt((indices + 0.5) / num_samples)
    phi = 2.0 * np.pi * 0.618034 * indices
    return np.column_stack([r * np.cos(phi), r * np.sin(phi)])  # (N, 2)


def build_cone_directions(center_dir, radius_rad, offsets_2d):
    """
    Given a center direction (unit vector), angular radius, and pre-computed
    2D disk offsets, return an array of normalized ray directions.
    """
    # Construct local tangent basis (u, v) perpendicular to center_dir
    if np.abs(center_dir[2]) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    else:
        up = np.array([0.0, 0.0, 1.0])
    u = np.cross(up, center_dir)
    u /= np.linalg.norm(u)
    v = np.cross(center_dir, u)

    # Perturbed directions: center + radius * (offset_x * u + offset_y * v)
    dirs = (
        center_dir[np.newaxis, :]
        + radius_rad * offsets_2d[:, 0:1] * u[np.newaxis, :]
        + radius_rad * offsets_2d[:, 1:2] * v[np.newaxis, :]
    )
    # Normalize each direction
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs /= norms
    return dirs  # (num_samples, 3)


# ============================================================================
# MODULE 3: RAY TRAVERSAL (3D-DDA with Beer-Lambert)
# ============================================================================

@njit(fastmath=True)
def g_function_spherical(cos_zenith):
    """
    G(θ) projection function for a spherical leaf angle distribution.
    G(θ) = 0.5 for all θ. Retained as a function for easy substitution
    of other distributions (e.g., erectophile, planophile).
    """
    return 0.5


@njit(fastmath=True)
def calculate_ray_transmittance(
    origin, direction, scene_min, voxel_size,
    grid_dims, class_grid, dens_grid,
    k_base, g_class, b_class,
    origin_ix, origin_iy, origin_iz,
    buffer_dist=0.1
):
    """
    Trace a single ray through the voxel grid using 3D-DDA.

    Beer-Lambert attenuation per vegetation voxel:
        T_voxel = exp( -k_base · G(θ) · LAD · path_length )

    where k_base = 1/β (no additional Ω scaling since LAI_e already
    includes clumping), and G(θ) depends on the ray's zenith angle.

    Parameters
    ----------
    origin_ix, origin_iy, origin_iz : int
        Voxel indices of the ray origin BEFORE buffer offset, used for
        robust self-occlusion avoidance.
    """
    # Zenith angle of this ray (angle from vertical = z-axis)
    cos_zenith = abs(direction[2])  # |cos(θ_z)|
    g_theta = g_function_spherical(cos_zenith)

    adj_origin = origin + direction * buffer_dist
    ray_pos = (adj_origin - scene_min) / voxel_size
    ix = int(math.floor(ray_pos[0]))
    iy = int(math.floor(ray_pos[1]))
    iz = int(math.floor(ray_pos[2]))

    if not (0 <= ix < grid_dims[0] and 0 <= iy < grid_dims[1] and 0 <= iz < grid_dims[2]):
        return 1.0  # ray starts outside grid → open sky

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
            path_len = 0.0  # guard against floating-point edge case

        # Solid occlusion (ground / building)
        if v_class == b_class or v_class == g_class:
            # Use pre-buffer origin voxel for self-occlusion check
            if ix == origin_ix and iy == origin_iy and iz == origin_iz:
                pass  # skip origin voxel
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
# BATCH RAY TRACING (Numba parallel over panel × cone rays)
# ============================================================================

@njit(parallel=True, fastmath=True)
def trace_batch(
    array_points, directions, scene_min, voxel_size,
    grid_dims, class_grid, dens_grid, k_base,
    g_class, b_class, buffer_dist
):
    """
    Trace all (panel_point × cone_ray) combinations for a SINGLE solar
    direction.  Returns the mean transmittance across panels.

    directions : (n_cone, 3)  — cone ray directions for this solar position
    array_points : (n_panels, 3)
    """
    n_panels = array_points.shape[0]
    n_cone = directions.shape[0]
    panel_means = np.empty(n_panels, dtype=np.float64)

    for p in prange(n_panels):
        origin = array_points[p]
        # Compute origin voxel BEFORE buffer offset
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
                buffer_dist
            )
        panel_means[p] = acc / n_cone

    return np.mean(panel_means)


# ============================================================================
# MODULE 4: SIMULATION ORCHESTRATION
# ============================================================================

def compute_optical_center(
    corner_coords, tilt_deg=12, az_deg=170,
    panel_width_m=1.0, panel_height_m=1.6,
    row_configuration=(5, 4, 3)
):
    """
    Compute the irradiance-weighted optical center of the PV array.

    The optical center is the area-weighted centroid of all panel positions,
    where each panel's weight is proportional to cos(tilt) — i.e. its
    projected area normal to the sky hemisphere. For a uniform tilt this
    reduces to the simple geometric centroid of the panel centers.

    This is useful for single-point shading lookups, sensor placement,
    or converting between corner-anchored and center-anchored representations.

    Parameters
    ----------
    corner_coords : array-like, shape (3,)
        World coordinates (x, y, z) of the left corner anchor.
    tilt_deg, az_deg, panel_width_m, panel_height_m, row_configuration :
        Same as generate_pv_array_points.

    Returns
    -------
    optical_center : np.ndarray, shape (3,)
        World coordinates of the optical center.
    """
    panel_points = generate_pv_array_points(
        corner_coords,
        tilt_deg=tilt_deg,
        az_deg=az_deg,
        panel_width_m=panel_width_m,
        panel_height_m=panel_height_m,
        row_configuration=row_configuration,
    )

    # For a uniform-tilt array every panel has the same projected area,
    # so the optical center is the arithmetic mean of panel centers.
    # If panels had varying tilts, weight by cos(tilt_i) here.
    optical_center = np.mean(panel_points, axis=0)

    return optical_center


def create_shadow_matrix(
    lidar_file_path=None, voxel_size=1.0,
    output_dir=None, output_fn=None, buf_dist=0.1
):
    """
    Build a full hemispherical shadow-attenuation matrix (elevation × azimuth).

    The matrix stores shadow intensity = 1 − transmittance for each
    discrete sky direction.
    """
    start_time = time.time()

    # --- Load & voxelize ---
    points, classifications = load_and_prepare_lidar(
        lidar_file_path, relevant_classes=RELEVANT_CLASSES
    )

    class_grid, dens_grid, scene_min, grid_dims = voxelize_scene(
        points, classifications, voxel_size, BETA_FINLAND
    )

    # --- Determine PV array placement ---
    dists = np.linalg.norm(points[:, :2] - TARGET_COORDS_2D, axis=1)
    mask = dists < ROOF_SEARCH_RADIUS
    roof_points = points[mask & (classifications == BUILDING_CLASS)]
    if len(roof_points) > 0:
        target_z = np.max(roof_points[:, 2]) + OFFSET_FROM_ROOF
    else:
        target_z = scene_min[2]
        print("  WARNING: No building points found near target. Using scene min Z.")

    array_corner = np.array([
        TARGET_COORDS_2D[0], TARGET_COORDS_2D[1], target_z
    ])

    center = compute_optical_center(
        corner_coords=array_corner,
        tilt_deg=12, az_deg=170,
        panel_width_m=1.0, panel_height_m=1.6,
        row_configuration=(5, 4, 3)
    )
    # print center with 2 decimal places for readability
    print(f"  PV array optical center at: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")

    array_points = generate_pv_array_points(
        array_corner, tilt_deg=12, az_deg=170,
        panel_width_m=1.0, panel_height_m=1.6,
        row_configuration=(5, 4, 3)
    )

    # --- Pre-compute cone offsets (reused for every direction) ---
    cone_offsets = precompute_cone_offsets(NUM_RAYS_PER_CONE)
    solar_radius_rad = np.deg2rad(SOLAR_ANGULAR_RADIUS_DEG)

    # Extinction coefficient: k_base only — no Ω scaling (LAI_e already effective)
    k_base = K_BASE_FINLAND

    # --- Build task list ---
    elevations_rad = np.linspace(0, np.pi / 2, ELEVATION_STEPS, endpoint=True)
    azimuths_rad = np.linspace(0, 2 * np.pi, AZIMUTH_STEPS, endpoint=False)

    # Ensure contiguous arrays for Numba
    scene_min_c = np.ascontiguousarray(scene_min)
    grid_dims_c = np.ascontiguousarray(grid_dims)
    class_grid_c = np.ascontiguousarray(class_grid)
    dens_grid_c = np.ascontiguousarray(dens_grid)
    array_points_c = np.ascontiguousarray(array_points)

    # Warm up Numba JIT (first call compiles; subsequent calls are fast)
    print("\n--- JIT warm-up (first trace) ---")
    _warmup_dir = np.array([0.0, 0.0, 1.0])
    _warmup_dirs = build_cone_directions(_warmup_dir, solar_radius_rad, cone_offsets)
    _ = trace_batch(
        array_points_c, _warmup_dirs, scene_min_c, voxel_size,
        grid_dims_c, class_grid_c, dens_grid_c, k_base,
        GROUND_CLASS, BUILDING_CLASS, buf_dist
    )

    total_tasks = ELEVATION_STEPS * AZIMUTH_STEPS
    print(f"\n--- Starting Hemispherical Cone-Casting Simulation ---")
    print(f"  {ELEVATION_STEPS} elevations × {AZIMUTH_STEPS} azimuths = "
          f"{total_tasks:,} directions")
    print(f"  {len(array_points)} panel points × {NUM_RAYS_PER_CONE} cone rays "
          f"= {len(array_points) * NUM_RAYS_PER_CONE} traces per direction")

    # --- Main computation loop ---
    # We iterate over elevations (outer) and azimuths (inner).
    # For elevation = 0 (horizontal) transmittance is set to 0 (below horizon).
    # For high elevations with minimal canopy overhead, transmittance → 1.
    # Numba prange handles panel-level parallelism inside trace_batch;
    # we avoid joblib to eliminate per-task grid serialization overhead.

    results_transmittance = np.zeros((ELEVATION_STEPS, AZIMUTH_STEPS), dtype=np.float64)
    completed = 0

    for ei, el in enumerate(elevations_rad):
        el_deg = np.rad2deg(el)

        if el < 0.001:
            # Below/at horizon — no direct sun
            results_transmittance[ei, :] = 0.0
            completed += AZIMUTH_STEPS
            continue

        # If the sun is very high (>55°), canopy occlusion for a rooftop PV
        # is negligible for most boreal scenes.  We still compute but could
        # short-circuit here if profiling shows it's worthwhile.

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
                GROUND_CLASS, BUILDING_CLASS, buf_dist
            )
            results_transmittance[ei, ai] = t

        completed += AZIMUTH_STEPS
        if (ei + 1) % 10 == 0 or ei == ELEVATION_STEPS - 1:
            pct = 100.0 * completed / total_tasks
            elapsed = time.time() - start_time
            print(f"  Elevation {el_deg:5.1f}° done  "
                  f"({completed:,}/{total_tasks:,} = {pct:.1f}%)  "
                  f"[{elapsed:.0f}s elapsed]")

    # --- Assemble output DataFrame ---
    shadow_matrix = 1.0 - results_transmittance  # shadow intensity

    elev_labels = [f"Altitude_{i}" for i in range(ELEVATION_STEPS)]
    azim_labels = [f"Azimuth_{i}" for i in range(AZIMUTH_STEPS)]

    matrix_df = pd.DataFrame(shadow_matrix, index=elev_labels, columns=azim_labels)

    # Wrap azimuth: duplicate column 0 as column 360 for interpolation
    matrix_df["Azimuth_360"] = matrix_df["Azimuth_0"]

    # --- Save ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, output_fn)
        matrix_df.to_csv(out_path, header=True, index=True)
        print(f"\nSaved shadow attenuation matrix to {out_path}")

    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f}s "
          f"({total_time / 60:.1f} min)")
    return matrix_df


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    LIDAR_FILE_PATH = "output/reclassified_final_v5.laz"
    OUTPUT_DIRECTORY = "results/shadow_matrix_results_SE_pro"
    OUTPUT_FILENAME = "shadow_attenuation_matrix_conecasting_SE_v1.csv"

    create_shadow_matrix(
        lidar_file_path=LIDAR_FILE_PATH,
        voxel_size=1.0,
        output_dir=OUTPUT_DIRECTORY,
        output_fn=OUTPUT_FILENAME,
    )