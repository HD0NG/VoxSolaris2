"""
LAD Diagnostic: Compare voxelization between LiDAR files / resolutions
========================================================================
Traces a ray at a specific solar angle and prints LAD values encountered
in each voxel along the path. Compares two configurations to find why
the new shadow matrix under-attenuates.

Usage:
    python diagnose_lad.py
"""

import numpy as np
import laspy
import math

# ============================================================================
# CONFIG — adjust these to match your setup
# ============================================================================

LIDAR_OLD = "output/recovered_SE_n2.laz"
LIDAR_NEW = "output/reclassified_final_v5.laz"

VOXEL_SIZES = [1.0, 2.0]  # test both resolutions

BETA_FINLAND = 2.08
K_BASE = 1.0 / BETA_FINLAND  # 0.481

TARGET_COORDS_2D = np.array([532884.0, 6983510.0])
ROOF_SEARCH_RADIUS = 2.0
OFFSET_FROM_ROOF = 1.5

GROUND_CLASS = 2
BUILDING_CLASS = 6
VEGETATION_CLASSES = {3, 4, 5}
RELEVANT_CLASSES = {2, 3, 4, 5, 6}

# Solar position to test: 2021-07-04 07:00 local (alt=18.8°, azi=76.2°)
TEST_ALTITUDE_DEG = 18.8
TEST_AZIMUTH_DEG = 76.2


# ============================================================================
# VOXELIZATION (simplified from shadow_matrix_simulation.py)
# ============================================================================

def voxelize(points, classifications, voxel_size, beta=BETA_FINLAND):
    """Simplified voxelization returning class_grid, dens_grid, scene_min, grid_dims."""
    scene_min = np.min(points, axis=0)
    scene_max = np.max(points, axis=0)
    grid_dims = np.maximum(np.ceil((scene_max - scene_min) / voxel_size).astype(int), 1)
    nx, ny, nz = grid_dims

    vi = np.clip(np.floor((points - scene_min) / voxel_size).astype(int), 0, grid_dims - 1)
    vi_x, vi_y, vi_z = vi[:, 0], vi[:, 1], vi[:, 2]

    # Class grid (simple: highest priority wins)
    class_grid = np.zeros((nx, ny, nz), dtype=np.int8)
    priority = {6: 5, 5: 4, 4: 3, 3: 2, 2: 1, 0: 0}
    for i in range(len(points)):
        ix, iy, iz = vi_x[i], vi_y[i], vi_z[i]
        c = classifications[i]
        if priority.get(c, 0) > priority.get(class_grid[ix, iy, iz], 0):
            class_grid[ix, iy, iz] = c

    # Column counts
    ground_mask = classifications == GROUND_CLASS
    veg_mask = np.isin(classifications, list(VEGETATION_CLASSES))

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

    # 3D veg counts
    flat_3d_veg = (
        vi_x[veg_mask].astype(np.int64) * ny * nz
        + vi_y[veg_mask].astype(np.int64) * nz
        + vi_z[veg_mask].astype(np.int64)
    )
    veg_counts_3d = np.zeros(nx * ny * nz, dtype=np.float32)
    np.add.at(veg_counts_3d, flat_3d_veg, 1.0)
    dens_grid = veg_counts_3d.reshape(nx, ny, nz)

    # API -> LAI_e -> LAD
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
    dens_grid[veg_counts_2d == 0, :] = 0.0

    return class_grid, dens_grid, scene_min, grid_dims, api_grid, column_lai_e, veg_counts_2d, total_counts


def trace_ray_verbose(origin, direction, scene_min, voxel_size, grid_dims,
                      class_grid, dens_grid, k_base, g_theta=0.5,
                      buffer_dist=0.1, max_steps=500):
    """Trace a ray and print every voxel encountered with its LAD and attenuation."""
    adj_origin = origin + direction * buffer_dist
    ray_pos = (adj_origin - scene_min) / voxel_size
    ix, iy, iz = int(math.floor(ray_pos[0])), int(math.floor(ray_pos[1])), int(math.floor(ray_pos[2]))

    # Origin voxel (for skip check)
    ov = (origin - scene_min) / voxel_size
    o_ix, o_iy, o_iz = int(math.floor(ov[0])), int(math.floor(ov[1])), int(math.floor(ov[2]))

    if not (0 <= ix < grid_dims[0] and 0 <= iy < grid_dims[1] and 0 <= iz < grid_dims[2]):
        print("    Ray starts outside grid → T=1.0")
        return 1.0

    step_x = 1 if direction[0] >= 0 else -1
    step_y = 1 if direction[1] >= 0 else -1
    step_z = 1 if direction[2] >= 0 else -1

    t_delta_x = voxel_size / abs(direction[0]) if direction[0] != 0 else 1e30
    t_delta_y = voxel_size / abs(direction[1]) if direction[1] != 0 else 1e30
    t_delta_z = voxel_size / abs(direction[2]) if direction[2] != 0 else 1e30

    t_max_x = ((ix + (1 if step_x > 0 else 0)) * voxel_size + scene_min[0] - adj_origin[0]) / direction[0] if direction[0] != 0 else 1e30
    t_max_y = ((iy + (1 if step_y > 0 else 0)) * voxel_size + scene_min[1] - adj_origin[1]) / direction[1] if direction[1] != 0 else 1e30
    t_max_z = ((iz + (1 if step_z > 0 else 0)) * voxel_size + scene_min[2] - adj_origin[2]) / direction[2] if direction[2] != 0 else 1e30

    transmittance = 1.0
    current_t = 0.0
    step_count = 0
    veg_voxels = 0

    CLASS_NAMES = {0: "empty", 2: "ground", 3: "low_veg", 4: "med_veg", 5: "high_veg", 6: "building"}

    while step_count < max_steps:
        if not (0 <= ix < grid_dims[0] and 0 <= iy < grid_dims[1] and 0 <= iz < grid_dims[2]):
            break

        v_class = class_grid[ix, iy, iz]
        next_t = min(t_max_x, t_max_y, t_max_z)
        path_len = max(0, next_t - current_t)

        is_origin = (ix == o_ix and iy == o_iy and iz == o_iz)
        cls_name = CLASS_NAMES.get(v_class, f"cls_{v_class}")

        if v_class == BUILDING_CLASS or v_class == GROUND_CLASS:
            if is_origin:
                pass  # skip
            else:
                print(f"    step {step_count:3d}: [{ix:4d},{iy:4d},{iz:4d}] {cls_name:8s}  "
                      f"path={path_len:.3f}m  → BLOCKED  T=0.0")
                return 0.0
        elif 3 <= v_class <= 5:
            density = dens_grid[ix, iy, iz]
            if density > 0:
                atten = math.exp(-k_base * g_theta * density * path_len)
                old_t = transmittance
                transmittance *= atten
                veg_voxels += 1
                if veg_voxels <= 20:  # print first 20 vegetation voxels
                    print(f"    step {step_count:3d}: [{ix:4d},{iy:4d},{iz:4d}] {cls_name:8s}  "
                          f"LAD={density:.4f}  path={path_len:.3f}m  "
                          f"atten={atten:.4f}  T: {old_t:.4f} → {transmittance:.4f}")

        if transmittance < 1e-6:
            print(f"    → T < 1e-6, stopped.")
            return 0.0

        current_t = next_t
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                ix += step_x; t_max_x += t_delta_x
            else:
                iz += step_z; t_max_z += t_delta_z
        else:
            if t_max_y < t_max_z:
                iy += step_y; t_max_y += t_delta_y
            else:
                iz += step_z; t_max_z += t_delta_z
        step_count += 1

    print(f"    → Exited grid after {step_count} steps, {veg_voxels} veg voxels")
    return transmittance


# ============================================================================
# MAIN
# ============================================================================

def analyze_file(lidar_path, voxel_size):
    """Load, voxelize, and trace a test ray."""
    print(f"\n{'='*70}")
    print(f"  FILE: {lidar_path}")
    print(f"  VOXEL SIZE: {voxel_size}m")
    print(f"{'='*70}")

    las = laspy.read(lidar_path)
    pts = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    cls = np.array(las.classification, dtype=np.int8)

    # Filter
    mask = np.isin(cls, list(RELEVANT_CLASSES))
    pts, cls = pts[mask], cls[mask]
    print(f"  Points: {len(pts):,}")

    # Class breakdown
    for c in sorted(RELEVANT_CLASSES):
        n = (cls == c).sum()
        print(f"    Class {c}: {n:,} ({100*n/len(cls):.1f}%)")

    # Voxelize
    class_grid, dens_grid, scene_min, grid_dims, api_grid, col_lai, veg2d, tot2d = \
        voxelize(pts, cls, voxel_size)

    nx, ny, nz = grid_dims
    print(f"  Grid: {nx}×{ny}×{nz} = {nx*ny*nz:,} voxels")

    # Find array position
    dists = np.linalg.norm(pts[:, :2] - TARGET_COORDS_2D, axis=1)
    roof_mask = (dists < ROOF_SEARCH_RADIUS) & (cls == BUILDING_CLASS)
    if roof_mask.any():
        target_z = np.max(pts[roof_mask, 2]) + OFFSET_FROM_ROOF
    else:
        target_z = np.median(pts[cls == GROUND_CLASS, 2]) + 5.0
    origin = np.array([TARGET_COORDS_2D[0], TARGET_COORDS_2D[1], target_z])
    print(f"  Array origin: ({origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f})")

    # Column stats near the array
    oi = np.clip(int((origin[0] - scene_min[0]) / voxel_size), 0, nx - 1)
    oj = np.clip(int((origin[1] - scene_min[1]) / voxel_size), 0, ny - 1)
    print(f"\n  Column at array [{oi},{oj}]:")
    print(f"    API: {api_grid[oi, oj]:.4f}")
    print(f"    LAI_e: {col_lai[oi, oj]:.4f}")
    print(f"    Veg points (2D): {veg2d[oi, oj]}")
    print(f"    Total points (2D): {tot2d[oi, oj]}")

    # Columns along the ray direction (east, ~azi 76°)
    print(f"\n  Columns along ray (azi={TEST_AZIMUTH_DEG}°, stepping east):")
    az_rad = np.radians(TEST_AZIMUTH_DEG)
    for dist in [5, 10, 15, 20, 25, 30]:
        cx = origin[0] + dist * np.sin(az_rad)
        cy = origin[1] + dist * np.cos(az_rad)
        ci = np.clip(int((cx - scene_min[0]) / voxel_size), 0, nx - 1)
        cj = np.clip(int((cy - scene_min[1]) / voxel_size), 0, ny - 1)
        api_val = api_grid[ci, cj]
        lai_val = col_lai[ci, cj]
        veg_val = veg2d[ci, cj]
        tot_val = tot2d[ci, cj]
        max_lad = dens_grid[ci, cj, :].max()
        n_veg_z = (dens_grid[ci, cj, :] > 0).sum()
        print(f"    {dist:2d}m away [{ci},{cj}]: API={api_val:.3f}  LAI_e={lai_val:.2f}  "
              f"veg_pts={veg_val:4d}  total={tot_val:4d}  "
              f"max_LAD={max_lad:.3f}  veg_layers={n_veg_z}")

    # Trace the test ray
    el_rad = np.radians(TEST_ALTITUDE_DEG)
    az_rad = np.radians(TEST_AZIMUTH_DEG)
    direction = np.array([
        np.cos(el_rad) * np.sin(az_rad),
        np.cos(el_rad) * np.cos(az_rad),
        np.sin(el_rad),
    ])

    print(f"\n  --- RAY TRACE: alt={TEST_ALTITUDE_DEG}°, azi={TEST_AZIMUTH_DEG}° ---")
    T = trace_ray_verbose(origin, direction, scene_min, voxel_size, grid_dims,
                          class_grid, dens_grid, K_BASE)

    print(f"\n  FINAL TRANSMITTANCE: {T:.4f}")
    print(f"  SHADOW FACTOR: {1 - T:.4f}")
    return T


if __name__ == "__main__":
    import os

    results = {}

    for lidar_path in [LIDAR_NEW, LIDAR_OLD]:
        if not os.path.exists(lidar_path):
            print(f"\n  SKIPPING {lidar_path} (not found)")
            continue
        for vs in VOXEL_SIZES:
            key = f"{os.path.basename(lidar_path)} @ {vs}m"
            T = analyze_file(lidar_path, vs)
            results[key] = T

    print(f"\n{'='*70}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*70}")
    for key, T in results.items():
        print(f"  {key:45s}  T={T:.4f}  shadow={1-T:.4f}")
