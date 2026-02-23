"""
Panel Z-Position Diagnostic
=============================
Checks where the panels are placed relative to the building
and surrounding vegetation to determine the correct OFFSET_FROM_ROOF
and skip_dist parameters.

Usage: python diagnose_z_position.py
"""

import numpy as np
import laspy

LIDAR_FILE = "output/reclassified_final_v5.laz"
TARGET_XY = np.array([532882.50, 6983507.00])
SEARCH_RADIUS = 3.0
VOXEL_SIZE = 2.0

GROUND_CLASS = 2
BUILDING_CLASS = 6
VEG_CLASSES = {3, 4, 5}

print("=" * 70)
print("  PANEL Z-POSITION DIAGNOSTIC")
print("=" * 70)

las = laspy.read(LIDAR_FILE)
pts = np.vstack((las.x, las.y, las.z)).T
cls = np.array(las.classification)

dists = np.linalg.norm(pts[:, :2] - TARGET_XY, axis=1)

# --- Building points near target ---
print(f"\n--- BUILDING POINTS within {SEARCH_RADIUS}m of target ---")
bld_mask = (dists < SEARCH_RADIUS) & (cls == BUILDING_CLASS)
bld_pts = pts[bld_mask]
if len(bld_pts) > 0:
    print(f"  Count: {len(bld_pts)}")
    print(f"  Z range: {bld_pts[:,2].min():.2f} – {bld_pts[:,2].max():.2f} m")
    print(f"  Z mean: {bld_pts[:,2].mean():.2f} m")
    print(f"  Z median: {np.median(bld_pts[:,2]):.2f} m")
    roof_max = bld_pts[:,2].max()
    print(f"  Roof top (max Z): {roof_max:.2f} m")
else:
    print("  NO building points found!")
    roof_max = None

# --- Ground points near target ---
print(f"\n--- GROUND POINTS within {SEARCH_RADIUS}m of target ---")
gnd_mask = (dists < SEARCH_RADIUS) & (cls == GROUND_CLASS)
gnd_pts = pts[gnd_mask]
if len(gnd_pts) > 0:
    print(f"  Count: {len(gnd_pts)}")
    print(f"  Z range: {gnd_pts[:,2].min():.2f} – {gnd_pts[:,2].max():.2f} m")
    ground_z = np.median(gnd_pts[:,2])
    print(f"  Ground level (median Z): {ground_z:.2f} m")
    if roof_max:
        print(f"  Building height above ground: {roof_max - ground_z:.2f} m")
else:
    print("  NO ground points found near target")
    # Try wider search
    gnd_mask_wide = (dists < 10) & (cls == GROUND_CLASS)
    gnd_pts_wide = pts[gnd_mask_wide]
    if len(gnd_pts_wide) > 0:
        ground_z = np.median(gnd_pts_wide[:,2])
        print(f"  Ground level (10m radius median): {ground_z:.2f} m")
    else:
        ground_z = None

# --- Voxel analysis ---
if roof_max is not None:
    print(f"\n--- VOXEL PLACEMENT ANALYSIS (voxel_size={VOXEL_SIZE}m) ---")
    
    scene_min_z = pts[:,2].min()
    
    # Which voxel Z-layer is the roof in?
    roof_voxel_z = int((roof_max - scene_min_z) / VOXEL_SIZE)
    roof_voxel_z_bottom = roof_voxel_z * VOXEL_SIZE + scene_min_z
    roof_voxel_z_top = (roof_voxel_z + 1) * VOXEL_SIZE + scene_min_z
    
    print(f"  Scene min Z: {scene_min_z:.2f} m")
    print(f"  Roof max Z: {roof_max:.2f} m")
    print(f"  Roof is in voxel layer {roof_voxel_z}")
    print(f"    Voxel Z range: {roof_voxel_z_bottom:.2f} – {roof_voxel_z_top:.2f} m")
    
    for offset in [-2.5, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        panel_z = roof_max + offset
        panel_voxel_z = int((panel_z - scene_min_z) / VOXEL_SIZE)
        panel_voxel_bottom = panel_voxel_z * VOXEL_SIZE + scene_min_z
        panel_voxel_top = (panel_voxel_z + 1) * VOXEL_SIZE + scene_min_z
        same_as_roof = " ← SAME VOXEL AS ROOF" if panel_voxel_z == roof_voxel_z else ""
        above_roof = " ← ONE ABOVE ROOF" if panel_voxel_z == roof_voxel_z + 1 else ""
        
        print(f"  OFFSET={offset:+.1f}m → panel_z={panel_z:.2f}m → "
              f"voxel layer {panel_voxel_z} "
              f"({panel_voxel_bottom:.2f}–{panel_voxel_top:.2f}m)"
              f"{same_as_roof}{above_roof}")

# --- Nearest vegetation ---
print(f"\n--- NEAREST VEGETATION ---")
for direction, angle in [("East (azi~90°)", 90), ("NE (azi~45°)", 45), 
                          ("West (azi~270°)", 270), ("South (azi~180°)", 180)]:
    az_rad = np.radians(angle)
    dx = np.sin(az_rad)
    dy = np.cos(az_rad)
    
    veg_mask = np.isin(cls, list(VEG_CLASSES))
    veg_pts = pts[veg_mask]
    
    # Project vegetation onto the direction from target
    rel = veg_pts[:, :2] - TARGET_XY
    proj = rel[:, 0] * dx + rel[:, 1] * dy  # distance along direction
    perp = np.abs(rel[:, 0] * dy - rel[:, 1] * dx)  # perpendicular distance
    
    # Trees within ±5m of the ray and in the forward direction
    near_mask = (proj > 0) & (perp < 5)
    if near_mask.any():
        near_veg = veg_pts[near_mask]
        near_proj = proj[near_mask]
        closest_idx = np.argmin(near_proj)
        closest_dist = near_proj[closest_idx]
        closest_z = near_veg[closest_idx, 2]
        max_z_near = near_veg[:, 2].max()
        
        # How many within skip_dist of 11m?
        within_skip = (near_proj < 11).sum()
        
        print(f"  {direction:20s}: closest veg at {closest_dist:.1f}m "
              f"(z={closest_z:.1f}m), tallest={max_z_near:.1f}m, "
              f"{within_skip} pts within 11m skip zone")
    else:
        print(f"  {direction:20s}: no vegetation in this direction within 5m corridor")

print(f"\n--- RECOMMENDATION ---")
if roof_max and ground_z:
    ideal_offset = VOXEL_SIZE - (roof_max - scene_min_z) % VOXEL_SIZE + 0.1
    print(f"  To place panels just above the roof voxel layer:")
    print(f"    OFFSET_FROM_ROOF = {ideal_offset:.1f}m")
    print(f"    This puts panels at z={roof_max + ideal_offset:.2f}m, "
          f"in voxel layer {int((roof_max + ideal_offset - scene_min_z) / VOXEL_SIZE)}")
    print(f"  With this offset, skip_dist can be reduced to {VOXEL_SIZE * 1.5:.1f}m")