import os
import time
import math
import laspy
import numpy as np
import pandas as pd
from numba import njit

# --- CONFIGURATION (Zhang et al. 2024 Boreal Calibration) ---

# Based on the Outokumpu site (near Kuopio), API-based beta is 2.63[cite: 276, 324, 429].
# This accounts for the vertical (erectophile) foliage common in the region[cite: 324].
BETA_FINLAND = 2.08 
# Effective extinction coefficient (k = 1/beta)[cite: 71, 73].
K_BASE_FINLAND = 1.0 / BETA_FINLAND 

# Shoot-level clumping factor (Omega_s) for Nordic conifers is 0.56.
# Failure to use this leads to 30-70% error in LAI/light models[cite: 48].
OMEGA_S = 0.56 

# Analysis point settings for solar panel simulation
TARGET_COORDS_2D = np.array([532884, 6983510])
ROOF_SEARCH_RADIUS = 2.0  # Search radius for detecting local roof peak
OFFSET_FROM_ROOF = 0.1    # Distance (m) above the roof for the sensor point

# ASPRS Classification Codes
GROUND_CLASS = 2
BUILDING_CLASS = 6
VEGETATION_CLASSES = {3, 4, 5}
RELEVANT_CLASSES = {2, 3, 4, 5, 6}

# Prioritize Buildings (6) over Vegetation for accurate roof segmentation
CLASS_PRIORITY = {6: 5, 5: 4, 4: 3, 3: 2, 2: 1, 0: 0}



# # 1. Input Data and Scene Definition
# # LIDAR_FILE_PATH = 'data/recovered_SE_n2.laz'
# # OUTPUT_DIRECTORY = 'results/shadow_matrix_results_re_SE_new'
# # OUTPUT_FILENAME = 'shadow_attenuation_matrix_conecasting_re_SE_n2.csv'
# # BOUNDING_BOX = None


# # --- NEW: Specific target coordinates for analysis ---
# # EPSG:3067 - ETRS89 / TM35FIN(E,N)
# TARGET_COORDS_2D = np.array([532884, 6983510])

# # ASPRS Standard Classification Codes
# # Including multiple vegetation classes (3: Low, 4: Medium, 5: High)
# RELEVANT_CLASSES = {2, 3, 4, 5, 6}
# GROUND_CLASS_CODE = 2
# BUILDING_CLASS_CODE = 6
# VEGETATION_CLASS_CODES = {3, 4, 5}

# # Priority for voxel classification (higher number = higher priority)
# # High vegetation (5) has priority over medium (4), etc.
# CLASS_PRIORITY = {6: 4, 5: 3, 4: 2, 3: 1, 2: 0, 0: -1}

# 2. Voxelization Parameters
# VOXEL_SIZE = 1.0

# 3. Solar Position & Cone-Casting Simulation Parameters
AZIMUTH_STEPS = 360  # 1-degree steps
ELEVATION_STEPS = 91 # 1-degree steps (0-90 inclusive)

# --- NEW: Cone-Casting Parameters ---
# The sun's angular radius is approx 0.265 degrees.
SOLAR_ANGULAR_RADIUS_DEG = 0.265
# Number of rays to cast to approximate the solar disk. More rays = more
# accurate penumbra but longer computation time.
NUM_RAYS_PER_CONE = 16

# 4. Ray-Casting and Attenuation Parameters
# --- UPDATED: Class-specific extinction coefficients ---
# Assign a different base extinction coefficient (k) to each vegetation class.
# These values may need to be calibrated for specific vegetation types.
# VEGETATION_EXTINCTION_COEFFICIENTS = {
#     3: 0.7,  # k for Low Vegetation
#     4: 0.6,  # k for Medium Vegetation
#     5: 0.7   # k for High Vegetation
# }
VEGETATION_EXTINCTION_COEFFICIENTS = {
    3: K_BASE_FINLAND,  # Low Vegetation
    4: K_BASE_FINLAND,  # Medium Vegetation
    5: K_BASE_FINLAND   # High Vegetation
}

# --- MODULE 1: DATA INGESTOR & PREPARER ---

def load_and_prepare_lidar(file_path, bounding_box=None, relevant_classes=None):
    """
    Reads a LAS/LAZ file, filters points by classification and bounding box.
    """
    print(f"Loading LiDAR data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None, None
    try:
        las = laspy.read(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

    points_xyz = np.vstack((las.x, las.y, las.z)).transpose()
    classifications = np.array(las.classification)

    mask = np.ones(len(points_xyz), dtype=bool)
    if bounding_box:
        mask &= (
            (points_xyz[:, 0] >= bounding_box['X_MIN']) & (points_xyz[:, 0] < bounding_box['X_MAX']) &
            (points_xyz[:, 1] >= bounding_box['Y_MIN']) & (points_xyz[:, 1] < bounding_box['Y_MAX'])
        )
    if relevant_classes:
        mask &= np.isin(classifications, list(relevant_classes))

    filtered_points = points_xyz[mask]
    filtered_classifications = classifications[mask]

    if len(filtered_points) == 0:
        print("Warning: No points remaining after filtering.")
        return None, None
    print(f"Data loaded and filtered. {len(filtered_points)} points remaining.")
    return filtered_points, filtered_classifications


# --- MODULE 2: VOXELIZER (UPDATED) ---

# def voxelize_scene(points, classifications, voxel_size):
#     """
#     Converts a point cloud into classification and density voxel grids.
#     Includes extrusion for both buildings and ground.
#     """
#     if points is None or len(points) == 0:
#         return None, None, None, None

#     print("Voxelizing the scene...")
#     scene_min = np.min(points, axis=0)
#     scene_max = np.max(points, axis=0)
#     grid_dims = np.ceil((scene_max - scene_min) / voxel_size).astype(int)
    
#     classification_grid = np.zeros(grid_dims, dtype=np.int8)
#     density_grid = np.zeros(grid_dims, dtype=np.float32)

#     voxel_indices = np.floor((points - scene_min) / voxel_size).astype(int)
#     for i in range(3):
#         voxel_indices[:, i] = np.clip(voxel_indices[:, i], 0, grid_dims[i] - 1)

#     print("Populating classification and density grids...")
#     for i in range(len(points)):
#         idx = tuple(voxel_indices[i])
#         current_class = classifications[i]
        
#         # Apply class priority logic
#         if CLASS_PRIORITY.get(current_class, -1) > CLASS_PRIORITY.get(classification_grid[idx], -1):
#             classification_grid[idx] = current_class
        
#         if current_class in VEGETATION_CLASS_CODES:
#             density_grid[idx] += 1

#     # --- NEW: Extrude ground and buildings downwards ---
#     # We process ground first to establish the 'floor', then buildings.
    
#     for class_to_extrude in [GROUND_CLASS_CODE, BUILDING_CLASS_CODE]:
#         label = "ground" if class_to_extrude == GROUND_CLASS_CODE else "building"
#         print(f"Extruding {label} footprints to create solid models...")
        
#         indices = np.argwhere(classification_grid == class_to_extrude)
        
#         for ix, iy, iz in indices:
#             # Iterate from the voxel below current down to z=0
#             for z_level in range(iz - 1, -1, -1):
#                 # Only fill if the target voxel has lower priority
#                 if CLASS_PRIORITY.get(classification_grid[ix, iy, z_level], -1) < CLASS_PRIORITY[class_to_extrude]:
#                     classification_grid[ix, iy, z_level] = class_to_extrude
#                 else:
#                     # If we hit something of equal/higher priority, stop extruding down
#                     break

#     voxel_volume = voxel_size ** 3
#     vegetation_voxels = np.isin(classification_grid, list(VEGETATION_CLASS_CODES))
#     density_grid[vegetation_voxels] /= voxel_volume
    
#     print("Voxelization complete.")
#     return classification_grid, density_grid, scene_min, grid_dims

def voxelize_scene(points, classifications, voxel_size):
    """
    Voxelizes scene with building-first priority to ensure clean roof surfaces.
    """
    scene_min = np.min(points, axis=0)
    grid_dims = np.ceil((np.max(points, axis=0) - scene_min) / voxel_size).astype(int)
    
    classification_grid = np.zeros(grid_dims, dtype=np.int8)
    density_grid = np.zeros(grid_dims, dtype=np.float32)
    voxel_indices = np.floor((points - scene_min) / voxel_size).astype(int)

    for i in range(len(points)):
        idx = tuple(np.clip(voxel_indices[i], 0, grid_dims - 1))
        # Building-priority logic: Ensure vegetation doesn't 'leak' into roof voxels
        if CLASS_PRIORITY.get(classifications[i], 0) > CLASS_PRIORITY.get(classification_grid[idx], 0):
            classification_grid[idx] = classifications[i]
        
        if classifications[i] in VEGETATION_CLASSES:
            density_grid[idx] += 1

        # --- NEW: Extrude ground and buildings downwards ---
    # We process ground first to establish the 'floor', then buildings.
    
    for class_to_extrude in [GROUND_CLASS, BUILDING_CLASS]:
        label = "ground" if class_to_extrude == GROUND_CLASS else "building"
        print(f"Extruding {label} footprints to create solid models...")
        
        indices = np.argwhere(classification_grid == class_to_extrude)
        
        for ix, iy, iz in indices:
            # Iterate from the voxel below current down to z=0
            for z_level in range(iz - 1, -1, -1):
                # Only fill if the target voxel has lower priority
                if CLASS_PRIORITY.get(classification_grid[ix, iy, z_level], -1) < CLASS_PRIORITY[class_to_extrude]:
                    classification_grid[ix, iy, z_level] = class_to_extrude
                else:
                    # If we hit something of equal/higher priority, stop extruding down
                    break

    # Normalise density to points per cubic meter (mimicking API approach) [cite: 207, 423]
    vegetation_mask = np.isin(classification_grid, list(VEGETATION_CLASSES))
    density_grid[vegetation_mask] /= (voxel_size ** 3)
    
    return classification_grid, density_grid, scene_min, grid_dims

# --- MODULE 3: RAY-CASTING & CONE-CASTING ENGINE ---


def generate_cone_vectors(center_direction, radius_rad, num_samples):
    """
    Generates a set of vectors distributed within a cone around a center direction.
    """
    # Create a basis (a coordinate system) aligned with the center_direction
    if np.allclose(np.abs(center_direction), [0, 0, 1]):
        # Handle case where direction is along Z-axis
        v_up = np.array([0, 1, 0])
    else:
        v_up = np.array([0, 0, 1])
    
    u = np.cross(v_up, center_direction)
    u /= np.linalg.norm(u)
    v = np.cross(center_direction, u)

    cone_vectors = []
    # Use stratified sampling for better distribution
    for i in range(num_samples):
        # Sample radius and angle to get points on a disk
        r = radius_rad * np.sqrt((i + 0.5) / num_samples)
        theta = 2 * np.pi * 0.61803398875 * i # Golden angle for good distribution

        # Map disk point to 3D offset and add to center direction
        offset_vec = r * (np.cos(theta) * u + np.sin(theta) * v)
        new_vec = center_direction + offset_vec
        new_vec /= np.linalg.norm(new_vec) # Re-normalize
        cone_vectors.append(new_vec)
        
    return cone_vectors

def trace_ray_fast(ray_origin, ray_direction, scene_min, voxel_size, grid_dims):
    """
    An efficient voxel traversal algorithm (Amanatides-Woo).
    """
    ray_pos = (ray_origin - scene_min) / voxel_size
    ix, iy, iz = int(ray_pos[0]), int(ray_pos[1]), int(ray_pos[2])
    step_x = 1 if ray_direction[0] >= 0 else -1
    step_y = 1 if ray_direction[1] >= 0 else -1
    step_z = 1 if ray_direction[2] >= 0 else -1
    
    next_voxel_boundary_x = (ix + (step_x > 0)) * voxel_size + scene_min[0]
    next_voxel_boundary_y = (iy + (step_y > 0)) * voxel_size + scene_min[1]
    next_voxel_boundary_z = (iz + (step_z > 0)) * voxel_size + scene_min[2]
    
    t_max_x = (next_voxel_boundary_x - ray_origin[0]) / ray_direction[0] if ray_direction[0] != 0 else float('inf')
    t_max_y = (next_voxel_boundary_y - ray_origin[1]) / ray_direction[1] if ray_direction[1] != 0 else float('inf')
    t_max_z = (next_voxel_boundary_z - ray_origin[2]) / ray_direction[2] if ray_direction[2] != 0 else float('inf')
    
    t_delta_x = voxel_size / abs(ray_direction[0]) if ray_direction[0] != 0 else float('inf')
    t_delta_y = voxel_size / abs(ray_direction[1]) if ray_direction[1] != 0 else float('inf')
    t_delta_z = voxel_size / abs(ray_direction[2]) if ray_direction[2] != 0 else float('inf')

    while True:
        if not (0 <= ix < grid_dims[0] and 0 <= iy < grid_dims[1] and 0 <= iz < grid_dims[2]):
            break
        yield (ix, iy, iz)
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

import numpy as np

# --- UPDATED RAY-TRACING ENGINE ---

@njit(fastmath=True)
def calculate_ray_transmittance(
    ray_origin, ray_direction, scene_min, voxel_size, grid_dims, 
    classification_grid, density_grid, k_eff, ground_class, building_class, buffer_dist=0.1
):
    """
    Combined Amanatides-Woo ray traversal and Beer-Lambert attenuation.
    Calculates exact path length through each voxel for accurate physics.
    """
    # 1. Apply buffer to avoid self-shadowing at the origin
    adjusted_origin = ray_origin + (ray_direction * buffer_dist)
    
    # 2. Grid initialization
    ray_pos = (adjusted_origin - scene_min) / voxel_size
    ix = int(math.floor(ray_pos[0]))
    iy = int(math.floor(ray_pos[1]))
    iz = int(math.floor(ray_pos[2]))
    
    # Boundary check: Exit early if the ray starts outside the grid
    if not (0 <= ix < grid_dims[0] and 0 <= iy < grid_dims[1] and 0 <= iz < grid_dims[2]):
        return 1.0 

    # Step directions
    step_x = 1 if ray_direction[0] >= 0 else -1
    step_y = 1 if ray_direction[1] >= 0 else -1
    step_z = 1 if ray_direction[2] >= 0 else -1
    
    # Distance ray must travel to cross one voxel along each axis
    t_delta_x = voxel_size / abs(ray_direction[0]) if ray_direction[0] != 0 else np.inf
    t_delta_y = voxel_size / abs(ray_direction[1]) if ray_direction[1] != 0 else np.inf
    t_delta_z = voxel_size / abs(ray_direction[2]) if ray_direction[2] != 0 else np.inf
    
    # Calculate distance to the first voxel boundaries
    next_boundary_x = (ix + (1 if step_x > 0 else 0)) * voxel_size + scene_min[0]
    next_boundary_y = (iy + (1 if step_y > 0 else 0)) * voxel_size + scene_min[1]
    next_boundary_z = (iz + (1 if step_z > 0 else 0)) * voxel_size + scene_min[2]

    t_max_x = (next_boundary_x - adjusted_origin[0]) / ray_direction[0] if ray_direction[0] != 0 else np.inf
    t_max_y = (next_boundary_y - adjusted_origin[1]) / ray_direction[1] if ray_direction[1] != 0 else np.inf
    t_max_z = (next_boundary_z - adjusted_origin[2]) / ray_direction[2] if ray_direction[2] != 0 else np.inf

    transmittance = 1.0
    current_t = 0.0  # Tracks total distance traveled along the ray
    
    while True:
        # Exit if we step out of the grid bounds
        if not (0 <= ix < grid_dims[0] and 0 <= iy < grid_dims[1] and 0 <= iz < grid_dims[2]):
            break
            
        voxel_class = classification_grid[ix, iy, iz]
        
        # --- NEW: Exact Path Length Calculation ---
        # The distance to the next voxel boundary minus the current distance
        next_t = min(t_max_x, t_max_y, t_max_z)
        path_length = next_t - current_t 
        
        # 1. Opaque objects block all light
        if voxel_class == building_class or voxel_class == ground_class:
            return 0.0
            
        # 2. Semi-transparent vegetation attenuation
        # Assuming classes 3, 4, 5 are vegetation for speed
        if 3 <= voxel_class <= 5: 
            density = density_grid[ix, iy, iz]
            if density > 0:
                transmittance *= math.exp(-k_eff * density * path_length)
                
        # Early exit if light is effectively blocked
        if transmittance < 1e-6:
            return 0.0
            
        # --- Advance to the next voxel ---
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


def trace_ray_with_buffer(ray_origin, ray_direction, scene_min, voxel_size, grid_dims, buffer_dist=0.1):
    """
    Efficient Amanatides-Woo traversal with a starting buffer to ignore self-blocking.
    :param buffer_dist: Distance to "hop" forward before checking for hits.
                        Usually set to roughly 0.5 * voxel_size.
    """
    # 1. Apply the buffer offset
    # Move the origin forward so we aren't stuck inside the starting building voxel
    adjusted_origin = ray_origin + (ray_direction * buffer_dist)
    
    # 2. Standard Amanatides-Woo Initialization
    ray_pos = (adjusted_origin - scene_min) / voxel_size
    ix, iy, iz = int(ray_pos[0]), int(ray_pos[1]), int(ray_pos[2])
    
    # Boundary check: Ensure adjusted origin is still inside the grid
    if not (0 <= ix < grid_dims[0] and 0 <= iy < grid_dims[1] and 0 <= iz < grid_dims[2]):
        return # Ray moved out of bounds before starting

    step_x = 1 if ray_direction[0] >= 0 else -1
    step_y = 1 if ray_direction[1] >= 0 else -1
    step_z = 1 if ray_direction[2] >= 0 else -1
    
    # t_max calculation (distance to next voxel boundary)
    next_voxel_boundary_x = (ix + (step_x > 0)) * voxel_size + scene_min[0]
    next_voxel_boundary_y = (iy + (step_y > 0)) * voxel_size + scene_min[1]
    next_voxel_boundary_z = (iz + (step_z > 0)) * voxel_size + scene_min[2]
    
    t_max_x = (next_voxel_boundary_x - adjusted_origin[0]) / ray_direction[0] if ray_direction[0] != 0 else float('inf')
    t_max_y = (next_voxel_boundary_y - adjusted_origin[1]) / ray_direction[1] if ray_direction[1] != 0 else float('inf')
    t_max_z = (next_voxel_boundary_z - adjusted_origin[2]) / ray_direction[2] if ray_direction[2] != 0 else float('inf')
    
    t_delta_x = voxel_size / abs(ray_direction[0]) if ray_direction[0] != 0 else float('inf')
    t_delta_y = voxel_size / abs(ray_direction[1]) if ray_direction[1] != 0 else float('inf')
    t_delta_z = voxel_size / abs(ray_direction[2]) if ray_direction[2] != 0 else float('inf')

    while True:
        if not (0 <= ix < grid_dims[0] and 0 <= iy < grid_dims[1] and 0 <= iz < grid_dims[2]):
            break
            
        yield (ix, iy, iz)
        
        # Incremental step
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

# def calculate_transmittance(voxel_path_generator, classification_grid, density_grid, voxel_size, k_coeffs):
#     """
#     Updated to treat GROUND_CLASS_CODE as opaque.
#     """
#     transmittance = 1.0
#     path_length_in_voxel = voxel_size
    
#     for ix, iy, iz in voxel_path_generator:
#         voxel_class = classification_grid[ix, iy, iz]
        
#         # Opaque objects: Buildings and Ground
#         if voxel_class == BUILDING_CLASS_CODE or voxel_class == GROUND_CLASS_CODE:
#             return 0.0
        
#         # Semi-transparent objects: Vegetation
#         if voxel_class in VEGETATION_CLASS_CODES:
#             k_base = k_coeffs.get(voxel_class, 0.0)
#             if k_base > 0:
#                 density = density_grid[ix, iy, iz]
#                 if density > 0:
#                     k = k_base * density
#                     transmittance *= math.exp(-k * path_length_in_voxel)
        
#         if transmittance < 1e-6: return 0.0
        
#     return transmittance

def calculate_transmittance(voxel_path_generator, classification_grid, density_grid, voxel_size):
    """
    Calculates light attenuation using the Zhang et al. (2024) semi-physical 
    model logic with clumping corrections for Finland[cite: 71, 182, 411].
    """
    transmittance = 1.0
    
    for ix, iy, iz in voxel_path_generator:
        voxel_class = classification_grid[ix, iy, iz]
        
        # Opaque: Buildings and Ground block 100% of light
        if voxel_class == BUILDING_CLASS or voxel_class == GROUND_CLASS:
            return 0.0
        
        # Semi-transparent: Vegetation attenuation [cite: 182]
        if voxel_class in VEGETATION_CLASSES:
            density = density_grid[ix, iy, iz]
            if density > 0:
                # Correct k for shoot-level clumping (Omega_s = 0.56) 
                k_eff = K_BASE_FINLAND * OMEGA_S
                # Beer-Lambert: exp(-k * L) where L is effective leaf area [cite: 10, 71]
                transmittance *= math.exp(-k_eff * density * voxel_size)
        
        if transmittance < 1e-6: return 0.0
    return transmittance

def create_shadow_matrix(lidar_file_path=None, bounding_box=None, voxel_size=1.0, output_directory=None, output_filename=None, save_csv=True, buffer_dist=0.1, offset_target_z=-1.0):
    """
    Main function to create the shadow matrix.
    """
    start_time = time.time()
    # 1. Load and Prepare Data
    # points, classifications = load_and_prepare_lidar(
    #     lidar_file_path, bounding_box, RELEVANT_CLASSES
    # )
    # if points is None: exit()

    # # 2. Voxelize Scene
    # classification_grid, density_grid, scene_min, grid_dims = voxelize_scene(
    #     points, classifications, voxel_size
    # )
    # if classification_grid is None: exit()
    # scene_max = scene_min + grid_dims * voxel_size

    # # 3. Define the single analysis point using the specified coordinates
    # print(f"Using specified target coordinates: {TARGET_COORDS_2D}")

    # # Find the Z value for the highest point (e.g., a roof) at the target coordinates.
    # search_radius = 3 # 5 meters
    # points_near_target_mask = (
    #     (points[:, 0] > TARGET_COORDS_2D[0] - search_radius) &
    #     (points[:, 0] < TARGET_COORDS_2D[0] + search_radius) &
    #     (points[:, 1] > TARGET_COORDS_2D[1] - search_radius) &
    #     (points[:, 1] < TARGET_COORDS_2D[1] + search_radius)
    # )
    # points_near_target = points[points_near_target_mask]

    # if len(points_near_target) > 0:
    #     # Find the highest Z value in the vicinity of the target
    #     target_z = np.max(points_near_target[:, 2]) # Slightly below the roof
    #     print(f"Found target Z coordinate from LiDAR data: {target_z}")
    # else:
    #     print(f"Warning: No LiDAR points found near target coordinates. Using scene maximum Z as a fallback.")
    #     target_z = scene_max[2]

    # # analysis_point = np.array([TARGET_COORDS_2D[0], TARGET_COORDS_2D[1], np.ceil(target_z)])
    # target_z += offset_target_z
    points, classifications = load_and_prepare_lidar(lidar_file_path, relevant_classes=RELEVANT_CLASSES)
    classification_grid, density_grid, scene_min, grid_dims = voxelize_scene(points, classifications, voxel_size)

    # REFINED ANALYSIS POINT: Find peak roof Z at target XY
    mask = (np.linalg.norm(points[:, :2] - TARGET_COORDS_2D, axis=1) < ROOF_SEARCH_RADIUS)
    roof_points = points[mask & (classifications == BUILDING_CLASS)]
    
    if len(roof_points) > 0:
        target_z = np.max(roof_points[:, 2]) + OFFSET_FROM_ROOF
        print(f"Sensor placed on roof peak: {target_z:.2f}m")
    else:
        target_z = np.max(points[mask][:, 2]) if any(mask) else scene_min[2]
        print(f"No building found. Defaulting to max surface Z: {target_z:.2f}m")

    
    target_z += offset_target_z
    analysis_point = np.array([TARGET_COORDS_2D[0], TARGET_COORDS_2D[1], target_z])
    # analysis_point = np.array([TARGET_COORDS_2D[0], TARGET_COORDS_2D[1], target_z])

    print(f"Analysis point set to: {analysis_point}")

    # 4. Define Solar Angles and Run Simulation Loop (SEQUENTIAL)
    azimuths = np.linspace(0, 2 * np.pi, AZIMUTH_STEPS, endpoint=False)
    elevations = np.linspace(0, np.pi / 2, ELEVATION_STEPS, endpoint=True)
    solar_radius_rad = np.deg2rad(SOLAR_ANGULAR_RADIUS_DEG)
    
    simulation_results = []
    
    total_directions = len(azimuths) * (len(elevations) - 1)
    print(f"\n--- Starting SEQUENTIAL Cone-Casting Simulation ---")
    print(f"Casting {NUM_RAYS_PER_CONE} rays per cone for each of {total_directions} solar positions...")

    current_direction = 0
    for el in elevations:
        if el < 0.001: continue
        
        for az in azimuths:
            current_direction += 1
            # if Azimuth is from 270 to 360, use the target_z + 0.5
            # if 270 <= np.rad2deg(az) < 360:
            #     analysis_point[2] = target_z + 1.0
            # else:
            #     analysis_point[2] = target_z
    
            # if 180 <= np.rad2deg(az) < 270:
            #     analysis_point[2] = target_z + 1.0
            if current_direction % 500 == 0:
                 print(f"Processing direction {current_direction} of {total_directions} (Az: {np.rad2deg(az):.1f}°, El: {np.rad2deg(el):.1f}°)...")
            
            center_ray_direction = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
            
            cone_ray_vectors = generate_cone_vectors(center_ray_direction, solar_radius_rad, NUM_RAYS_PER_CONE)
            
            cone_transmittances = []
            for ray_vec in cone_ray_vectors:
                # if 235 <= np.rad2deg(az) < 360:
                #     analysis_point[2] = target_z - 1.0
                # elif 75 <= np.rad2deg(az) < 120:
                #     analysis_point[2] = target_z - 0.5
                # else:
                #     analysis_point[2] = target_z
                # voxel_size = VOXEL_SIZE
                voxel_path_gen = trace_ray_with_buffer(analysis_point, ray_vec, scene_min, voxel_size, grid_dims, buffer_dist=buffer_dist)
                transmittance = calculate_transmittance(
                    voxel_path_gen, classification_grid, density_grid, voxel_size
                )
                cone_transmittances.append(transmittance)
            
            avg_transmittance = np.mean(cone_transmittances)
            
            simulation_results.append({
                'azimuth': az, 'elevation': el, 'transmittance': avg_transmittance
            })

    # 5. Format and Save Final Attenuation Matrix
    print("\n--- Aggregating Results into CSV Matrix ---")
    if not simulation_results:
        print("No results to save. Exiting.")
        exit()

    df = pd.DataFrame(simulation_results)
    df['azimuth_deg'] = np.round(np.rad2deg(df['azimuth'])).astype(int)
    df['elevation_deg'] = np.round(np.rad2deg(df['elevation'])).astype(int)
    
    shadow_matrix_df = df.pivot_table(
        index='elevation_deg', columns='azimuth_deg', values='transmittance'
    )
    
    # Format headers to match the example output
    shadow_matrix_df.index = [f"Altitude_{i}" for i in shadow_matrix_df.index]
    shadow_matrix_df.columns = [f"Azimuth_{c}" for c in shadow_matrix_df.columns]
    
    # Sort the DataFrame by numeric index and columns to ensure correct order
    shadow_matrix_df = shadow_matrix_df.sort_index(key=lambda x: np.array([int(i.split('_')[1]) for i in x]))
    shadow_matrix_df = shadow_matrix_df.sort_index(axis=1, key=lambda x: np.array([int(i.split('_')[1]) for i in x]))

    shadow_matrix_df = 1- shadow_matrix_df  # Convert to shadow attenuation (1 - transmittance)

        # --- NEW: Add Azimuth_360 column as a copy of Azimuth_0 ---
    if 'Azimuth_0' in shadow_matrix_df.columns:
        shadow_matrix_df['Azimuth_360'] = shadow_matrix_df['Azimuth_0']
    
    end_time = time.time()
    print(f"\n--- Simulation Finished ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    if save_csv:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            
        output_path = os.path.join(output_directory, output_filename)
        # Write to CSV with header and index
        shadow_matrix_df.to_csv(output_path, header=True, index=True)
    
    print(f"Saved shadow attenuation matrix to {output_path}")

    return shadow_matrix_df



if __name__ == '__main__':
    # Example usage
    LIDAR_FILE_PATH = 'data/recovered_SE_n2.laz'
    OUTPUT_DIRECTORY = 'results/shadow_matrix_results_re_SE_new'
    OUTPUT_FILENAME = 'shadow_attenuation_matrix_conecasting_re_SE_n2.csv'
    BOUNDING_BOX = None

    start_time = time.time()
    create_shadow_matrix(
        lidar_file_path=LIDAR_FILE_PATH,
        bounding_box=BOUNDING_BOX,
        voxel_size=1.0,
        output_directory=OUTPUT_DIRECTORY,
        output_filename=OUTPUT_FILENAME,
        save_csv=True
    )