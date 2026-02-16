import os
import time
import math
import laspy
import numpy as np
import pandas as pd
from numba import njit
from joblib import Parallel, delayed

# --- CONFIGURATION (Zhang et al. 2024 Boreal Calibration) ---

BETA_FINLAND = 2.08 
K_BASE_FINLAND = 1.0 / BETA_FINLAND 
OMEGA_S = 0.56 

TARGET_COORDS_2D = np.array([532884, 6983510])
ROOF_SEARCH_RADIUS = 2.0  
OFFSET_FROM_ROOF = 0.1    

GROUND_CLASS = 2
BUILDING_CLASS = 6
VEGETATION_CLASSES = {3, 4, 5}
RELEVANT_CLASSES = {2, 3, 4, 5, 6}
CLASS_PRIORITY = {6: 5, 5: 4, 4: 3, 3: 2, 2: 1, 0: 0}

AZIMUTH_STEPS = 360  
ELEVATION_STEPS = 91 
SOLAR_ANGULAR_RADIUS_DEG = 0.265
NUM_RAYS_PER_CONE = 16

# --- MODULE 1: DATA INGESTOR & PREPARER ---

def load_and_prepare_lidar(file_path, bounding_box=None, relevant_classes=None):
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


# --- MODULE 2: VOXELIZER ---

def voxelize_scene(points, classifications, voxel_size):
    scene_min = np.min(points, axis=0)
    grid_dims = np.ceil((np.max(points, axis=0) - scene_min) / voxel_size).astype(int)
    
    classification_grid = np.zeros(grid_dims, dtype=np.int8)
    density_grid = np.zeros(grid_dims, dtype=np.float32)
    voxel_indices = np.floor((points - scene_min) / voxel_size).astype(int)

    for i in range(len(points)):
        idx = tuple(np.clip(voxel_indices[i], 0, grid_dims - 1))
        if CLASS_PRIORITY.get(classifications[i], 0) > CLASS_PRIORITY.get(classification_grid[idx], 0):
            classification_grid[idx] = classifications[i]
        
        if classifications[i] in VEGETATION_CLASSES:
            density_grid[idx] += 1

    for class_to_extrude in [GROUND_CLASS, BUILDING_CLASS]:
        label = "ground" if class_to_extrude == GROUND_CLASS else "building"
        print(f"Extruding {label} footprints to create solid models...")
        
        indices = np.argwhere(classification_grid == class_to_extrude)
        for ix, iy, iz in indices:
            for z_level in range(iz - 1, -1, -1):
                if CLASS_PRIORITY.get(classification_grid[ix, iy, z_level], -1) < CLASS_PRIORITY[class_to_extrude]:
                    classification_grid[ix, iy, z_level] = class_to_extrude
                else:
                    break

    vegetation_mask = np.isin(classification_grid, list(VEGETATION_CLASSES))
    density_grid[vegetation_mask] /= (voxel_size ** 3)
    
    return classification_grid, density_grid, scene_min, grid_dims


# --- MODULE 3: GEOMETRY & MATH ---

def generate_cone_vectors(center_direction, radius_rad, num_samples):
    if np.allclose(np.abs(center_direction), [0, 0, 1]):
        v_up = np.array([0, 1, 0])
    else:
        v_up = np.array([0, 0, 1])
    
    u = np.cross(v_up, center_direction)
    u /= np.linalg.norm(u)
    v = np.cross(center_direction, u)

    cone_vectors = []
    for i in range(num_samples):
        r = radius_rad * np.sqrt((i + 0.5) / num_samples)
        theta = 2 * np.pi * 0.61803398875 * i 
        offset_vec = r * (np.cos(theta) * u + np.sin(theta) * v)
        new_vec = center_direction + offset_vec
        new_vec /= np.linalg.norm(new_vec) 
        cone_vectors.append(new_vec)
        
    return cone_vectors

@njit(fastmath=True)
def calculate_ray_transmittance(
    ray_origin, ray_direction, scene_min, voxel_size, grid_dims, 
    classification_grid, density_grid, k_eff, ground_class, building_class, buffer_dist=0.1
):
    adjusted_origin = ray_origin + (ray_direction * buffer_dist)
    
    ray_pos = (adjusted_origin - scene_min) / voxel_size
    ix = int(math.floor(ray_pos[0]))
    iy = int(math.floor(ray_pos[1]))
    iz = int(math.floor(ray_pos[2]))
    
    if not (0 <= ix < grid_dims[0] and 0 <= iy < grid_dims[1] and 0 <= iz < grid_dims[2]):
        return 1.0 

    step_x = 1 if ray_direction[0] >= 0 else -1
    step_y = 1 if ray_direction[1] >= 0 else -1
    step_z = 1 if ray_direction[2] >= 0 else -1
    
    t_delta_x = voxel_size / abs(ray_direction[0]) if ray_direction[0] != 0 else np.inf
    t_delta_y = voxel_size / abs(ray_direction[1]) if ray_direction[1] != 0 else np.inf
    t_delta_z = voxel_size / abs(ray_direction[2]) if ray_direction[2] != 0 else np.inf
    
    next_boundary_x = (ix + (1 if step_x > 0 else 0)) * voxel_size + scene_min[0]
    next_boundary_y = (iy + (1 if step_y > 0 else 0)) * voxel_size + scene_min[1]
    next_boundary_z = (iz + (1 if step_z > 0 else 0)) * voxel_size + scene_min[2]

    t_max_x = (next_boundary_x - adjusted_origin[0]) / ray_direction[0] if ray_direction[0] != 0 else np.inf
    t_max_y = (next_boundary_y - adjusted_origin[1]) / ray_direction[1] if ray_direction[1] != 0 else np.inf
    t_max_z = (next_boundary_z - adjusted_origin[2]) / ray_direction[2] if ray_direction[2] != 0 else np.inf

    transmittance = 1.0
    current_t = 0.0  
    
    while True:
        if not (0 <= ix < grid_dims[0] and 0 <= iy < grid_dims[1] and 0 <= iz < grid_dims[2]):
            break
            
        voxel_class = classification_grid[ix, iy, iz]
        
        next_t = min(t_max_x, t_max_y, t_max_z)
        path_length = next_t - current_t 
        
        if voxel_class == building_class or voxel_class == ground_class:
            return 0.0
            
        if 3 <= voxel_class <= 5: 
            density = density_grid[ix, iy, iz]
            if density > 0:
                transmittance *= math.exp(-k_eff * density * path_length)
                
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


# --- MODULE 4: SIMULATION ORCHESTRATION ---

def process_single_solar_position(
    el, az, analysis_point, scene_min, voxel_size, grid_dims, 
    classification_grid, density_grid, k_eff, solar_radius_rad, buffer_dist
):
    """Worker function to process a single solar angle (designed for parallel execution)."""
    if el < 0.001: 
        return {'azimuth': az, 'elevation': el, 'transmittance': 0.0}

    center_ray_direction = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
    cone_ray_vectors = generate_cone_vectors(center_ray_direction, solar_radius_rad, NUM_RAYS_PER_CONE)
    
    cone_transmittances = []
    for ray_vec in cone_ray_vectors:
        transmittance = calculate_ray_transmittance(
            analysis_point, ray_vec, scene_min, voxel_size, grid_dims, 
            classification_grid, density_grid, k_eff, 
            GROUND_CLASS, BUILDING_CLASS, buffer_dist
        )
        cone_transmittances.append(transmittance)
    
    avg_transmittance = np.mean(cone_transmittances)
    return {'azimuth': az, 'elevation': el, 'transmittance': avg_transmittance}


def create_shadow_matrix(lidar_file_path=None, voxel_size=1.0, output_directory=None, output_filename=None, buffer_dist=0.1, offset_target_z=0.0):
    start_time = time.time()
    
    points, classifications = load_and_prepare_lidar(lidar_file_path, relevant_classes=RELEVANT_CLASSES)
    if points is None: return None

    classification_grid, density_grid, scene_min, grid_dims = voxelize_scene(points, classifications, voxel_size)

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
    print(f"Analysis point set to: {analysis_point}")

    azimuths = np.linspace(0, 2 * np.pi, AZIMUTH_STEPS, endpoint=False)
    elevations = np.linspace(0, np.pi / 2, ELEVATION_STEPS, endpoint=True)
    solar_radius_rad = np.deg2rad(SOLAR_ANGULAR_RADIUS_DEG)
    k_eff = K_BASE_FINLAND * OMEGA_S

    # Prepare parameter list for parallel processing
    tasks = [(el, az) for el in elevations for az in azimuths]
    total_directions = len(tasks)
    
    print(f"\n--- Starting PARALLEL Cone-Casting Simulation ---")
    print(f"Casting {NUM_RAYS_PER_CONE} rays per cone for {total_directions} positions...")

    # Run ray-casting in parallel across all available CPU cores
    simulation_results = Parallel(n_jobs=-1, batch_size="auto")(
        delayed(process_single_solar_position)(
            el, az, analysis_point, scene_min, voxel_size, grid_dims, 
            classification_grid, density_grid, k_eff, solar_radius_rad, buffer_dist
        ) for el, az in tasks
    )

    print("\n--- Aggregating Results into CSV Matrix ---")
    df = pd.DataFrame(simulation_results)
    df['azimuth_deg'] = np.round(np.rad2deg(df['azimuth'])).astype(int)
    df['elevation_deg'] = np.round(np.rad2deg(df['elevation'])).astype(int)
    
    shadow_matrix_df = df.pivot_table(
        index='elevation_deg', columns='azimuth_deg', values='transmittance'
    )
    
    shadow_matrix_df.index = [f"Altitude_{i}" for i in shadow_matrix_df.index]
    shadow_matrix_df.columns = [f"Azimuth_{c}" for c in shadow_matrix_df.columns]
    
    shadow_matrix_df = shadow_matrix_df.sort_index(key=lambda x: np.array([int(i.split('_')[1]) for i in x]))
    shadow_matrix_df = shadow_matrix_df.sort_index(axis=1, key=lambda x: np.array([int(i.split('_')[1]) for i in x]))

    shadow_matrix_df = 1 - shadow_matrix_df  

    if 'Azimuth_0' in shadow_matrix_df.columns:
        shadow_matrix_df['Azimuth_360'] = shadow_matrix_df['Azimuth_0']
    
    end_time = time.time()
    print(f"\n--- Simulation Finished ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    output_path = os.path.join(output_directory, output_filename)
    shadow_matrix_df.to_csv(output_path, header=True, index=True)
    
    print(f"Saved shadow attenuation matrix to {output_path}")

    return shadow_matrix_df

if __name__ == '__main__':
    LIDAR_FILE_PATH = 'data/recovered_SE_n2.laz'
    OUTPUT_DIRECTORY = 'results/shadow_matrix_results_re_SE_new'
    OUTPUT_FILENAME = 'shadow_attenuation_matrix_conecasting_re_SE_n2.csv'

    create_shadow_matrix(
        lidar_file_path=LIDAR_FILE_PATH,
        voxel_size=1.0,
        output_directory=OUTPUT_DIRECTORY,
        output_filename=OUTPUT_FILENAME
    )