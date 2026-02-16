import os
import time
import math
import laspy
import numpy as np
import pandas as pd
from numba import njit
from joblib import Parallel, delayed

# --- CONFIGURATION (Boreal Calibration) ---

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

# --- MODULE 1: DATA INGESTOR ---

def load_and_prepare_lidar(file_path, relevant_classes=None):
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

    if relevant_classes:
        mask = np.isin(classifications, list(relevant_classes))
        points_xyz = points_xyz[mask]
        classifications = classifications[mask]

    print(f"Data loaded and filtered. {len(points_xyz)} points remaining.")
    return points_xyz, classifications


# --- MODULE 2: VOXELIZER & GEOMETRY ---

def voxelize_scene(points, classifications, voxel_size, beta=BETA_FINLAND):
    scene_min = np.min(points, axis=0)
    grid_dims = np.ceil((np.max(points, axis=0) - scene_min) / voxel_size).astype(int)
    nx, ny, nz = grid_dims
    
    class_grid = np.zeros(grid_dims, dtype=np.int8)
    dens_grid = np.zeros(grid_dims, dtype=np.float32)
    voxel_indices = np.clip(np.floor((points - scene_min) / voxel_size).astype(int), 0, grid_dims - 1)

    ground_counts = np.zeros((nx, ny), dtype=np.int32)
    total_counts = np.zeros((nx, ny), dtype=np.int32)
    veg_counts_2d = np.zeros((nx, ny), dtype=np.int32)

    for i in range(len(points)):
        ix, iy, iz = voxel_indices[i]
        current_class = classifications[i]
        
        if CLASS_PRIORITY.get(current_class, 0) > CLASS_PRIORITY.get(class_grid[ix, iy, iz], 0):
            class_grid[ix, iy, iz] = current_class
            
        total_counts[ix, iy] += 1
        if current_class == GROUND_CLASS:
            ground_counts[ix, iy] += 1
        elif current_class in VEGETATION_CLASSES:
            dens_grid[ix, iy, iz] += 1  
            veg_counts_2d[ix, iy] += 1

    api_grid = np.ones((nx, ny), dtype=np.float32)
    valid_cols = total_counts > 0
    api_grid[valid_cols] = ground_counts[valid_cols] / total_counts[valid_cols]
    api_grid = np.clip(api_grid, 0.01, 1.0)
    
    column_lai_e = -beta * np.log(api_grid)

    for ix in range(nx):
        for iy in range(ny):
            if veg_counts_2d[ix, iy] > 0:
                foliage_fraction = dens_grid[ix, iy, :] / veg_counts_2d[ix, iy]
                dens_grid[ix, iy, :] = foliage_fraction * (column_lai_e[ix, iy] / voxel_size)

    for class_ext in [GROUND_CLASS, BUILDING_CLASS]:
        indices = np.argwhere(class_grid == class_ext)
        for ix, iy, iz in indices:
            for z_level in range(iz - 1, -1, -1):
                if CLASS_PRIORITY.get(class_grid[ix, iy, z_level], -1) < CLASS_PRIORITY[class_ext]:
                    class_grid[ix, iy, z_level] = class_ext
                else: 
                    break

    return class_grid, dens_grid, scene_min, grid_dims


def generate_pv_array_points(center_coords, tilt_deg=12, az_deg=170, panel_width_m=1.0, panel_height_m=1.6, row_configuration=(5, 4, 3)):
    tilt_rad = np.radians(tilt_deg)
    math_az_rad = np.radians(90 - az_deg)

    local_points = []
    num_rows = len(row_configuration)
    
    total_height = (num_rows - 1) * panel_height_m
    y_steps = np.linspace(total_height / 2, -total_height / 2, num_rows)
    
    for i, num_panels in enumerate(row_configuration):
        y = y_steps[i]
        if num_panels == 1:
            x_steps = [0.0]
        else:
            start_x = - (num_panels - 1) * panel_width_m / 2
            end_x = (num_panels - 1) * panel_width_m / 2
            x_steps = np.linspace(start_x, end_x, num_panels)
            
        for x in x_steps:
            local_points.append([x, y, 0.0])
            
    local_points = np.array(local_points)

    R_tilt = np.array([
        [1, 0, 0],
        [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
        [0, np.sin(tilt_rad),  np.cos(tilt_rad)]
    ])
    
    R_az = np.array([
        [np.cos(math_az_rad), -np.sin(math_az_rad), 0],
        [np.sin(math_az_rad),  np.cos(math_az_rad), 0],
        [0, 0, 1]
    ])

    world_points = ((R_az @ R_tilt) @ local_points.T).T + center_coords
    return world_points

def generate_cone_vectors(center_direction, radius_rad, num_samples):
    v_up = np.array([0, 1, 0]) if np.allclose(np.abs(center_direction), [0, 0, 1]) else np.array([0, 0, 1])
    u = np.cross(v_up, center_direction)
    u /= np.linalg.norm(u)
    v = np.cross(center_direction, u)

    return [center_direction + (radius_rad * np.sqrt((i + 0.5) / num_samples) * (np.cos(2 * np.pi * 0.618034 * i) * u + np.sin(2 * np.pi * 0.618034 * i) * v)) 
           for i in range(num_samples)]

@njit(fastmath=True)
def calculate_ray_transmittance(origin, direction, scene_min, voxel_size, grid_dims, class_grid, dens_grid, k_eff, g_class, b_class, buffer_dist=0.1):
    adj_origin = origin + (direction * buffer_dist)
    ray_pos = (adj_origin - scene_min) / voxel_size
    ix, iy, iz = int(math.floor(ray_pos[0])), int(math.floor(ray_pos[1])), int(math.floor(ray_pos[2]))
    
    if not (0 <= ix < grid_dims[0] and 0 <= iy < grid_dims[1] and 0 <= iz < grid_dims[2]): return 1.0 

    step_x, step_y, step_z = (1 if direction[0] >= 0 else -1), (1 if direction[1] >= 0 else -1), (1 if direction[2] >= 0 else -1)
    t_delta_x = voxel_size / abs(direction[0]) if direction[0] != 0 else np.inf
    t_delta_y = voxel_size / abs(direction[1]) if direction[1] != 0 else np.inf
    t_delta_z = voxel_size / abs(direction[2]) if direction[2] != 0 else np.inf
    
    t_max_x = ((ix + (1 if step_x > 0 else 0)) * voxel_size + scene_min[0] - adj_origin[0]) / direction[0] if direction[0] != 0 else np.inf
    t_max_y = ((iy + (1 if step_y > 0 else 0)) * voxel_size + scene_min[1] - adj_origin[1]) / direction[1] if direction[1] != 0 else np.inf
    t_max_z = ((iz + (1 if step_z > 0 else 0)) * voxel_size + scene_min[2] - adj_origin[2]) / direction[2] if direction[2] != 0 else np.inf

    transmittance, current_t = 1.0, 0.0  
    while True:
        if not (0 <= ix < grid_dims[0] and 0 <= iy < grid_dims[1] and 0 <= iz < grid_dims[2]): break
        
        v_class = class_grid[ix, iy, iz]
        next_t = min(t_max_x, t_max_y, t_max_z)
        path_len = next_t - current_t 
        
        if v_class == b_class or v_class == g_class: return 0.0
        if 3 <= v_class <= 5: 
            density = dens_grid[ix, iy, iz]
            if density > 0: transmittance *= math.exp(-k_eff * density * path_len)
        if transmittance < 1e-6: return 0.0
            
        current_t = next_t 
        if t_max_x < t_max_y:
            if t_max_x < t_max_z: ix += step_x; t_max_x += t_delta_x
            else: iz += step_z; t_max_z += t_delta_z
        else:
            if t_max_y < t_max_z: iy += step_y; t_max_y += t_delta_y
            else: iz += step_z; t_max_z += t_delta_z
                
    return transmittance


# --- MODULE 3: SIMULATION ORCHESTRATION ---

def process_single_solar_position(el, az, array_points, scene_min, voxel_size, grid_dims, class_grid, dens_grid, k_eff, sol_rad, buf_dist):
    if el < 0.001: return {'azimuth': az, 'elevation': el, 'transmittance': 0.0}

    center_dir = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
    cone_vecs = generate_cone_vectors(center_dir, sol_rad, NUM_RAYS_PER_CONE)
    
    panel_transmittances = []
    for point in array_points:
        cone_trans = [calculate_ray_transmittance(point, v, scene_min, voxel_size, grid_dims, class_grid, dens_grid, k_eff, GROUND_CLASS, BUILDING_CLASS, buf_dist) for v in cone_vecs]
        panel_transmittances.append(np.mean(cone_trans))
    
    return {'azimuth': az, 'elevation': el, 'transmittance': np.mean(panel_transmittances)}

def create_shadow_matrix(lidar_file_path=None, voxel_size=1.0, output_dir=None, output_fn=None, buf_dist=0.1):
    start_time = time.time()
    points, classifications = load_and_prepare_lidar(lidar_file_path, relevant_classes=RELEVANT_CLASSES)
    if points is None: return None

    class_grid, dens_grid, scene_min, grid_dims = voxelize_scene(points, classifications, voxel_size, BETA_FINLAND)

    mask = (np.linalg.norm(points[:, :2] - TARGET_COORDS_2D, axis=1) < ROOF_SEARCH_RADIUS)
    roof_points = points[mask & (classifications == BUILDING_CLASS)]
    target_z = np.max(roof_points[:, 2]) + OFFSET_FROM_ROOF if len(roof_points) > 0 else scene_min[2]
    
    analysis_center = np.array([TARGET_COORDS_2D[0], TARGET_COORDS_2D[1], target_z])
    array_points = generate_pv_array_points(analysis_center, tilt_deg=12, az_deg=170, panel_width_m=1.0, panel_height_m=1.6, row_configuration=(5, 4, 3))
    
    tasks = [(el, az) for el in np.linspace(0, np.pi / 2, ELEVATION_STEPS, endpoint=True) for az in np.linspace(0, 2 * np.pi, AZIMUTH_STEPS, endpoint=False)]
    k_eff = K_BASE_FINLAND * OMEGA_S

    print(f"\n--- Starting PARALLEL Cone-Casting Simulation ---")
    results = Parallel(n_jobs=-1, batch_size="auto")(
        delayed(process_single_solar_position)(el, az, array_points, scene_min, voxel_size, grid_dims, class_grid, dens_grid, k_eff, np.deg2rad(SOLAR_ANGULAR_RADIUS_DEG), buf_dist) 
        for el, az in tasks
    )

    df = pd.DataFrame(results)
    df['azimuth_deg'], df['elevation_deg'] = np.round(np.rad2deg(df['azimuth'])).astype(int), np.round(np.rad2deg(df['elevation'])).astype(int)
    matrix_df = df.pivot_table(index='elevation_deg', columns='azimuth_deg', values='transmittance')
    
    matrix_df.index, matrix_df.columns = [f"Altitude_{i}" for i in matrix_df.index], [f"Azimuth_{c}" for c in matrix_df.columns]
    matrix_df = 1 - matrix_df  
    if 'Azimuth_0' in matrix_df.columns: matrix_df['Azimuth_360'] = matrix_df['Azimuth_0']
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, output_fn)
        matrix_df.to_csv(out_path, header=True, index=True)
        print(f"Saved shadow attenuation matrix to {out_path}")

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    return matrix_df

if __name__ == '__main__':
    LIDAR_FILE_PATH = 'data/recovered_SE_n2.laz'
    OUTPUT_DIRECTORY = 'results/shadow_matrix_results_re_SE_new'
    OUTPUT_FILENAME = 'shadow_attenuation_matrix_conecasting_re_SE_n2.csv'

    create_shadow_matrix(lidar_file_path=LIDAR_FILE_PATH, voxel_size=1.0, output_dir=OUTPUT_DIRECTORY, output_fn=OUTPUT_FILENAME)