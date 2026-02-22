
import os
import laspy
import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import open3d as o3d
import rasterio
from pyproj import CRS

# %%
def merge_las_files(path1, path2, output_path):
    print(f"Reading {path1} and {path2}...")
    las1 = laspy.read(path1)
    las2 = laspy.read(path2)

    # 1. Ensure point formats match (e.g., both are Format 6 or Format 1)
    if las1.header.point_format.id != las2.header.point_format.id:
        raise ValueError("Point formats do not match! Cannot merge different formats easily.")

    # 2. Setup the new header
    # We use the scales and offsets from the first file as our global reference
    new_header = laspy.LasHeader(
        point_format=las1.header.point_format, 
        version=las1.header.version
    )
    new_header.offsets = las1.header.offsets
    new_header.scales = las1.header.scales

    # 3. CONCATENATE THE UNDERLYING ARRAYS
    # This avoids the ScaleAwarePointRecord TypeError
    merged_array = np.concatenate([las1.points.array, las2.points.array])

    # 4. Create the new LAS object and assign the array
    merged_las = laspy.LasData(new_header)
    merged_las.points = laspy.ScaleAwarePointRecord(
        merged_array, 
        las1.header.point_format, 
        las1.header.scales, 
        las1.header.offsets
    )

    # 5. Write to disk
    print(f"Writing {len(merged_array)} points to {output_path}...")
    merged_las.write(output_path)
    print("Merge successful.")

# --- Usage ---
# merge_las_files("data/tile_A.laz", "data/tile_B.laz", "data/merged_study_area.laz")

def crop_las_file(input_path, output_path, bounds):
    """
    Crops a LAS/LAZ file based on specific coordinate bounds.
    :param bounds: A dictionary with {'min_x', 'max_x', 'min_y', 'max_y'}
    """
    print(f"Loading {input_path}...")
    las = laspy.read(input_path)
    
    # 1. Extract Bounds from the dictionary
    min_x, max_x = bounds['min_x'], bounds['max_x']
    min_y, max_y = bounds['min_y'], bounds['max_y']
    
    print(f"Cropping to custom Bounds:")
    print(f" - X: {min_x} to {max_x}")
    print(f" - Y: {min_y} to {max_y}")

    # 2. Create Boolean Mask
    mask = (
        (las.x >= min_x) & (las.x <= max_x) & 
        (las.y >= min_y) & (las.y <= max_y)
    )
    
    # 3. Apply Mask
    cropped_points = las.points[mask]
    
    if len(cropped_points) == 0:
        print("Error: No points found in the specified range!")
        return

    # 4. Create New LAS Data
    new_header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
    new_header.offsets = las.header.offsets
    new_header.scales = las.header.scales
    
    cropped_las = laspy.LasData(new_header)
    cropped_las.points = cropped_points

    print(f"Saving {len(cropped_points)} points to {output_path}...")
    cropped_las.write(output_path)
    print("Done.")

# --- Usage for your "Study House" Site ---
# Let's say your house is at (532934, 6983793) 
# and you want 50m to the West and 150m to the East (Asymmetrical)

# target_x = 532934.33
# target_y = 6983793.63

# custom_bounds = {
#     'min_x': target_x - 50,   # 50m left
#     'max_x': target_x + 150,  # 150m right
#     'min_y': target_y - 100,  # 100m down
#     'max_y': target_y + 100   # 100m up
# }

# crop_las_file("input.laz", "study_house_area.laz", custom_bounds)

def filter_lidar_data(input_path, output_path, nb_neighbors=20, std_ratio=2.0):
    """
    Processes raw LiDAR: Reads LAS -> Statistical Outlier Removal -> Classification Filter.
    
    Args:
        input_path (str): Path to the raw .las or .laz file.
        output_path (str): Path to save the processed file.
        nb_neighbors (int): K-neighbors for SOR filter.
        std_ratio (float): Standard deviation multiplier for SOR filter.
    """
    print(f"--- Starting Processing for {os.path.basename(input_path)} ---")
    
    # 1. Load LAS file
    las = laspy.read(input_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # 2. Statistical Outlier Removal (SOR) via Open3D
    print("Running Statistical Outlier Removal...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # cl: cleaned cloud (unused here), ind: indices of points to keep
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    # Apply the SOR indices to the original las object to keep all dimensions (intensity, etc.)
    las = las[ind]
    print(f"SOR complete. Removed {len(points) - len(las)} outliers.")

    # 3. Filter by Classification (Classes 2, 3, 4, 5, 6)
    print("Filtering for classes 2, 3, 4, 5, and 6...")
    valid_classes = [2, 3, 4, 5, 6]
    class_mask = np.isin(las.classification, valid_classes)
    
    final_las = las[class_mask]
    
    # 4. Save results
    final_las.write(output_path)
    print(f"Successfully saved {len(final_las)} points to {output_path}")
    print("--- Processing Complete ---\n")


def reclassify_buildings_from_veg_optimized(las, neighbor_k=20, planarity_threshold=0.6):
    """
    Reclassifies points from Class 5 to Class 6 using vectorized geometric planarity.
    Operates directly on the laspy object in memory.
    """
    print("Identifying candidate points...")
    candidate_mask = las.classification == 5
    if np.sum(candidate_mask) == 0:
        candidate_mask = las.classification == 1
        
    candidate_indices = np.where(candidate_mask)[0]
    if len(candidate_indices) == 0:
        return las

    # Extract coordinates
    coords = np.vstack((las.x, las.y, las.z)).transpose()
    candidate_coords = coords[candidate_indices]

    # Use Open3D to rapidly compute covariances for the candidates
    print("Computing covariance matrices via Open3D...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(candidate_coords)
    
    # Estimate covariances using a KDTree search natively in Open3D
    pcd.estimate_covariances(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=neighbor_k)
    )
    
    # Extract the covariance matrices (N x 3 x 3 array)
    covariances = np.asarray(pcd.covariances)
    
    # Vectorized eigenvalue computation
    print("Calculating planarity...")
    # np.linalg.eigvalsh is optimized for symmetric matrices like covariances
    eigenvalues = np.linalg.eigvalsh(covariances) 
    
    # eigvalsh returns ascending order: e3, e2, e1
    e3 = eigenvalues[:, 0]
    e2 = eigenvalues[:, 1]
    e1 = eigenvalues[:, 2]
    
    # Avoid division by zero
    valid_mask = e1 > 1e-6
    
    # Calculate planarity for all points simultaneously
    planarity = np.zeros_like(e1)
    planarity[valid_mask] = (e2[valid_mask] - e3[valid_mask]) / e1[valid_mask]
    
    # Find points exceeding the threshold
    building_local_indices = np.where(planarity > planarity_threshold)[0]
    new_building_indices = candidate_indices[building_local_indices]
    
    print(f"Reclassifying {len(new_building_indices)} points to Buildings (Class 6).")
    if len(new_building_indices) > 0:
        las.classification[new_building_indices] = 6
        
    return las


def recover_roof_edges(input_path, output_path, search_radius=1.5, z_tolerance=0.5, loose_threshold=0.35):
    """
    Expands the building classification to include roof edges that were missed
    due to lower planarity scores.
    
    Parameters:
    - search_radius: Max distance to look for a 'core' building point.
    - z_tolerance: Max vertical difference allowed (prevents grabbing ground/trees below).
    - loose_threshold: The relaxed planarity score for edges (usually 0.3 to 0.5).
    """
    print(f"Loading {input_path} for Edge Recovery...")
    las = laspy.read(input_path)
    
    # 1. Separate "Core Buildings" and "Candidates"
    # Core = Already classified as 6
    # Candidates = Vegetation (5) or Unclassified (1)
    core_mask = las.classification == 6
    cand_mask = np.isin(las.classification, [1, 5])
    
    core_indices = np.where(core_mask)[0]
    cand_indices = np.where(cand_mask)[0]
    
    if len(core_indices) == 0 or len(cand_indices) == 0:
        print("Not enough points to process.")
        return

    print(f"Refining edges using {len(core_indices)} core building points...")

    # 2. Re-calculate Planarity for Candidates (Crucial Step)
    # We need to know which candidates are 'flat-ish' (edges) vs 'scattered' (trees)
    coords = np.vstack((las.x, las.y, las.z)).transpose()
    cand_coords = coords[cand_indices]
    
    # Build tree on candidates to compute THEIR planarity
    # (Using k=15, slightly smaller context than before to catch sharper edges)
    tree_geom = KDTree(cand_coords, leaf_size=40)
    dist, ind = tree_geom.query(cand_coords, k=15)
    
    # Identify "Weak Candidates" (Planarity between loose_threshold and 0.6)
    weak_candidate_local_indices = []
    
    for i, point_neighbors in enumerate(ind):
        neighbor_xyz = cand_coords[point_neighbors]
        cov = np.cov(neighbor_xyz, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov)
        e3, e2, e1 = eigenvalues
        
        if e1 == 0: continue
            
        planarity = (e2 - e3) / e1
        
        # We accept lower planarity here because we will add a proximity check later
        if planarity > loose_threshold:
            weak_candidate_local_indices.append(i)

    # Convert local list to original LAS indices
    # These are points that LOOK like edges but weren't strict enough to be buildings
    potential_edge_indices = cand_indices[weak_candidate_local_indices]
    potential_edge_coords = coords[potential_edge_indices]

    print(f"Found {len(potential_edge_indices)} potential edge points (Planarity > {loose_threshold}).")
    print("Verifying spatial connection to core buildings...")

    # 3. Spatial Verification (The "Anchor" Check)
    # Build a tree of the CONFIRMED buildings
    core_coords = coords[core_indices]
    core_tree = KDTree(core_coords, leaf_size=40)
    
    # For every potential edge point, find the CLOSEST core building point
    dists, nearest_core_indices = core_tree.query(potential_edge_coords, k=1)
    
    points_to_upgrade = []
    
    for i, (d, core_idx_rel) in enumerate(zip(dists, nearest_core_indices)):
        d = d[0] # Distance to nearest building
        
        # Condition A: Must be horizontally close (within 1.5m)
        if d > search_radius:
            continue
            
        # Condition B: Must be vertically aligned (prevent grabbing the ground below an eave)
        # Get Z of the edge point and Z of the nearest building point
        z_edge = potential_edge_coords[i][2]
        z_core = core_coords[core_idx_rel[0]][2]
        
        if abs(z_edge - z_core) < z_tolerance:
            points_to_upgrade.append(potential_edge_indices[i])
            
    # 4. Apply Changes
    if points_to_upgrade:
        las.classification[points_to_upgrade] = 6
        print(f"Recovered {len(points_to_upgrade)} edge points!")
    else:
        print("No edges recovered.")
        
    las.write(output_path)
    print("Done.")

def filter_building_outliers(input_path, output_path, eps=1.5, min_cluster_size=100):
    """
    Removes small, isolated clusters of points classified as Building (6)
    and reverts them to High Vegetation (5).
    
    Parameters:
    - eps: The maximum distance between two points to be considered neighbors (in meters).
           Increase this if your building points have gaps (e.g., lower density).
    - min_cluster_size: Minimum number of points required to constitute a 'real' building.
    """
    print(f"Loading {input_path}...")
    las = laspy.read(input_path)
    
    # 1. Select only current Building points
    building_mask = las.classification == 6
    building_indices = np.where(building_mask)[0]
    
    if len(building_indices) == 0:
        print("No building points found to filter.")
        return

    print(f"Clustering {len(building_indices)} building points...")
    
    # Get coordinates of building points
    # We use only XY for clustering if we want to treat a building as a single footprint,
    # but using XYZ is safer to avoid merging a low tree with a high roof.
    building_coords = np.vstack((las.x[building_indices], 
                                 las.y[building_indices], 
                                 las.z[building_indices])).transpose()

    # 2. Run DBSCAN Clustering
    # eps=1.5m means points within 1.5m of each other are part of the same object
    # min_samples=10 ensures we don't start a cluster on pure noise
    db = DBSCAN(eps=eps, min_samples=10).fit(building_coords)
    labels = db.labels_

    # 3. Analyze Cluster Sizes
    # labels == -1 are noise (points that didn't even form a small cluster)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Identify which labels are "valid buildings" (large enough)
    valid_labels = set()
    
    # Track stats
    noise_points = 0
    small_cluster_points = 0
    
    for label, count in zip(unique_labels, counts):
        if label == -1:
            noise_points += count
            continue
            
        if count >= min_cluster_size:
            valid_labels.add(label)
        else:
            small_cluster_points += count

    print(f"Filtering Report:")
    print(f" - Total Clusters Found: {len(unique_labels)}")
    print(f" - Noise Points Removed: {noise_points}")
    print(f" - Small Cluster Points Reverted: {small_cluster_points}")
    print(f" - Real Buildings Kept: {len(valid_labels)}")

    # 4. Apply Filtering
    # We iterate through our local building_indices. 
    # If their cluster label is NOT in valid_labels, we revert them.
    
    points_to_revert = []
    
    for i, label in enumerate(labels):
        if label not in valid_labels:
            # Get the original index in the LAS file
            original_idx = building_indices[i]
            points_to_revert.append(original_idx)
            
    # 5. Reclassify
    if points_to_revert:
        # Revert to High Vegetation (5)
        las.classification[points_to_revert] = 5
        
    print(f"Saving cleaned file to {output_path}...")
    las.write(output_path)
    print("Done.")

def filter_building_outliers_optimized(las, eps=1.5, min_cluster_size=100):
    """
    Removes small, isolated building clusters using Open3D's rapid DBSCAN.
    """
    building_mask = las.classification == 6
    building_indices = np.where(building_mask)[0]
    
    if len(building_indices) == 0:
        return las

    building_coords = np.vstack((las.x[building_indices], 
                                 las.y[building_indices], 
                                 las.z[building_indices])).transpose()

    print(f"Clustering {len(building_indices)} building points with Open3D...")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(building_coords)
    
    # Open3D's DBSCAN returns a list of labels
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=10, print_progress=False))
    
    # Analyze cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Find valid clusters (excluding noise which is -1)
    valid_labels = unique_labels[(counts >= min_cluster_size) & (unique_labels != -1)]
    
    # Create a boolean mask of points that belong to valid clusters
    is_valid_cluster = np.isin(labels, valid_labels)
    
    # Points to revert are those that are NOT in a valid cluster
    points_to_revert = building_indices[~is_valid_cluster]
    
    print(f"Reverting {len(points_to_revert)} noise/small cluster points to Vegetation (5).")
    if len(points_to_revert) > 0:
        las.classification[points_to_revert] = 5
        
    return las

# %%
def convert_to_hag(input_path, output_path):
    """
    Normalizes the LAS file Z coordinates to Height Above Ground (HAG).
    """
    print(f"Loading {input_path} for HAG conversion...")
    las = laspy.read(input_path)
    
    # 1. Extract Ground Points (Class 2)
    ground_mask = las.classification == 2
    
    # Fallback: If no class 2, assume the lowest 1% of points are ground
    if np.sum(ground_mask) == 0:
        print("Warning: No Ground Class (2) found. Estimating ground from lowest points...")
        # Sort by Z and take the bottom 1% as a proxy for ground
        sorted_indices = np.argsort(las.z)
        n_ground = int(len(las.points) * 0.01)
        ground_indices = sorted_indices[:n_ground]
        ground_mask = np.zeros(len(las.points), dtype=bool)
        ground_mask[ground_indices] = True
    
    ground_points = np.vstack((las.x[ground_mask], 
                               las.y[ground_mask], 
                               las.z[ground_mask])).transpose()
    
    print(f"Using {len(ground_points)} ground points to build DTM...")

    # 2. Build 2D KDTree for Ground (X, Y only)
    # This allows us to find the nearest ground point for every other point
    ground_xy = ground_points[:, :2]
    ground_z = ground_points[:, 2]
    
    tree = cKDTree(ground_xy)
    
    # 3. Query Nearest Ground Point for ALL points
    # We query the X,Y of all points against the ground tree
    all_xy = np.vstack((las.x, las.y)).transpose()
    
    # k=1 means find the single nearest ground point (fastest method)
    # For smoother terrain, you could use k=3 and average them, but k=1 is sufficient for 100m.
    dists, indices = tree.query(all_xy, k=1)
    
    # 4. Calculate HAG
    # Z_ground_ref is the Z value of the nearest ground neighbor
    z_ground_ref = ground_z[indices]
    
    hag_values = las.z - z_ground_ref
    
    # 5. Update Z values
    # We overwrite the Z dimension. Now Z=0 means "On the ground".
    las.z = hag_values
    
    print(f"HAG Calculated. Min Z: {np.min(las.z):.2f}, Max Z: {np.max(las.z):.2f}")
    
    # Optional: Reset ground points exactly to 0 to remove noise
    las.z[ground_mask] = 0.0
    
    print(f"Saving HAG file to {output_path}...")
    las.write(output_path)



# %%
def visualize_classification(las_file_path):
    print(f"Loading {las_file_path} for visualization...")
    las = laspy.read(las_file_path)
    
    # 1. Extract coordinates
    # We stack them into an (N, 3) array that Open3D expects
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # 2. Extract classification
    classification = las.classification
    
    # 3. Create a Color Map
    # Initialize all points to Gray (default for unclassified/other)
    # Shape must be (N, 3) for RGB channels, values 0.0 to 1.0
    colors = np.zeros((len(points), 3))
    colors[:] = [0.5, 0.5, 0.5]  # Grey
    
    # Color Class 5 (High Vegetation) -> Green
    veg_indices = np.where(classification == 5)[0]
    colors[veg_indices] = [0.0, 0.6, 0.0]  # Dark Green
    
    # Color Class 6 (Building) -> Red
    bldg_indices = np.where(classification == 6)[0]
    colors[bldg_indices] = [1.0, 0.0, 0.0]  # Bright Red
    
    print(f"Stats:")
    print(f" - Vegetation (Green): {len(veg_indices)} points")
    print(f" - Buildings (Red):   {len(bldg_indices)} points")
    
    # 4. Create Open3D PointCloud Object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 5. Visualize
    print("Opening visualizer... (Use mouse to rotate, scroll to zoom)")
    
    # We create a visualization window with a black background for better contrast
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Building vs Vegetation Check")
    vis.add_geometry(pcd)
    
    # Set background to black
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    
    vis.run()
    vis.destroy_window()



if __name__ == "__main__":
    # --- File Paths ---
    input_laz_merged = "data/study_area.laz"
    cropped_laz_path = "data/P5123C2_9_cropped.laz"
    final_output_laz = "output/cleaned_and_recovered.laz"

    target_x, target_y = 532885, 6983510
    custom_bounds = {
        'min_x': target_x - 150,
        'max_x': target_x + 350,
        'min_y': target_y - 350,
        'max_y': target_y + 150
    }

    # 1. Crop the initial file (this still writes the cropped file to disk as a base)
    crop_las_file(input_laz_merged, cropped_laz_path, custom_bounds)

    # 2. Load the cropped file into memory ONCE
    print(f"Loading {cropped_laz_path} into memory for processing...")
    las_memory = laspy.read(cropped_laz_path)

    # 3. Pass the in-memory object through the pipeline
    # Note: You will need to make sure filter_lidar_data and recover_roof_edges 
    # are also updated to accept and return the 'las' object instead of file paths.

    # Example of the flow:
    # las_memory = filter_lidar_data_optimized(las_memory, nb_neighbors=20, std_ratio=3.0)
    las_memory = reclassify_buildings_from_veg_optimized(las_memory, planarity_threshold=0.5)
    las_memory = filter_building_outliers_optimized(las_memory, eps=1.5, min_cluster_size=120)
    # las_memory = recover_roof_edges_optimized(las_memory, ...)

    # 4. Write the final processed data to disk ONCE
    print(f"Saving fully processed file to {final_output_laz}...")
    las_memory.write(final_output_laz)

    # 5. Visualize
    visualize_classification(final_output_laz)

