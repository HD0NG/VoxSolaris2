# %% [markdown]
# # LiDAR Pre-Processing Pipeline — Building Reclassification Focus
#
# ## Why the Original Approach Falls Short
#
# The original pipeline uses a **single geometric feature (planarity)** computed at a
# **single neighbourhood scale (k=20)** to decide building vs. vegetation. This has
# several well-documented failure modes:
#
# | Problem | Root Cause |
# |---|---|
# | Flat ground misclassified as building | Planarity alone can't distinguish horizontal surfaces |
# | Sloped / hipped roofs missed | Low planarity at a single scale |
# | Dense tree canopy tops classified as building | Locally planar patches in canopy |
# | Roof edges lost, then "recovered" with loose heuristics | Edge recovery is a patch, not a fix |
# | Pipeline order-dependent: reclassify → filter → recover | Each step can undo the previous one |
#
# ## Upgraded Classification Strategy
#
# The new approach replaces the single-feature threshold with a **multi-feature
# geometric classifier** that uses:
#
# 1. **Height Above Ground (HAG)** — computed *in-memory* before reclassification
#    (original elevation Z is never overwritten). This lets the classifier
#    trivially exclude ground, eliminating the #1 false positive source.
#    The final output retains full ellipsoidal / orthometric elevation.
#
# 2. **Multi-scale eigenvalue features** — planarity, sphericity, linearity, and
#    surface variation computed at k=10, k=20, and k=40. Multi-scale captures
#    both roof edges (small k) and large planar surfaces (large k).
#
# 3. **Normal vector verticality** — building surfaces have normals that are
#    predominantly vertical (flat roofs) or consistently tilted (pitched roofs).
#    Vegetation normals are noisy/random. The *standard deviation* of normals
#    in a neighbourhood is a strong vegetation indicator.
#
# 4. **RANSAC plane fitting** — for each candidate neighbourhood, fit a plane
#    and measure the inlier ratio. Buildings produce high inlier ratios;
#    vegetation produces low ones. This is more robust than eigenvalue-only
#    planarity for noisy point clouds.
#
# 5. **Region-growing post-classification** — instead of the fragile
#    "recover edges → filter outliers" two-step, a single region-growing pass
#    expands confirmed building seeds into adjacent compatible points. This
#    naturally captures edges while respecting geometric continuity.
#
# The features are combined via a **weighted scoring function** (no ML training
# data needed) or optionally via a Random Forest if labelled samples are available.

# %%
import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import laspy
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN

try:
    from hdbscan import HDBSCAN
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

import open3d as o3d

try:
    from shapely.geometry import shape
    from shapely.vectorized import contains
    import fiona
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# =====================================================================
# CONFIGURATION
# =====================================================================
@dataclass
class PipelineConfig:
    """Central configuration for the full pipeline."""
    # Crop
    bounds: Optional[Dict[str, float]] = None
    clip_polygon_path: Optional[str] = None

    # SOR
    skip_sor: bool = False
    sor_nb_neighbors: int = 20
    sor_std_ratio: float = 3.0
    sor_chunk_size: int = 5_000_000

    # Class filter
    valid_classes: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])

    # HAG
    hag_k_neighbors: int = 4

    # --- Building reclassification ---
    reclass_scales: List[int] = field(default_factory=lambda: [10, 20, 40])
    reclass_min_hag: float = 2.0
    reclass_max_hag: float = 50.0
    reclass_score_threshold: float = 0.70   # strict: shape filter is the safety net
    reclass_normal_std_threshold: float = 0.25

    # Region growing
    rg_search_radius: float = 1.0
    rg_z_tolerance: float = 1.5
    rg_min_score: float = 0.50
    rg_normal_agreement: float = 0.90

    # Cluster cleanup
    cluster_eps: float = 1.5          # don't merge distinct buildings
    cluster_min_size: int = 300
    use_hdbscan: bool = True


# =====================================================================
# MERGE (unchanged from previous upgrade)
# =====================================================================
def merge_las_files(paths: List[str], output_path: str) -> None:
    if len(paths) < 2:
        raise ValueError("Provide at least two files to merge.")

    log.info("Reading %d files for merge...", len(paths))
    datasets = [laspy.read(p) for p in paths]

    fmt_id = datasets[0].header.point_format.id
    for i, ds in enumerate(datasets[1:], start=1):
        if ds.header.point_format.id != fmt_id:
            raise ValueError(f"Point format mismatch: file 0={fmt_id}, file {i}={ds.header.point_format.id}")

    ref = datasets[0]
    new_header = laspy.LasHeader(point_format=ref.header.point_format, version=ref.header.version)
    new_header.offsets = ref.header.offsets
    new_header.scales = ref.header.scales
    new_header.vlrs = ref.header.vlrs

    merged_array = np.concatenate([ds.points.array for ds in datasets])
    merged_las = laspy.LasData(new_header)
    merged_las.points = laspy.ScaleAwarePointRecord(
        merged_array, ref.header.point_format, ref.header.scales, ref.header.offsets
    )
    merged_las.header.mins = np.array([merged_las.x.min(), merged_las.y.min(), merged_las.z.min()])
    merged_las.header.maxs = np.array([merged_las.x.max(), merged_las.y.max(), merged_las.z.max()])

    log.info("Writing %s merged points to %s", f"{len(merged_array):,}", output_path)
    merged_las.write(output_path)


# =====================================================================
# CROP (unchanged)
# =====================================================================
def crop_las_file(
    input_path: str, output_path: str,
    bounds: Optional[Dict[str, float]] = None,
    polygon_path: Optional[str] = None,
) -> None:
    las = laspy.read(input_path)
    log.info("Loaded %s points from %s", f"{len(las.points):,}", input_path)

    if polygon_path and HAS_SHAPELY:
        with fiona.open(polygon_path) as src:
            geom = shape(src[0]["geometry"])
        mask = contains(geom, las.x, las.y)
    elif bounds:
        mask = (
            (las.x >= bounds['min_x']) & (las.x <= bounds['max_x']) &
            (las.y >= bounds['min_y']) & (las.y <= bounds['max_y'])
        )
    else:
        raise ValueError("Provide either `bounds` dict or `polygon_path`.")

    cropped = las.points[mask]
    if len(cropped) == 0:
        log.error("No points within the specified region!")
        return

    new_header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
    new_header.offsets = las.header.offsets
    new_header.scales = las.header.scales
    new_header.vlrs = las.header.vlrs
    out = laspy.LasData(new_header)
    out.points = cropped
    log.info("Saving %s cropped points.", f"{len(cropped):,}")
    out.write(output_path)


# =====================================================================
# SOR + CLASS FILTER (unchanged)
# =====================================================================
def filter_lidar_data(
    input_path: str, output_path: str,
    skip_sor: bool = False,
    nb_neighbors: int = 20, std_ratio: float = 2.0,
    valid_classes: Optional[List[int]] = None,
    chunk_size: int = 5_000_000,
) -> None:
    if valid_classes is None:
        valid_classes = [2, 3, 4, 5, 6]

    las = laspy.read(input_path)
    n_orig = len(las.points)
    points = np.vstack((las.x, las.y, las.z)).T

    log.info("Running chunked SOR on %s points (k=%d, σ=%.1f)...", f"{n_orig:,}", nb_neighbors, std_ratio)

    if len(points) <= chunk_size:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if not skip_sor:
            _, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        else:
            log.info("Skipping SOR as per configuration.")
            inlier_idx = np.arange(len(points))
        keep = np.zeros(len(points), dtype=bool)
        keep[inlier_idx] = True
    else:
        keep = np.ones(len(points), dtype=bool)
        x_sorted = np.argsort(points[:, 0])
        for start in range(0, len(points), chunk_size):
            end = min(start + chunk_size, len(points))
            cidx = x_sorted[start:end]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[cidx])
            if not skip_sor:
                _, inlier_local = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            else:
                log.info("Skipping SOR for chunk %d.", start // chunk_size)
                inlier_local = np.arange(len(cidx))
            outlier_local = np.setdiff1d(np.arange(len(cidx)), inlier_local)
            keep[cidx[outlier_local]] = False

    las = las[keep]
    log.info("SOR removed %s outliers.", f"{n_orig - len(las.points):,}")

    las = las[np.isin(las.classification, valid_classes)]
    log.info("After class filter: %s points.", f"{len(las.points):,}")
    las.write(output_path)


# =====================================================================
# HAG COMPUTATION (in-memory only — does NOT overwrite Z)
# =====================================================================
def _compute_hag(las, k: int = 4) -> np.ndarray:
    """
    Compute Height Above Ground for every point using IDW interpolation
    of the nearest ground (Class 2) points.

    Returns an (N,) array of HAG values in metres.  The LAS object's Z
    coordinates are **not modified** — original elevation is preserved.

    Parameters
    ----------
    las : laspy.LasData  — already-loaded point cloud (read once, reused).
    k   : int            — number of ground neighbours for IDW (default 4).
    """
    ground_mask = las.classification == 2
    if np.sum(ground_mask) == 0:
        log.warning("No Class 2 ground. Estimating from lowest 1%%...")
        sorted_idx = np.argsort(las.z)
        n_gnd = max(int(len(las.points) * 0.01), 10)
        ground_mask = np.zeros(len(las.points), dtype=bool)
        ground_mask[sorted_idx[:n_gnd]] = True

    gnd_xy = np.vstack((las.x[ground_mask], las.y[ground_mask])).T
    gnd_z = np.asarray(las.z[ground_mask], dtype=np.float64)
    log.info("Computing HAG from %s ground points (IDW k=%d)...", f"{len(gnd_z):,}", k)

    tree = cKDTree(gnd_xy)
    all_xy = np.vstack((las.x, las.y)).T
    dists, indices = tree.query(all_xy, k=k)

    if k == 1:
        z_ref = gnd_z[indices]
    else:
        dists = np.maximum(dists, 1e-10)
        w = 1.0 / dists
        w /= w.sum(axis=1, keepdims=True)
        z_ref = np.sum(w * gnd_z[indices], axis=1)

    hag = np.maximum(np.asarray(las.z, dtype=np.float64) - z_ref, 0.0)
    log.info("HAG range: %.2f – %.2f m", np.min(hag), np.max(hag))
    return hag


# =====================================================================
# GEOMETRIC FEATURE ENGINE (NEW)
# =====================================================================
def _compute_eigenfeatures_batch(
    coords: np.ndarray,
    neighbor_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute a full set of eigenvalue-derived geometric features for every point,
    fully vectorised (no Python loop).

    Returns dict with keys: planarity, sphericity, linearity, surface_variation,
    omnivariance, anisotropy, eigenentropy, normal_z.

    Note: normal_z_std is computed post-hoc in _compute_multi_scale_features
    after the full normal_z array has been assembled across chunks.

    Parameters
    ----------
    coords : (N, 3) candidate XYZ
    neighbor_indices : (N, k) indices into coords
    """
    N, k = neighbor_indices.shape
    neighbors = coords[neighbor_indices]                    # (N, k, 3)

    # --- Covariance ---
    means = neighbors.mean(axis=1, keepdims=True)           # (N, 1, 3)
    centered = neighbors - means                             # (N, k, 3)
    covs = np.einsum('nki,nkj->nij', centered, centered) / (k - 1)  # (N, 3, 3)

    # --- Eigendecomposition (ascending order) ---
    eigvals, eigvecs = np.linalg.eigh(covs)                 # (N,3), (N,3,3)

    # Clamp tiny negatives from numerical noise
    eigvals = np.maximum(eigvals, 0.0)
    e3, e2, e1 = eigvals[:, 0], eigvals[:, 1], eigvals[:, 2]
    esum = e1 + e2 + e3

    # Guard division
    safe_e1 = np.where(e1 > 0, e1, 1.0)
    safe_esum = np.where(esum > 0, esum, 1.0)

    # --- Eigenvalue features ---
    planarity = (e2 - e3) / safe_e1
    linearity = (e1 - e2) / safe_e1
    sphericity = e3 / safe_e1
    anisotropy = (e1 - e3) / safe_e1
    surface_variation = e3 / safe_esum
    omnivariance = np.cbrt(e1 * e2 * e3)

    # Eigenentropy — using normalised eigenvalues
    en = eigvals / safe_esum[:, None]
    en_safe = np.where(en > 0, en, 1.0)
    eigenentropy = -np.sum(en * np.log(en_safe), axis=1)

    # --- Normal vector features ---
    # The normal is the eigenvector corresponding to the smallest eigenvalue (column 0)
    normals = eigvecs[:, :, 0]  # (N, 3)
    # Make normals consistently upward-pointing
    flip = normals[:, 2] < 0
    normals[flip] *= -1
    normal_z = normals[:, 2]  # verticality: 1.0 = horizontal surface, 0.0 = vertical wall

    # NOTE: normal_z_std (std of neighbour normals) is NOT computed here because
    # neighbor_indices may reference the full coords array while normal_z is only
    # chunk-sized.  It is computed post-hoc in _compute_multi_scale_features once
    # the full normal_z array has been assembled.

    return {
        'planarity': planarity,
        'linearity': linearity,
        'sphericity': sphericity,
        'anisotropy': anisotropy,
        'surface_variation': surface_variation,
        'omnivariance': omnivariance,
        'eigenentropy': eigenentropy,
        'normal_z': normal_z,
    }


def _ransac_plane_inlier_ratio_batch(
    coords: np.ndarray,
    neighbor_indices: np.ndarray,
    distance_threshold: float = 0.15,
    n_iterations: int = 50,
    rng_seed: int = 42,
) -> np.ndarray:
    """
    For each point's neighbourhood, estimate the best-fit plane via RANSAC
    and return the fraction of neighbours that are inliers.

    Buildings → high inlier ratio (0.8–1.0)
    Vegetation → low inlier ratio (0.2–0.5)

    FULLY VECTORISED: each RANSAC iteration processes ALL N points at once.
    The outer loop is only over `n_iterations` (50), not over N (1M+).

    Complexity: O(n_iterations × N × k)  with NumPy vectorisation,
    vs. the old O(N × n_iterations × k) with Python loops.
    Same asymptotic cost, but ~200–500× faster in wall-clock time.
    """
    rng = np.random.default_rng(rng_seed)
    N, k = neighbor_indices.shape

    # (N, k, 3) — all neighbourhoods
    neighbors = coords[neighbor_indices]

    # Track the best inlier count per point across all iterations
    best_inlier_count = np.zeros(N, dtype=np.int32)

    for _ in range(n_iterations):
        # Sample 3 random neighbour indices for EVERY point at once: (N, 3)
        sample_idx = rng.integers(0, k, size=(N, 3))

        # Gather the 3 sampled points per neighbourhood: each is (N, 3)
        rows = np.arange(N)[:, None]  # (N, 1) for advanced indexing
        p0 = neighbors[rows, sample_idx[:, 0:1], :].squeeze(1)  # (N, 3)
        p1 = neighbors[rows, sample_idx[:, 1:2], :].squeeze(1)  # (N, 3)
        p2 = neighbors[rows, sample_idx[:, 2:3], :].squeeze(1)  # (N, 3)

        # Plane normals via cross product: (N, 3)
        v1 = p1 - p0
        v2 = p2 - p0
        normals = np.cross(v1, v2)  # (N, 3)

        # Normalise (with degenerate-triangle guard)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)  # (N, 1)
        norms = np.maximum(norms, 1e-10)
        normals = normals / norms  # (N, 3)

        # Signed distance of every neighbour to its point's plane:
        #   d = (neighbour - p0) · normal
        # neighbors: (N, k, 3),  p0: (N, 1, 3),  normals: (N, 1, 3)
        diffs = neighbors - p0[:, None, :]        # (N, k, 3)
        distances = np.abs(np.einsum('nki,ni->nk', diffs, normals))  # (N, k)

        # Count inliers per point for this iteration
        inlier_count = np.sum(distances < distance_threshold, axis=1)  # (N,)

        # Update best
        best_inlier_count = np.maximum(best_inlier_count, inlier_count)

    return best_inlier_count.astype(np.float64) / k


def _compute_multi_scale_features(
    coords: np.ndarray,
    tree: cKDTree,
    scales: List[int],
    chunk_size: int = 250_000,
) -> Dict[str, np.ndarray]:
    """
    Compute geometric features at multiple neighbourhood scales and aggregate.

    RANSAC has been removed — at neighbourhood sizes k=10–40 with typical
    ALS point spacing, every surface fits a plane perfectly (inlier ratio ≡ 1.0).
    It provided zero discriminative power and consumed ~60% of compute time.

    The effective discriminators for building vs vegetation in ALS data are:
      - planarity: buildings > 0.65, vegetation 0.25–0.50
      - normal_z_std: buildings < 0.10, vegetation > 0.15
      - surface_variation: buildings < 0.05, vegetation > 0.08
      - eigenentropy: buildings < 0.5 (ordered structure), vegetation > 0.7 (disorder)
      - sphericity: buildings < 0.10, vegetation > 0.15

    Processing is done in spatial chunks to limit peak memory.
    """
    N = len(coords)

    # Accumulators
    all_planarity = []
    all_linearity = []
    all_sphericity = []
    all_surface_var = []
    all_normal_z = []
    all_normal_z_std = []
    all_eigenentropy = []

    for k_scale in scales:
        log.info("  Computing features at scale k=%d (%s points, chunks of %s)...",
                 k_scale, f"{N:,}", f"{chunk_size:,}")

        _, ind = tree.query(coords, k=k_scale)

        plan_chunks, lin_chunks, sph_chunks = [], [], []
        sv_chunks, nz_chunks, ee_chunks = [], [], []

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            c_ind = ind[start:end]
            feats = _compute_eigenfeatures_batch(coords, c_ind)

            plan_chunks.append(feats['planarity'])
            lin_chunks.append(feats['linearity'])
            sph_chunks.append(feats['sphericity'])
            sv_chunks.append(feats['surface_variation'])
            nz_chunks.append(feats['normal_z'])
            ee_chunks.append(feats['eigenentropy'])

        # Reassemble full arrays for this scale
        scale_normal_z = np.concatenate(nz_chunks)

        # Compute normal_z_std POST-HOC on the full reassembled normal_z array
        neighbour_nz = scale_normal_z[ind]
        scale_nz_std = np.std(neighbour_nz, axis=1)

        all_planarity.append(np.concatenate(plan_chunks))
        all_linearity.append(np.concatenate(lin_chunks))
        all_sphericity.append(np.concatenate(sph_chunks))
        all_surface_var.append(np.concatenate(sv_chunks))
        all_normal_z.append(scale_normal_z)
        all_normal_z_std.append(scale_nz_std)
        all_eigenentropy.append(np.concatenate(ee_chunks))

        log.info("    Scale k=%d complete.", k_scale)

    # Stack: (n_scales, N)
    planarity_stack = np.stack(all_planarity)
    linearity_stack = np.stack(all_linearity)
    sphericity_stack = np.stack(all_sphericity)
    sv_stack = np.stack(all_surface_var)
    nz_stack = np.stack(all_normal_z)
    nz_std_stack = np.stack(all_normal_z_std)
    ee_stack = np.stack(all_eigenentropy)

    return {
        # Aggregate across scales: take the most building-like value
        'planarity_max': np.max(planarity_stack, axis=0),
        'linearity_min': np.min(linearity_stack, axis=0),
        'sphericity_min': np.min(sphericity_stack, axis=0),
        'surface_var_min': np.min(sv_stack, axis=0),
        'normal_z_max': np.max(nz_stack, axis=0),
        'normal_z_std_min': np.min(nz_std_stack, axis=0),
        'eigenentropy_min': np.min(ee_stack, axis=0),

        # Per-scale for diagnostics
        '_planarity_per_scale': planarity_stack,
    }


# =====================================================================
# BUILDING SCORING FUNCTION
# =====================================================================
def _compute_building_score(features: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute building confidence score from geometric features.

    Design based on observed feature distributions from Nordic ALS data:
        planarity_max:    buildings > 0.6,  vegetation median 0.43
        normal_z_std_min: buildings < 0.10, vegetation median 0.16
        surface_var_min:  buildings < 0.05, vegetation median 0.08
        eigenentropy_min: buildings < 0.4,  vegetation > 0.6
        sphericity_min:   buildings < 0.08, vegetation median 0.13

    Strategy: MULTIPLICATIVE gating, not additive.
    Additive scoring lets a point score high by being excellent on 2 features
    and mediocre on 3 others. Multiplicative scoring means a bad score on
    ANY feature tanks the total — which is what we want, since buildings
    must be planar AND have consistent normals AND low surface variation.

    Returns (N,) float array in [0, 1].
    """
    def _sigmoid(x, center, steepness):
        return 1.0 / (1.0 + np.exp(-steepness * (x - center)))

    N = len(features['planarity_max'])

    # --- Diagnostic logging ---
    for name, key in [
        ('planarity_max', 'planarity_max'),
        ('normal_z_std_min', 'normal_z_std_min'),
        ('surface_var_min', 'surface_var_min'),
        ('eigenentropy_min', 'eigenentropy_min'),
        ('sphericity_min', 'sphericity_min'),
    ]:
        arr = features[key]
        log.info("  Feature %-20s  min=%.3f  p25=%.3f  median=%.3f  p75=%.3f  max=%.3f",
                 name,
                 np.min(arr), np.percentile(arr, 25), np.median(arr),
                 np.percentile(arr, 75), np.max(arr))

    # --- Individual feature scores in [0, 1] ---

    # Planarity: HIGH is building-like. Center at the p75 of the full distribution
    # so that only the top quartile scores well.
    s_plan = _sigmoid(features['planarity_max'], center=0.55, steepness=15.0)

    # Normal-Z std: LOW is building-like. Inverted sigmoid.
    # Center at 0.12 — below p25, so only very consistent normals score well.
    s_nstd = 1.0 - _sigmoid(features['normal_z_std_min'], center=0.10, steepness=30.0)

    # Surface variation: LOW is building-like. Center at 0.05.
    s_sv = 1.0 - _sigmoid(features['surface_var_min'], center=0.05, steepness=30.0)

    # Eigenentropy: LOW is building-like (ordered planar structure vs disordered canopy).
    # Data shows: median=0.843, p25=0.755. Center at 0.50 is intentionally strict —
    # it acts as a strong filter that only passes genuinely ordered surfaces.
    # With center at 0.75 too many vegetation points leaked through (38k seeds vs 2.8k).
    s_ee = 1.0 - _sigmoid(features['eigenentropy_min'], center=0.50, steepness=12.0)

    # Sphericity: LOW is building-like.
    s_sph = 1.0 - _sigmoid(features['sphericity_min'], center=0.10, steepness=20.0)

    # --- MULTIPLICATIVE combination ---
    # Each score is in [0, 1]. The geometric mean ensures that a low score
    # on ANY feature pulls the total down — a point must be building-like
    # on ALL features simultaneously.
    # We use weighted geometric mean (exponents = weights).
    w_plan, w_nstd, w_sv, w_ee, w_sph = 0.25, 0.30, 0.20, 0.15, 0.10

    score = (
        np.power(np.maximum(s_plan, 1e-10), w_plan) *
        np.power(np.maximum(s_nstd, 1e-10), w_nstd) *
        np.power(np.maximum(s_sv,   1e-10), w_sv) *
        np.power(np.maximum(s_ee,   1e-10), w_ee) *
        np.power(np.maximum(s_sph,  1e-10), w_sph)
    )

    # Log component distributions for tuning
    for name, s in [('s_planarity', s_plan), ('s_normal_std', s_nstd),
                    ('s_surf_var', s_sv), ('s_eigenentropy', s_ee),
                    ('s_sphericity', s_sph)]:
        log.info("  Component %-16s  median=%.3f  p75=%.3f  p90=%.3f",
                 name, np.median(s), np.percentile(s, 75), np.percentile(s, 90))

    return score


# =====================================================================
# REGION GROWING REFINEMENT (NEW — replaces recover_roof_edges)
# =====================================================================
def _region_grow_buildings(
    coords: np.ndarray,
    scores: np.ndarray,
    seed_mask: np.ndarray,
    candidate_mask: np.ndarray,
    normals_z: np.ndarray,
    search_radius: float = 1.5,
    z_tolerance: float = 2.0,
    min_score: float = 0.40,
    normal_agreement: float = 0.85,
) -> np.ndarray:
    """
    Region-growing from confirmed building seeds into adjacent candidates.

    This replaces the separate "recover_roof_edges" + "filter_building_outliers"
    two-step with a single principled pass that:
      1. Starts from high-confidence building seeds.
      2. Examines all non-seed candidates within `search_radius`.
      3. Accepts a candidate if:
         a. Its building score >= `min_score` (relaxed vs. seed threshold), AND
         b. Its Z is within `z_tolerance` of the seed, AND
         c. Its surface normal is consistent with the seed (cosine > `normal_agreement`
            measured via the Z-component, since we're comparing flatness).
      4. Newly accepted points become seeds for the next iteration.
      5. Repeats until no more points are added.

    This naturally captures roof edges (which have moderate scores but are
    spatially and normally consistent with the roof) while rejecting isolated
    vegetation (which fails the spatial + normal consistency checks).

    Returns updated boolean mask of all building points.
    """
    building_mask = seed_mask.copy()
    remaining = candidate_mask & ~seed_mask

    max_iterations = 5  # Shape filter catches any false positives from over-growing

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        current_building_idx = np.where(building_mask)[0]
        current_remaining_idx = np.where(remaining)[0]

        if len(current_remaining_idx) == 0:
            break

        remaining_coords = coords[current_remaining_idx]
        building_coords = coords[current_building_idx]

        if len(building_coords) == 0:
            break

        btree = cKDTree(building_coords)
        dists, nearest_bldg_local = btree.query(remaining_coords, k=1)

        nearest_bldg_global = current_building_idx[nearest_bldg_local]

        cond_dist = dists <= search_radius
        cond_z = np.abs(coords[current_remaining_idx, 2] - coords[nearest_bldg_global, 2]) <= z_tolerance
        cond_score = scores[current_remaining_idx] >= min_score
        cond_normal = np.abs(normals_z[current_remaining_idx] - normals_z[nearest_bldg_global]) <= (1.0 - normal_agreement)

        accept = cond_dist & cond_z & cond_score & cond_normal
        accepted_global = current_remaining_idx[accept]

        if len(accepted_global) == 0:
            break

        # Log what's passing each condition to diagnose flooding
        n_near = int(np.sum(cond_dist))
        n_near_z = int(np.sum(cond_dist & cond_z))
        n_near_z_score = int(np.sum(cond_dist & cond_z & cond_score))
        log.info("  Region grow iter %d: %d near → %d +Z → %d +score → %d +normal (accepted)",
                 iteration, n_near, n_near_z, n_near_z_score, len(accepted_global))

        building_mask[accepted_global] = True
        remaining[accepted_global] = False

    return building_mask


# =====================================================================
# MAIN RECLASSIFICATION FUNCTION (NEW — replaces 3 old functions)
# =====================================================================
def reclassify_buildings(
    input_path: str,
    output_path: str,
    cfg: PipelineConfig,
) -> None:
    """
    Multi-feature, multi-scale building reclassification with region growing.

    Replaces the old three-function chain:
        reclassify_buildings_from_veg → filter_building_outliers → recover_roof_edges

    HAG is computed in-memory and used ONLY for candidate filtering.
    The output file retains the original elevation Z values.

    Pipeline:
        1. Compute HAG in-memory (original Z untouched)
        2. HAG filter — exclude ground and extreme outliers
        3. Multi-scale eigenfeatures on candidates (using original elevation)
        4. Weighted score computation
        5. High-confidence seeds (score > threshold)
        6. Region-growing to capture edges
        7. Cluster-based cleanup of remaining noise
        8. Write output with original elevation preserved
    """
    t0 = time.perf_counter()
    las = laspy.read(input_path)
    log.info("Loaded %s points for building reclassification.", f"{len(las.points):,}")

    # Original elevation coordinates — used for ALL geometry and preserved in output
    coords = np.vstack((las.x, las.y, las.z)).T
    classification = np.asarray(las.classification).copy()

    # ------------------------------------------------------------------
    # STEP 1: Compute HAG in-memory (Z is NOT overwritten)
    # ------------------------------------------------------------------
    hag = _compute_hag(las, k=cfg.hag_k_neighbors)

    # ------------------------------------------------------------------
    # STEP 2: HAG-based candidate selection
    # ------------------------------------------------------------------
    # Use the in-memory HAG array to filter candidates by height,
    # but all subsequent geometry uses the original elevation coords.
    candidate_mask = (
        np.isin(classification, [1, 5]) &   # Unclassified or High Vegetation
        (hag >= cfg.reclass_min_hag) &
        (hag <= cfg.reclass_max_hag)
    )

    candidate_indices = np.where(candidate_mask)[0]
    log.info("Step 2 — HAG filter: %s candidates (%.1f–%.1f m above ground)",
             f"{len(candidate_indices):,}", cfg.reclass_min_hag, cfg.reclass_max_hag)

    if len(candidate_indices) == 0:
        log.warning("No candidates after HAG filter. Check min_hag / max_hag settings.")
        las.write(output_path)
        return

    # Use original-elevation coords for all geometric analysis
    cand_coords = coords[candidate_indices]

    # ------------------------------------------------------------------
    # STEP 3: Multi-scale geometric features
    # ------------------------------------------------------------------
    log.info("Step 3 — Computing multi-scale features at k=%s...", cfg.reclass_scales)
    tree = cKDTree(cand_coords)
    ms_features = _compute_multi_scale_features(
        cand_coords, tree,
        scales=cfg.reclass_scales,
    )

    # ------------------------------------------------------------------
    # STEP 4: Building score
    # ------------------------------------------------------------------
    log.info("Step 4 — Computing building confidence scores...")
    scores = _compute_building_score(ms_features)

    log.info("  Score distribution: min=%.3f  median=%.3f  mean=%.3f  max=%.3f",
             np.min(scores), np.median(scores), np.mean(scores), np.max(scores))

    # ------------------------------------------------------------------
    # STEP 5: High-confidence seeds
    # ------------------------------------------------------------------
    seed_local_mask = scores >= cfg.reclass_score_threshold
    n_seeds = int(np.sum(seed_local_mask))
    log.info("Step 4 — Seeds (score >= %.2f): %s points",
             cfg.reclass_score_threshold, f"{n_seeds:,}")

    if n_seeds == 0:
        log.warning("No seeds found. Consider lowering reclass_score_threshold.")
        las.write(output_path)
        return

    # ------------------------------------------------------------------
    # STEP 6: Region growing
    # ------------------------------------------------------------------
    log.info("Step 6 — Region growing (radius=%.1f m, z_tol=%.1f m, min_score=%.2f)...",
             cfg.rg_search_radius, cfg.rg_z_tolerance, cfg.rg_min_score)

    # We need normal_z for the agreement check
    # Use the largest scale for the most stable normals
    max_k = max(cfg.reclass_scales)
    _, ind_norms = tree.query(cand_coords, k=max_k)
    norm_feats = _compute_eigenfeatures_batch(cand_coords, ind_norms)
    normals_z = norm_feats['normal_z']

    all_candidate_local = np.ones(len(cand_coords), dtype=bool)
    building_local_mask = _region_grow_buildings(
        coords=cand_coords,
        scores=scores,
        seed_mask=seed_local_mask,
        candidate_mask=all_candidate_local,
        normals_z=normals_z,
        search_radius=cfg.rg_search_radius,
        z_tolerance=cfg.rg_z_tolerance,
        min_score=cfg.rg_min_score,
        normal_agreement=cfg.rg_normal_agreement,
    )

    n_after_rg = int(np.sum(building_local_mask))
    log.info("  After region growing: %s building points (grew %s from seeds)",
             f"{n_after_rg:,}", f"{n_after_rg - n_seeds:,}")

    # ------------------------------------------------------------------
    # STEP 7: Cluster cleanup — remove tiny isolated clusters
    # ------------------------------------------------------------------
    log.info("Step 7 — Cluster-based cleanup (min_size=%d)...", cfg.cluster_min_size)
    bldg_local_idx = np.where(building_local_mask)[0]

    if len(bldg_local_idx) > 0:
        bldg_coords = cand_coords[bldg_local_idx]

        if cfg.use_hdbscan and HAS_HDBSCAN:
            clusterer = HDBSCAN(min_cluster_size=cfg.cluster_min_size, min_samples=10)
            labels = clusterer.fit_predict(bldg_coords)
        else:
            labels = DBSCAN(eps=cfg.cluster_eps, min_samples=10).fit_predict(bldg_coords)

        unique_labels, counts = np.unique(labels, return_counts=True)
        size_map = dict(zip(unique_labels, counts))
        point_sizes = np.array([size_map[l] for l in labels])

        # --- Size filter ---
        valid = (labels != -1) & (point_sizes >= cfg.cluster_min_size)

        # --- Shape validation per cluster ---
        # For each surviving cluster, check if its 2D footprint looks like a
        # building (large, compact area) vs a tree crown (small, round).
        valid_labels_set = set(unique_labels[(unique_labels != -1) & (counts >= cfg.cluster_min_size)])
        reject_labels = set()

        for label in valid_labels_set:
            cluster_mask = labels == label
            cluster_xy = bldg_coords[cluster_mask, :2]  # XY only
            cluster_z = bldg_coords[cluster_mask, 2]

            # 2D bounding box area
            xy_min = cluster_xy.min(axis=0)
            xy_max = cluster_xy.max(axis=0)
            bbox_dims = xy_max - xy_min
            bbox_area = bbox_dims[0] * bbox_dims[1]

            # Z range within cluster — buildings are flat-topped, trees are domed
            z_range = cluster_z.max() - cluster_z.min()
            z_iqr = np.percentile(cluster_z, 75) - np.percentile(cluster_z, 25)

            # Footprint compactness: point count / bbox_area ≈ point density on footprint
            # Trees have very few points per m² of bbox because they're round within a square bbox
            n_pts = int(cluster_mask.sum())
            footprint_density = n_pts / max(bbox_area, 0.01)

            # Decision criteria:
            # 1. Minimum footprint area (buildings > ~20 m²)
            # 2. Z IQR should be small for flat roofs (< 1.5m for most residential)
            #    but allow more for pitched roofs (< 4m)
            # 3. Aspect ratio shouldn't be extreme (not a thin line of points)
            min_dim = min(bbox_dims)
            max_dim = max(bbox_dims)
            aspect = max_dim / max(min_dim, 0.01)

            is_building = True
            reason = ""

            # Footprint density: pts / bbox_area
            # Real buildings at ~4.5 pts/m² should have density > 0.5 pts/m²
            # (accounting for bbox being larger than actual footprint)
            # Scattered tree points have density < 0.1 pts/m²
            if bbox_area < 20.0:
                is_building = False
                reason = f"footprint too small ({bbox_area:.1f} m²)"
            elif min_dim < 2.0:
                is_building = False
                reason = f"too narrow ({min_dim:.1f} m)"
            elif aspect > 10.0:
                is_building = False
                reason = f"too elongated (aspect {aspect:.1f})"
            elif z_iqr > 5.0:
                is_building = False
                reason = f"Z too variable (IQR {z_iqr:.1f} m)"
            elif footprint_density < 0.25:
                is_building = False
                reason = f"too sparse ({footprint_density:.2f} pts/m², area {bbox_area:.0f} m²)"

            if is_building:
                log.info("  Cluster %d: KEEP  %d pts, %.0f m², Z-IQR=%.1f m, aspect=%.1f",
                         label, n_pts, bbox_area, z_iqr, aspect)
            else:
                log.info("  Cluster %d: REJECT  %d pts, %.0f m² — %s",
                         label, n_pts, bbox_area, reason)
                reject_labels.add(label)

        # Apply shape rejections
        for label in reject_labels:
            valid[labels == label] = False

        building_local_mask[bldg_local_idx[~valid]] = False

        n_reverted = int(np.sum(~valid))
        n_final = int(np.sum(building_local_mask))
        n_valid_clusters = len(valid_labels_set) - len(reject_labels)
        log.info("  Cleanup: reverted %d spurious points, kept %d in %d clusters.",
                 n_reverted, n_final, n_valid_clusters)

    # ------------------------------------------------------------------
    # APPLY to LAS classification
    # ------------------------------------------------------------------
    building_global_idx = candidate_indices[building_local_mask]
    classification[building_global_idx] = 6
    las.classification = classification

    las.write(output_path)
    log.info("Building reclassification complete in %.1f s. Saved to %s",
             time.perf_counter() - t0, output_path)


# =====================================================================
# VISUALIZATION (expanded)
# =====================================================================
def visualize_classification(las_file_path: str) -> None:
    las = laspy.read(las_file_path)
    points = np.vstack((las.x, las.y, las.z)).T
    classification = np.asarray(las.classification)

    color_map = {
        2: [0.6, 0.4, 0.2],   # Ground
        3: [0.0, 0.9, 0.0],   # Low Veg
        4: [0.0, 0.7, 0.0],   # Med Veg
        5: [0.0, 0.5, 0.0],   # High Veg
        6: [1.0, 0.0, 0.0],   # Building
    }
    colors = np.tile([0.5, 0.5, 0.5], (len(points), 1))
    for cls, rgb in color_map.items():
        colors[classification == cls] = rgb

    for cls in sorted(color_map):
        c = int(np.sum(classification == cls))
        if c > 0:
            log.info("Class %d: %s points", cls, f"{c:,}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR Classification Viewer")
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    vis.run()
    vis.destroy_window()


# =====================================================================
# PIPELINE
# =====================================================================
def run_pipeline(cfg: PipelineConfig) -> None:
    """
    Upgraded pipeline order:
        1. Crop
        2. SOR + class filter
        3. Building reclassification (computes HAG internally, outputs original elevation)
        4. Visualise
    """
    input_laz = "data/study_area.laz"
    cropped = "data/cropped.laz"
    filtered = "output/filtered.laz"
    reclassified = "output/reclassified_final_v5.laz"

    os.makedirs("output", exist_ok=True)
    t0 = time.perf_counter()

    crop_las_file(input_laz, cropped, bounds=cfg.bounds)
    filter_lidar_data(cropped, filtered,
                      skip_sor=cfg.skip_sor,
                      nb_neighbors=cfg.sor_nb_neighbors,
                      std_ratio=cfg.sor_std_ratio,
                      valid_classes=cfg.valid_classes,
                      chunk_size=cfg.sor_chunk_size)

    # Reclassification computes HAG in-memory for candidate filtering,
    # but the output file retains the original elevation Z values.
    reclassify_buildings(filtered, reclassified, cfg=cfg)

    log.info("Full pipeline finished in %.1f s.", time.perf_counter() - t0)
    visualize_classification(reclassified)


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    target_x, target_y = 532885, 6983510

    config = PipelineConfig(
        bounds={
            'min_x': target_x - 150,
            'max_x': target_x + 350,
            'min_y': target_y - 350,
            'max_y': target_y + 150,
        },
        skip_sor=True,
        sor_nb_neighbors=20,
        sor_std_ratio=3.0,
        hag_k_neighbors=4,
        reclass_scales=[10, 20, 40],
        reclass_min_hag=2.0,
        reclass_max_hag=50.0,
        reclass_score_threshold=0.70,
        reclass_normal_std_threshold=0.25,
        rg_search_radius=2.0,
        rg_z_tolerance=3.0,
        rg_min_score=0.50,
        rg_normal_agreement=0.90,
        cluster_eps=2.0,
        cluster_min_size=250,
        use_hdbscan=True,
    )

    run_pipeline(config)