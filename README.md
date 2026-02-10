# VoxSolaris
A Voxel-Based Ray-Casting Framework for Simulating Solar Radiation Attenuation in Complex Environments Using LiDAR Data

## Overview
VoxSolaris is a research-focused, high-performance Python simulation engine designed to accurately model solar radiation and shadow dynamics in complex 3D environments. By converting LiDAR point clouds into a volumetric grid (voxels), it can simulate the path of sunlight through intricate urban and forested landscapes with high physical realism.

The engine employs a cone-casting methodology to account for the sun's angular diameter, producing soft, realistic shadows (penumbra). It supports distinct physical models for different environmental features, treating buildings as opaque objects and modeling vegetation as a porous medium using the Beer-Lambert law with class-specific extinction coefficients.

The final output is a detailed shadow attenuation matrix, a CSV file quantifying the solar transmittance at a specific point for every degree of the sky, making it an ideal input for a wide range of scientific applications.

## Key Features
- Voxel-Based Representation: Accurately models true 3D environments, including vertical facades, overhangs, and complex canopy structures, overcoming the limitations of 2.5D DSM/raster methods.

- Cone-Casting for Realism: Simulates sunlight as a cone of rays, not a single point, to produce physically accurate soft shadows (penumbra).

- Multi-Class Attenuation: Supports distinct attenuation models for different object types:

- Binary Occlusion for opaque objects like buildings.

- Beer-Lambert Law for porous media like vegetation.

- Differentiated Vegetation: Allows for multiple vegetation classes (e.g., low, medium, high) with unique extinction coefficients (k-values) for more nuanced forest canopy modeling.

- High-Resolution Output: Generates a detailed shadow attenuation matrix with 1° resolution in both solar azimuth and altitude.
