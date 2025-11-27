# SurfacePatcher

Protein surface comparison toolkit with multiple complementary methods for finding similar surface patches.

## Features

- **Geodesic Patch Descriptors**: High-resolution radial descriptors with biochemical feature weighting
- **Point Cloud Descriptors**: FPFH, SHOT, and Zernike 3D moments for geometric comparison
- **Topological Descriptors**: Persistent homology for multi-scale shape analysis
- **Hybrid Comparison**: Weighted fusion of multiple methods with preset configurations

## Installation

### Create conda environment

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate surfacepatcher
```

### Install Package

```bash
# Clone or download the repository
cd surfacepatcher

# Install in development mode
pip install -e .

# Or install directly
pip install .
```

### Install Dependencies

All Python dependencies will be installed automatically. If you need to install manually:

```bash
pip install -r requirements.txt
```

**Note**: Installing `giotto-tda` may require additional system dependencies. See [giotto-tda installation guide](https://giotto-ai.github.io/gtda-docs/latest/installation.html).

## Quick Start

### 1. Extract Surface Patches

```python
from geodesic_patcher import GeodesicPatcher

# Initialize patcher
patcher = GeodesicPatcher()

# Extract patches from protein
patches = patcher(
    pdb_file="protein.pdb",
    chain_id=0,
    radius_angstrom=15,  # Patch radius
    M=36,                # Angular sampling (10° intervals)
    K=6                  # Radial sampling points
)
```

### 2. Compare Using Different Methods

#### Method 1: Geodesic Patch Comparison (Feature-Weighted)

```python
from patch_comparison import PatchComparison

comparator = PatchComparison(
    patches1, 
    patches2,
    use_epitope_weights=True  # Optimized for epitope finding
)

distances, indices, keys1, keys2 = comparator.compute(batch_size=100)
```

#### Method 2: Point Cloud Geometric Descriptors

```python
from pointcloud_comparison import PointCloudComparison

# Option A: FPFH descriptors
comparator = PointCloudComparison(
    patches1, patches2,
    descriptor_type='fpfh'
)

# Option B: Zernike 3D moments (rotation-invariant)
comparator = PointCloudComparison(
    patches1, patches2,
    descriptor_type='zernike',
    zernike_order=10,      # Polynomial order
    zernike_grid_size=32   # Voxel resolution
)

distances, keys1, keys2 = comparator.compute(distance_metric='euclidean')
```

#### Method 3: Topological Descriptors (Persistent Homology)

```python
from topological_comparison import TopologicalComparison

comparator = TopologicalComparison(
    patches1, patches2,
    homology_dimensions=(0, 1, 2),  # H0, H1, H2
    max_edge_length=15.0,
    use_feature_filtration=True     # Include biochemical features
)

distances, keys1, keys2 = comparator.compute()
```

#### Method 4: Hybrid Comparison (Combine All Methods)

```python
from hybrid_comparison import HybridComparison

# Compute with all three methods first
geodesic_results = geodesic_comparator.compute()
pointcloud_results = pointcloud_comparator.compute()
topological_results = topological_comparator.compute()

# Combine with hybrid approach
hybrid = HybridComparison(
    geodesic_results=geodesic_results,
    pointcloud_results=pointcloud_results,
    topological_results=topological_results,
    weights=HybridComparison.get_preset_weights('epitope'),
    fusion_method='weighted_sum'
)

hybrid_distances, keys1, keys2 = hybrid.compute()

# Get top matches
top_matches = hybrid.get_top_matches(n=10, method='hybrid')

# Analyze method agreement
correlations = hybrid.get_method_agreement()
```

## Method Comparison

| Method | Best For | Speed | Key Features |
|--------|----------|-------|--------------|
| **Geodesic** | Detailed local patterns | Medium | High-res radial features, biochemistry |
| **FPFH/SHOT** | Overall shape | Fast | Local geometric histograms |
| **Zernike** | Global shape | Medium | Rotation-invariant moments |
| **Topological** | Cavities & topology | Slow | Persistent homology (H0, H1, H2) |
| **Hybrid** | Comprehensive analysis | Slow | Combines all methods |

## Configuration Options

### Geodesic Patch Comparison

```python
# Feature weight presets
PatchComparison.get_preset_weights('epitope')    # Antibody binding sites
PatchComparison.get_preset_weights('enzyme')     # Active sites
PatchComparison.get_preset_weights('interface')  # Protein-protein interfaces
PatchComparison.get_preset_weights('general')    # Equal weights

# Custom weights (order: shape_index, mean_curvature, electrostatic, 
#                        h_bond_donor, h_bond_acceptor, hydrophobicity)
custom_weights = [2.0, 1.5, 2.0, 1.8, 1.8, 1.2]
comparator = PatchComparison(patches1, patches2, feature_weights=custom_weights)
```

### Point Cloud Descriptors

```python
# Descriptor types
descriptor_type='fpfh'      # Fast Point Feature Histogram
descriptor_type='shot'      # Signature of Histograms of Orientations
descriptor_type='zernike'   # 3D Zernike moments
descriptor_type='all'       # Combine all three

# Zernike parameters
zernike_order=10           # Polynomial order (10-20 typical)
zernike_grid_size=32       # Voxel resolution (16=fast, 32=balanced, 64=detailed)

# Distance metrics
distance_metric='euclidean'  # L2 distance
distance_metric='cosine'     # Cosine similarity
distance_metric='chi2'       # Chi-square (for histograms)
```

### Topological Descriptors

```python
# Homology dimensions
homology_dimensions=(0, 1, 2)  # H0=components, H1=loops, H2=voids

# Filtration parameters
max_edge_length=15.0           # Maximum edge length for Vietoris-Rips
use_feature_filtration=True    # Include electrostatic/hydrophobic filtrations
```

### Hybrid Comparison

```python
# Fusion methods
fusion_method='weighted_sum'   # Weighted average of normalized distances
fusion_method='rank_fusion'    # Borda count on rankings
fusion_method='product'        # Geometric mean

# Weight presets
HybridComparison.get_preset_weights('balanced')         # Equal weights
HybridComparison.get_preset_weights('epitope')          # Optimized for epitopes
HybridComparison.get_preset_weights('shape_focused')    # Emphasize geometry
HybridComparison.get_preset_weights('topology_focused') # Emphasize topology
```

## Recommendations for Epitope Finding

1. **Use Hybrid Approach** with `'epitope'` preset weights
2. **Emphasize**:
   - Geodesic method (detailed biochemical features)
   - Topological method (cavity/protrusion detection)
3. **Descriptor Choice**: Zernike for global shape + FPFH for local geometry
4. **Fusion**: Use `'weighted_sum'` or `'rank_fusion'` for stable results
5. **Validation**: Check method agreement with `hybrid.get_method_agreement()`

## API Reference

### GeodesicPatcher

```python
patcher = GeodesicPatcher()
patches = patcher(
    pdb_file: str,           # Path to PDB file
    chain_id: int,           # Chain to process
    radius_angstrom: int,    # Patch radius in Angstroms
    M: int,                  # Number of angular rays (360/M = angle)
    K: int,                  # Radial sampling points
    cleanup: bool = True     # Remove temporary files
)
```

### PatchComparison

```python
comparator = PatchComparison(
    patches1: ProteinPatches,
    patches2: ProteinPatches,
    feature_weights: Optional[List[float]] = None,
    use_epitope_weights: bool = True
)

distances, indices, keys1, keys2 = comparator.compute(batch_size: int = 100)
```

### PointCloudComparison

```python
comparator = PointCloudComparison(
    patches1: ProteinPatches,
    patches2: ProteinPatches,
    descriptor_type: str = 'fpfh',  # 'fpfh', 'shot', 'zernike', 'all'
    zernike_order: int = 10,
    zernike_grid_size: int = 32
)

distances, keys1, keys2 = comparator.compute(distance_metric: str = 'euclidean')
```

### TopologicalComparison

```python
comparator = TopologicalComparison(
    patches1: ProteinPatches,
    patches2: ProteinPatches,
    homology_dimensions: Tuple[int, ...] = (0, 1, 2),
    max_edge_length: float = 15.0,
    use_feature_filtration: bool = True
)

distances, keys1, keys2 = comparator.compute(distance_metric: str = 'euclidean')
```

### HybridComparison

```python
hybrid = HybridComparison(
    geodesic_results: Optional[Tuple] = None,
    pointcloud_results: Optional[Tuple] = None,
    topological_results: Optional[Tuple] = None,
    weights: Optional[Dict[str, float]] = None,
    fusion_method: str = 'weighted_sum'
)

distances, keys1, keys2 = hybrid.compute()
top_matches = hybrid.get_top_matches(n: int = 10, method: str = 'hybrid')
correlations = hybrid.get_method_agreement()
```

## Troubleshooting

### MSMS not found
```bash
# Ensure MSMS is in PATH
which msms
# If not found, add to PATH or install
```

### giotto-tda installation issues
```bash
# Install system dependencies first (Ubuntu/Debian)
sudo apt-get install build-essential cmake

# Then install giotto-tda
pip install giotto-tda
```

### Out of memory errors
```bash
# Reduce batch size for geodesic comparison
comparator.compute(batch_size=50)

# Reduce grid size for Zernike
zernike_grid_size=16

# Process fewer patches at once
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{surfacepatcher,
  title = {SurfacePatcher: Multi-Method Protein Surface Comparison},
  author = {Alper Celik},
  year = {2025},
  url = {https://github.com/celalp/surfacepatcher}
}
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For questions or issues, please open an issue on GitHub or contact [alper.celik@sickkids.ca]
