from unittest.mock import patch

# Geodesic suface patches

more coming soon. 


## Usage

```python
from surfacepatcher.surfacepatcher import GeodesicPatcher

patcher=GeodesicPatcher()
patches=patcher(pdb_file="pdb_file", chain_id="chain_id", radius_angstrom=25, M=25, K=5, cleanup=True)

patcher.save_patches(patches, "path to pickle")
```

```python
from surfacepatcher.patchcomparison import PatchComparison

comparisons=PatchComparison(patches1, patches2)
comparisons.save_distances("patch_comparisons")
```

## Structure

`GeodesicPatcher` returns a ProteinPatcher dataclass that looks like this

```python
class ProteinPatches:
    pdb_file: str
    radius_angstrom: int
    sampling_angle: float
    num_points: int
    patches: dict
    descriptors: torch.Tensor
```

The patches look like this:

```python
patches={
                'center': vertices[center_idx],
                'indices': patch_indices,
                'features': patch_features,
                'area': np.sum(patch_mask),  # Normalize by patch area later
                'atom_ids': self.atom_ids[patch_indices],  # Add this
                'residues': self._get_patch_residues(self.atom_ids[patch_indices]),
            }
```

the `descriptors` is a [M, M, K, 6] `torch.tensor` the two Ms refer to rotations because we compare everything to 
everything else. 

## How it's done

coming soon


