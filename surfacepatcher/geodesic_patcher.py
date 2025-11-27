import os
import pickle
import tempfile
from dataclasses import dataclass

import torch
from scipy.interpolate import interp1d

from surfacepatcher.utils import *

@dataclass(frozen=True)
class ProteinPatches:
    pdb_file: str
    radius_angstrom: int
    sampling_angle: float
    num_points: int
    patches: dict
    descriptors: torch.Tensor


class GeodesicPatcher:
    def surface(self, pdb_file, chain_id=None):
        """
        takes a pdb file, runs pdb2pqr then msms
        :return: md.trajectory, vertices, faces, face normals and atom ids
        """
        # 1. Load protein and select chain
        traj = md.load(pdb_file)
        if chain_id:
            traj = traj.atom_slice(traj.topology.select(f"chainid == {self.chain_id}")) #TODO Change to expression

        vertices, faces, normals, atom_ids = compute_msms_surface(traj.xyz[0], traj)
        return traj, vertices, faces, normals, atom_ids

    def vertex_properties(self, traj, vertices, faces):
        """
        caluclate basic features for each vertex these include shape index (convex, concave, flat), curvature (how much of former)
        simple electrostatic charge, donor, acceptor hbond propensity and hydrophobicity (Doolittle)
        :return: properties for each vertex
        """
        properties = {}

        # 1. **SHAPE**: Local curvature (mean + Gaussian)
        curvature, shape_index = compute_curvature(vertices, faces)
        properties['shape_index'] = shape_index  # -1(convex) to +1(concave)
        properties['mean_curvature'] = curvature

        # 2. **ELECTROSTATICS**: Project atomic charges to surface
        properties['electrostatic'] = project_electrostatics(traj, vertices)

        # 3. **H-BOND PROPENSITY**: Donor/acceptor density
        properties['h_bond_donor'] = project_hbond_propensity(traj, vertices, 'donor')
        properties['h_bond_acceptor'] = project_hbond_propensity(traj, vertices, 'acceptor')

        # 4. **HYDROPHOBICITY**: Weighted average from nearby residues
        properties['hydrophobicity'] = project_hydrophobicity(traj, vertices)

        return properties

    def extract_geodesic_patches(self, vertices, faces, radius, properties, traj, atoms):
        """
        extract a circular patch of radicu self.radius, the circle is calculated using geodesic distance
        :return: a dict with all the properties per patch, there can be 1000s of patches per protein and might
        take a while each patch will have the following dict

        {
                'center': self.vertices[center_idx],
                'indices': patch_indices,
                'features': patch_features,
                'area': np.sum(patch_mask),  # Normalize by patch area later
                'atom_ids': self.atom_ids[patch_indices],  # Add this
                'residues': self._get_patch_residues(self.atom_ids[patch_indices]),
            }

        """
        # Build geodesic distance matrix (approximate via mesh graph)
        dist_matrix = compute_geodesic_distances(vertices, faces)

        patches = {}
        for center_idx in range(len(vertices)):
            # Find all vertices within geodesic radius n
            patch_mask = dist_matrix[center_idx] <= radius
            patch_indices = np.where(patch_mask)[0]

            # Store patch features
            patch_features = {}
            for prop_name, prop_values in properties.items():
                patch_features[prop_name] = prop_values[patch_indices]

            patches[center_idx] = {
                'center': vertices[center_idx],
                'indices': patch_indices,
                'features': patch_features,
                'area': np.sum(patch_mask),  # Normalize by patch area later
                'atom_ids': atoms[patch_indices],  # Add this
                'residues': self._get_patch_residues(traj, atoms[patch_indices]),
            }

        return patches, dist_matrix


    def compute_detailed_patch_descriptors(self, patches, vertices, dist_matrix, normals, radius, M=36, K=5) -> torch.Tensor:
        """
        compute radial descriptors, starting form the center of the patch move to the diameter with M interval so 36 means
        every 10 degrees and within that radius sample K points, for this to make sense it needs to be < radius there is
        no point in getting things < A resolution, not that we can
        :param M: number of rays (360/K) gives you your angle between rays
        :param K: num samples per K
        :return: M, M, K, 6 torch tensor, the reason for the additional diameter is that the initial tensor is now rotated
        and we are getting every possible rotation per descriptor
        """
        prop_names = ['shape_index', 'mean_curvature', 'electrostatic', 'h_bond_donor', 'h_bond_acceptor',
                      'hydrophobicity']
        P = len(prop_names)

        detailed_descriptors = {}

        for center_idx, patch in patches.items():
            center_pos = patch['center']
            center_normal = normals[center_idx]

            patch_indices = patch['indices']
            if len(patch_indices) < 360/M:  # Skip small patches
                continue

            local_indices = np.arange(len(patch_indices))  # 0 to len-1 for local
            patch_vertices = vertices[patch_indices]
            patch_dists = dist_matrix[center_idx, patch_indices]

            vecs = patch_vertices - center_pos
            proj_vecs = vecs - np.outer(np.dot(vecs, center_normal), center_normal)
            proj_norms = np.linalg.norm(proj_vecs, axis=1)

            # Use more robust epsilon for numerical stability
            epsilon_norm = 1e-8
            mask = proj_norms > epsilon_norm  # Avoid zero
            
            # Check if we have enough valid projected vectors
            if np.sum(mask) < 2:
                continue  # Skip patches with insufficient tangent plane data
            
            unit_proj = np.zeros_like(proj_vecs)
            unit_proj[mask] = proj_vecs[mask] / proj_norms[mask, None]

            # Choose reference direction u0: closest neighbor
            nonzero_local = local_indices[mask & (patch_dists > epsilon_norm)]
            if len(nonzero_local) == 0:
                continue
            closest_local = nonzero_local[np.argmin(patch_dists[nonzero_local])]
            u0 = unit_proj[closest_local]
            
            # Ensure u0 is normalized
            u0_norm = np.linalg.norm(u0)
            if u0_norm < epsilon_norm:
                continue
            u0 = u0 / u0_norm
            
            v0 = np.cross(center_normal, u0)
            v0_norm = np.linalg.norm(v0)
            if v0_norm < epsilon_norm:
                continue
            v0 = v0 / v0_norm

            # Compute angles
            dot_u = np.dot(unit_proj, u0)
            dot_v = np.dot(unit_proj, v0)
            theta = np.arctan2(dot_v, dot_u)
            theta[~mask] = 0  # Arbitrary for center

            # Bin edges for M rays
            theta_bins = np.linspace(-np.pi, np.pi, M + 1)

            descriptor = np.zeros((M, K, P))

            sample_d = np.linspace(0, radius, K)

            for i in range(M):
                # Wrap around
                if i == M - 1:
                    bin_mask = (theta >= theta_bins[i]) | (theta < theta_bins[0] + 2 * np.pi)
                else:
                    bin_mask = (theta >= theta_bins[i]) & (theta < theta_bins[i + 1])

                bin_local = local_indices[bin_mask]
                if len(bin_local) < 2:  # Need at least 2 for interp
                    descriptor[i] = 0.0
                    continue

                bin_dists = patch_dists[bin_local]
                sort_idx = np.argsort(bin_dists)
                sorted_d = bin_dists[sort_idx]

                for p, prop in enumerate(prop_names):
                    bin_prop = patch['features'][prop][bin_local]
                    sorted_prop = bin_prop[sort_idx]

                    # Robust interpolation with boundary handling instead of extrapolation
                    # Check if we have valid data range
                    if len(sorted_d) < 2 or sorted_d[-1] - sorted_d[0] < 1e-8:
                        # Insufficient range, use mean value
                        descriptor[i, :, p] = np.mean(sorted_prop)
                        continue
                    
                    # Use bounds_error=False with fill_value for safer interpolation
                    # Fill values outside range with boundary values instead of extrapolating
                    interp_func = interp1d(
                        sorted_d, 
                        sorted_prop, 
                        kind='linear', 
                        bounds_error=False,
                        fill_value=(sorted_prop[0], sorted_prop[-1])
                    )
                    sampled_prop = interp_func(sample_d)
                    
                    # Additional safety: clamp to reasonable range based on input data
                    prop_min, prop_max = np.min(sorted_prop), np.max(sorted_prop)
                    prop_range = prop_max - prop_min
                    if prop_range > 1e-8:
                        # Allow 10% extrapolation beyond observed range
                        safe_min = prop_min - 0.1 * prop_range
                        safe_max = prop_max + 0.1 * prop_range
                        sampled_prop = np.clip(sampled_prop, safe_min, safe_max)

                    descriptor[i, :, p] = sampled_prop

            # Final safety checks for any remaining numerical issues
            descriptor = np.nan_to_num(descriptor, nan=0.0, posinf=0.0, neginf=0.0)
            descriptor = torch.tensor(descriptor).float()
            # Generate M rotations by rolling along the angular dimension
            descriptor = torch.stack([torch.roll(descriptor, shifts=i, dims=0) for i in range(M)])
            detailed_descriptors[center_idx] = descriptor

        return detailed_descriptors

    def _get_patch_residues(self, traj, atoms):
        """
        Because we are also collencting atom ids in each patch we can go back and get the residuies, this is mostly for
        testing and visualizaion which will become importatnt later
        :param atoms: the atoms of the patch
        :return: a dict that looks like
        {'chain':chain_id, 'residue': [list of residue numbers]}
        """

        unique_atom_indices = np.unique(atoms)
        unique_atom_indices = unique_atom_indices[unique_atom_indices >= 0]  # Filter invalid

        residues = set()
        for ai in unique_atom_indices:
            atom = traj.top.atom(ai)
            residues.add(atom.residue)

        # Sort by residue index
        res_list = sorted(residues, key=lambda r: r.index)

        # Format as "Chain C Res 123ALA"
        residues = [{"chain":r.chain.index,
                     "residue": r.resSeq} for r in res_list]

        return residues

    def __call__(self, pdb_file, chain_id, radius_angstrom=25, M=25, K=5, cleanup=True):
        """
        for a given pdb and chain calculate the pathches
        :param radius_angstrom: the radius of the patch
        :param M: number of ray emanating from the center of the patch
        :param K: number of samples per patch, this shuold be >> then radius, we cannot possible have curvatures
        that are less than an atom (we could if you think about electron clouds but we are dealing with meshes)
        :return: ProteinPatches dataclass
        """
        traj, vertices, faces, normals, atoms = self.surface(pdb_file, chain_id)
        properties = self.vertex_properties(traj, vertices, faces)
        patches, dist_matrix =self.extract_geodesic_patches(vertices, faces, radius_angstrom, properties, traj, atoms)
        descriptors=self.compute_detailed_patch_descriptors(patches, vertices, dist_matrix,
                                                            normals, radius_angstrom, M, K)
        protein_patches=ProteinPatches(pdb_file=pdb_file,
                                       radius_angstrom=radius_angstrom,
                                       sampling_angle=360/M,
                                       num_points=K,
                                       patches=patches,
                                       descriptors=descriptors)
        if cleanup:
            items=["tmp.face", "tmp.pdb", "tmp.vert", "tmp.xyzr"]
            for item in items:
                os.remove(item)
        return protein_patches


    def save_patches(self, patches, path):
        with open(path, 'wb') as f:
            pickle.dump(patches, f)


