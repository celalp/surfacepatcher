import pickle
from dataclasses import dataclass

import open3d as o3d

from surfacepatcher.utils import *
from surfacepatcher.process_pdb import PDBProcessor

@dataclass(frozen=True)
class ProteinPatches:
    pdb_file: str
    patches: dict
    fpfh_radius: int
    max_nn: int
    patch_radius: int


class GeodesicPatcher:
    def __init__(self, pdb):
        self.pdb = pdb

    def process(self, apbs_bin, multivalue_bin, cleanup=True):
        processor = PDBProcessor(self.pdb, apbs_bin, multivalue_bin)
        (self.vertices, self.faces, self.normals, self.atom_ids,
         self.potenial_values, self.traj) =processor.process(cleanup=cleanup)

    def vertex_properties(self):
        properties = {}
        properties['h_bond_donor'] = project_hbond_propensity(self.traj, self.vertices, 'donor')
        properties['h_bond_acceptor'] = project_hbond_propensity(self.traj, self.vertices, 'acceptor')
        properties['hydrophobicity'] = project_hydrophobicity(self.traj, self.vertices)
        properties['electrostatic']=self.potenial_values

        return properties

    def get_patches(self, radius, properties, fpfh_radius=5, max_nn=30):
        dist_matrix = compute_geodesic_distances(self.vertices, self.faces)

        #TODO convert to pointcloud here with the normals
        patches = {}
        for center_idx in range(len(self.vertices)):
            pcd = o3d.geometry.PointCloud()
            # Find all vertices within geodesic radius n
            patch_mask = dist_matrix[center_idx] <= radius
            patch_indices = np.where(patch_mask)[0]
            vertices = self.vertices[patch_indices]
            normals = self.normals[patch_indices]
            pcd.points = o3d.utility.Vector3dVector(self.vertices[patch_indices])
            pcd.normals=o3d.utility.Vector3dVector(self.normals[patch_indices]) # no need to calculate again
            fpfh_features=o3d.pipelines.registration.compute_fpfh_feature(pcd,
                                                                          o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius,
                                                                                                               max_nn=max_nn))

            # Store patch features
            patch_features = {}
            for prop_name, prop_values in properties.items():
                patch_features[prop_name] = prop_values[patch_indices]

            patches[center_idx] = {
                'center': self.vertices[center_idx],
                'indices': patch_indices,
                'biochem_features': patch_features,
                'fpfh_features':fpfh_features,
                'pcd':pcd,
                'vertices':vertices,
                'normals':normals,
                'area': np.sum(patch_mask),  # Normalize by patch area later, maybe
                'atom_ids': self.atom_ids[patch_indices],  # Add this
                'residues': get_patch_residues(self.traj, self.atom_ids[patch_indices]),
            }

        return patches

    def save_patches(self, patches, path):
        with open(path, 'wb') as f:
            pickle.dump(patches, f)
