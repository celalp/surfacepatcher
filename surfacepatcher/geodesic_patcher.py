import shelve
from dataclasses import dataclass
from functools import cached_property

import numpy
import open3d as o3d

from surfacepatcher.utils import *
from surfacepatcher.process_pdb import PDBProcessor

# keeping these bare bones for now no methods

@dataclass(frozen=True)
class ProteinPatch:
    center: int
    indices: numpy.ndarray
    biochem_features: dict
    fpfh_features: numpy.ndarray
    atom_ids: numpy.ndarray
    residues: list
    skip:bool=False

    @cached_property
    def pcd(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.vertices)
        pcd.normals = o3d.utility.Vector3dVector(self.normals)
        return pcd

    @cached_property
    def fpfh(self):
        fpfh=o3d.pipelines.registration.Feature()
        fpfh.data=self.fpfh_features
        return fpfh


@dataclass(frozen=True)
class ProteinPatches:
    pdb_file: str
    patches: dict  # dict[int, ProteinPatch]
    fpfh_radius: int
    max_nn: int
    patch_radius: int
    filter_radius: int
    vertices: numpy.ndarray
    normals: numpy.ndarray
    shelve_path: str = None  # If loaded from shelve, store the path

    @classmethod
    def from_shelve(cls, shelve_path, load_patches=True):
        """
        Create a ProteinPatches object from a shelve database.

        Args:
            shelve_path: Path to shelve database (without extension)
            load_patches: If True, loads all patches into memory.
                         If False, creates empty patches dict (load on-demand later).

        Returns:
            ProteinPatches object
        """
        with shelve.open(shelve_path, 'r') as db:
            metadata = db['metadata']

            if load_patches:
                # Load all patches into memory
                patches = {}
                for key in db.keys():
                    if key != 'metadata':
                        patches[int(key)] = db[key]
            else:
                # Just metadata, no patches loaded
                patches = {}

            return cls(
                pdb_file=metadata['pdb_file'],
                patches=patches,
                fpfh_radius=metadata['fpfh_radius'],
                max_nn=metadata['max_nn'],
                patch_radius=metadata['patch_radius'],
                filter_radius=metadata['filter_radius'],
                shelve_path=shelve_path
            )

    def get_patch(self, patch_id):
        """
        Get a patch by ID. If patches dict is empty, loads from shelve.
        If already in memory, returns from memory.

        Args:
            patch_id: ID of patch to retrieve

        Returns:
            ProteinPatch object
        """
        # If already in memory, return it
        if patch_id in self.patches:
            return self.patches[patch_id]

        # Otherwise load from shelve
        if self.shelve_path is None:
            raise ValueError(f"Patch {patch_id} not found and no shelve_path available")

        return self.load_patch(self.shelve_path, patch_id)

    @cached_property
    def dist_matrix(self):
        dist_matrix = compute_geodesic_distances(self.vertices, self.faces)
        return dist_matrix

    def get_neighbors(self, patch_id, radius=None, return_patches=False):
        mask=self.dist_matrix[patch_id]<=radius
        patch_indices = np.where(mask)[0]
        if return_patches:
            patches=self.load_patches(self.shelve_path, patch_indices)
            return patches
        else:
            return patch_indices


    def save_patches(self, shelve_path):
        """Save patches to a shelve database for random access by key."""
        with shelve.open(shelve_path, 'n') as db:
            db['metadata'] = {
                'pdb_file': self.pdb_file,
                'fpfh_radius': self.fpfh_radius,
                'max_nn': self.max_nn,
                'patch_radius': self.patch_radius,
                'vertices': self.vertices,
                'normals': self.normals,
                'filter_radius': self.filter_radius,
                'num_patches': len(self.patches)
            }

            for patch_id, patch in self.patches.items():
                db[str(patch_id)] = patch

    @staticmethod
    def load_patch(shelve_path, patch_id):
        """Load a single patch by ID from shelve database."""
        with shelve.open(shelve_path, 'r') as db:
            return patch_id, db[str(patch_id)]

    @staticmethod
    def load_patches(shelve_path, patch_ids):
        """Load multiple patches by IDs from shelve database."""
        patches = {}
        with shelve.open(shelve_path, 'r') as db:
            for patch_id in patch_ids:
                key = str(patch_id)
                if key in db:
                    patches[patch_id] = db[key]
        return patches

    @staticmethod
    def get_all_patch_ids(shelve_path):
        """Get all available patch IDs from shelve database."""
        with shelve.open(shelve_path, 'r') as db:
            return [int(key) for key in db.keys() if key != 'metadata']

    def __getitem__(self, patch_id):
        """Get a patch by ID (loads from shelve if needed)."""
        return self.get_patch(patch_id)

    def __len__(self):
        """Number of patches (loaded or available in shelve)."""
        if self.patches:
            return len(self.patches)
        elif self.shelve_path:
            return len(self.get_all_patch_ids(self.shelve_path))
        return 0

    def __iter__(self):
        """Iterate over patches (from memory or shelve)."""
        if self.patches:
            return iter(self.patches.items())
        else:
            return self.iter_patches_lazy()

    def iter_patches_lazy(self, patch_ids=None):
        """
        Iterate over patches, loading from shelve on-demand.

        Args:
            patch_ids: Optional list of specific patch IDs to iterate.
                      If None, iterates over all patches in shelve.

        Yields:
            Tuple of (patch_id: int, patch: ProteinPatch)
        """
        if self.shelve_path is None:
            # If no shelve path, just iterate over loaded patches
            if patch_ids:
                for pid in patch_ids:
                    if pid in self.patches:
                        yield pid, self.patches[pid]
            else:
                yield from self.patches.items()
        else:
            # Load from shelve on-demand
            if patch_ids is None:
                patch_ids = self.get_all_patch_ids(self.shelve_path)

            with shelve.open(self.shelve_path, 'r') as db:
                for patch_id in patch_ids:
                    key = str(patch_id)
                    if key in db:
                        yield patch_id, db[key]



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

    def get_patches(self, radius, properties, filter_radius=10, fpfh_radius=5, max_nn=30):
        dist_matrix = compute_geodesic_distances(self.vertices, self.faces)
        patches = {}
        to_skip=[]
        for center_idx in range(len(self.vertices)):
            if center_idx in to_skip:
                skip=True
            else:
                skip=False
            pcd = o3d.geometry.PointCloud()
            # Find all vertices within geodesic radius n
            patch_mask = dist_matrix[center_idx] <= radius
            if not skip:
                to_skip.extend(np.where(dist_matrix[center_idx] <= filter_radius)[0].tolist())
                to_skip=list(set(to_skip))
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

            patches[center_idx] = \
                ProteinPatch(center_idx,
                             patch_indices,
                             patch_features,
                             fpfh_features.data,
                             self.atom_ids[patch_indices],
                             get_patch_residues(self.traj, self.atom_ids[patch_indices]),
                             skip)

        patches = ProteinPatches(self.pdb, patches, fpfh_radius, max_nn, radius, filter_radius, self.vertices, self.normals)
        return patches

