"""
Helper utilities for point cloud and topological comparison methods.
Provides functions to reconstruct surface information from patches.
"""

import mdtraj as md
from surfacepatcher.utils import compute_msms_surface


def load_surface_from_pdb(pdb_file, chain_id=None):
    """
    Load protein surface vertices and faces from PDB file.
    
    :param pdb_file: Path to PDB file
    :param chain_id: Optional chain ID to select
    :return: (traj, vertices, faces, normals, atom_ids)
    """
    traj = md.load(pdb_file)
    if chain_id is not None:
        traj = traj.atom_slice(traj.topology.select(f"chainid == {chain_id}"))
    
    vertices, faces, normals, atom_ids = compute_msms_surface(traj.xyz[0], traj)
    return traj, vertices, faces, normals, atom_ids


def get_patch_vertices(patch, full_vertices):
    """
    Extract vertex coordinates for a patch from the full surface.
    
    :param patch: Patch dict with 'indices' key
    :param full_vertices: (N, 3) array of all surface vertices
    :return: (M, 3) array of patch vertices
    """
    indices = patch['indices']
    return full_vertices[indices]


def get_patch_normals(patch, full_normals):
    """
    Extract normals for a patch from the full surface.
    
    :param patch: Patch dict with 'indices' key
    :param full_normals: (N, 3) array of all surface normals
    :return: (M, 3) array of patch normals
    """
    indices = patch['indices']
    return full_normals[indices]


class SurfaceCache:
    """
    Cache surface data for efficient access during comparison.
    Stores vertices, normals, and faces for proteins.
    """
    
    def __init__(self):
        self.cache = {}
    
    def load_surface(self, pdb_file, chain_id=None):
        """
        Load and cache surface data for a protein.
        
        :param pdb_file: Path to PDB file
        :param chain_id: Optional chain ID
        :return: (traj, vertices, faces, normals, atom_ids)
        """
        cache_key = (pdb_file, chain_id)
        
        if cache_key not in self.cache:
            surface_data = load_surface_from_pdb(pdb_file, chain_id)
            self.cache[cache_key] = surface_data
        
        return self.cache[cache_key]
    
    def get_patch_data(self, pdb_file, patch, chain_id=None):
        """
        Get vertices and normals for a specific patch.
        
        :param pdb_file: Path to PDB file
        :param patch: Patch dict
        :param chain_id: Optional chain ID
        :return: (vertices, normals)
        """
        _, full_vertices, _, full_normals, _ = self.load_surface(pdb_file, chain_id)
        
        vertices = get_patch_vertices(patch, full_vertices)
        normals = get_patch_normals(patch, full_normals)
        
        return vertices, normals
