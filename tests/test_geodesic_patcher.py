
import unittest
import numpy as np
import sys
import os
import torch
from unittest.mock import MagicMock, patch

# Ensure we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from surfacepatcher.geodesic_patcher import GeodesicPatcher, ProteinPatches

class TestGeodesicPatcher(unittest.TestCase):
    def setUp(self):
        self.patcher = GeodesicPatcher()
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.pdb_file = os.path.join(self.data_dir, 'D1YSM5.pdb')

    def test_surface(self):
        # Real integration test using D1YSM5.pdb
        traj, v, f, n, a = self.patcher.surface(self.pdb_file)
        
        self.assertIsNotNone(traj)
        self.assertTrue(len(v) > 0)
        self.assertTrue(len(f) > 0)
        self.assertTrue(len(n) > 0)
        self.assertTrue(len(a) > 0)
        self.assertEqual(len(v), len(n))
        self.assertEqual(len(v), len(a))

    def test_vertex_properties(self):
        # Use real surface data
        traj, v, f, n, a = self.patcher.surface(self.pdb_file)
        
        props = self.patcher.vertex_properties(traj, v, f)
        
        self.assertIn('shape_index', props)
        self.assertIn('mean_curvature', props)
        self.assertIn('electrostatic', props)
        self.assertIn('h_bond_donor', props)
        self.assertIn('h_bond_acceptor', props)
        self.assertIn('hydrophobicity', props)
        
        # Check shapes
        self.assertEqual(len(props['shape_index']), len(v))
        self.assertEqual(len(props['mean_curvature']), len(v))

    def test_extract_geodesic_patches(self):
        # Use real surface data
        traj, v, f, n, a = self.patcher.surface(self.pdb_file)
        props = self.patcher.vertex_properties(traj, v, f)
        
        radius = 5.0
        # Limit to a subset of vertices for speed if necessary, 
        # but extract_geodesic_patches iterates over all vertices.
        # For testing, we might want to mock the loop or just test a small protein.
        # D1YSM5 is small (96KB), so it should be fine.
        
        patches, dist_matrix = self.patcher.extract_geodesic_patches(v, f, radius, props, traj, a)
        
        self.assertTrue(len(patches) > 0)
        first_patch = next(iter(patches.values()))
        self.assertIn('center', first_patch)
        self.assertIn('indices', first_patch)
        self.assertIn('features', first_patch)
        self.assertIn('residues', first_patch)

    def test_compute_detailed_patch_descriptors(self):
        # End-to-end test with real data
        traj, v, f, n, a = self.patcher.surface(self.pdb_file)
        props = self.patcher.vertex_properties(traj, v, f)
        radius = 5.0
        patches, dist_matrix = self.patcher.extract_geodesic_patches(v, f, radius, props, traj, a)
        
        # Only compute for a few patches to save time
        subset_patches = {k: patches[k] for k in list(patches.keys())[:5]}
        
        M = 4
        K = 2
        descriptors = self.patcher.compute_detailed_patch_descriptors(subset_patches, v, dist_matrix, n, radius=radius, M=M, K=K)
        
        self.assertTrue(len(descriptors) > 0)
        first_desc = next(iter(descriptors.values()))
        self.assertIsInstance(first_desc, torch.Tensor)
        # Shape: (M, M, K, 6)
        self.assertEqual(first_desc.shape, (M, M, K, 6))

if __name__ == '__main__':
    unittest.main()
