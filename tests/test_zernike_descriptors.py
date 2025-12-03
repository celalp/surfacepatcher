
import unittest
import numpy as np
import sys
import os

# Ensure we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from surfacepatcher.zernike_descriptors import Zernike3D, compute_zernike_descriptor_for_patch

class TestZernike3D(unittest.TestCase):
    def setUp(self):
        self.zernike = Zernike3D(order=5, grid_size=16)
        # Create points roughly in a unit sphere
        points = np.random.randn(100, 3)
        self.random_points = points / np.linalg.norm(points, axis=1)[:, None] * np.random.rand(100, 1)

    def test_init(self):
        z = Zernike3D(order=10, grid_size=32)
        self.assertEqual(z.order, 10)
        self.assertEqual(z.grid_size, 32)

    def test_voxelize_patch_binary(self):
        grid = self.zernike._voxelize_patch(self.random_points)
        self.assertEqual(grid.shape, (16, 16, 16))
        self.assertTrue(np.all((grid == 0) | (grid == 1)))
        self.assertTrue(np.sum(grid) > 0)

    def test_voxelize_patch_features(self):
        features = np.random.rand(100, 1)
        grid = self.zernike._voxelize_patch(self.random_points, features=features)
        self.assertEqual(grid.shape, (16, 16, 16))
        # Should have values other than 0 and 1 (likely)
        self.assertTrue(np.any((grid > 0) & (grid < 1)))

    def test_cartesian_to_spherical(self):
        x, y, z = 1.0, 0.0, 0.0
        r, theta, phi = self.zernike._cartesian_to_spherical(x, y, z)
        self.assertTrue(np.isclose(r, 1.0))
        self.assertTrue(np.isclose(theta, np.pi/2)) # z=0 -> theta=90 deg
        self.assertTrue(np.isclose(phi, 0.0))

        x, y, z = 0.0, 0.0, 1.0
        r, theta, phi = self.zernike._cartesian_to_spherical(x, y, z)
        self.assertTrue(np.isclose(r, 1.0))
        self.assertTrue(np.isclose(theta, 0.0)) # z=1 -> theta=0 deg

    def test_zernike_polynomial_radial(self):
        r = np.linspace(0, 1, 10)
        # R_0^0(r) = 1
        poly = self.zernike._zernike_polynomial_radial(0, 0, r)
        self.assertTrue(np.allclose(poly, 1.0))

    def test_compute_descriptors(self):
        desc = self.zernike.compute_descriptors(self.random_points)
        self.assertIsInstance(desc, np.ndarray)
        self.assertTrue(len(desc) > 0)
        self.assertFalse(np.any(np.isnan(desc)))

    def test_rotation_invariance(self):
        # Rotate points
        theta = np.pi / 4
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        rotated_points = self.random_points @ rotation_matrix.T
        
        desc1 = self.zernike.compute_descriptors(self.random_points)
        desc2 = self.zernike.compute_descriptors(rotated_points)
        
        # Should be relatively close (perfect invariance is hard with voxelization)
        # Checking correlation or distance
        dist = np.linalg.norm(desc1 - desc2)
        norm = np.linalg.norm(desc1)
        
        # Allow some error due to discretization
        self.assertTrue(dist / norm < 0.2)

class TestZernikePatch(unittest.TestCase):
    def test_compute_zernike_descriptor_for_patch(self):
        # Mock data
        num_points = 50
        vertices_full = np.random.rand(200, 3)
        indices = np.arange(num_points)
        
        features = {
            'shape_index': np.random.rand(num_points),
            'mean_curvature': np.random.rand(num_points),
            'electrostatic': np.random.rand(num_points),
            'h_bond_donor': np.random.rand(num_points),
            'h_bond_acceptor': np.random.rand(num_points),
            'hydrophobicity': np.random.rand(num_points)
        }
        
        patch = {
            'indices': indices,
            'features': features
        }
        
        desc = compute_zernike_descriptor_for_patch(patch, vertices_full, order=5, grid_size=16)
        
        # Check length: 7 channels (1 shape + 6 features)
        # Calculate expected length for one channel
        z = Zernike3D(order=5)
        dummy_desc = z.compute_descriptors(np.random.rand(10, 3))
        single_len = len(dummy_desc)
        
        self.assertEqual(len(desc), single_len * 7)
        self.assertFalse(np.any(np.isnan(desc)))

    def test_compute_zernike_descriptor_missing_features(self):
        # Test robustness when some features are missing
        num_points = 50
        vertices_full = np.random.rand(200, 3)
        indices = np.arange(num_points)
        
        # Only provide electrostatic
        features = {
            'electrostatic': np.random.rand(num_points)
        }
        
        patch = {
            'indices': indices,
            'features': features
        }
        
        desc = compute_zernike_descriptor_for_patch(patch, vertices_full, order=5, grid_size=16)
        
        # Should still produce 7 channels, but missing ones might be skipped or handled?
        # Current implementation SKIPS missing features in the loop but appends to list.
        # So if 5 are missing, length will be shorter.
        
        z = Zernike3D(order=5)
        dummy_desc = z.compute_descriptors(np.random.rand(10, 3))
        single_len = len(dummy_desc)
        
        # 1 shape + 1 feature (electrostatic) = 2 channels
        self.assertEqual(len(desc), single_len * 2)

if __name__ == '__main__':
    unittest.main()
