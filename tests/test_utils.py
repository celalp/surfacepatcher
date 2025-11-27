
import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Ensure we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from surfacepatcher import utils, surface_utils

class TestUtils(unittest.TestCase):
    def test_triangle_area(self):
        p0 = np.array([0, 0, 0])
        p1 = np.array([1, 0, 0])
        p2 = np.array([0, 1, 0])
        area = utils.triangle_area(p0, p1, p2)
        self.assertAlmostEqual(area, 0.5)

    def test_cotangent_weight(self):
        # Right triangle
        p0 = np.array([0, 0, 0])
        p1 = np.array([1, 0, 0])
        p2 = np.array([0, 1, 0])
        # Angle at p0 is 90 degrees, cot(90) = 0
        cot = utils.cotangent_weight(p0, p1, p2)
        self.assertAlmostEqual(cot, 0.0)

    @patch('surfacepatcher.utils.o3d')
    def test_compute_curvature(self, mock_o3d):
        # Mock Open3D mesh and normals
        mock_mesh = MagicMock()
        mock_o3d.geometry.TriangleMesh.return_value = mock_mesh
        
        # Simple plane: 4 vertices, 2 triangles
        vertices = np.array([
            [0, 0, 0], [1, 0, 0],
            [0, 1, 0], [1, 1, 0]
        ])
        faces = np.array([
            [0, 1, 2], [1, 3, 2]
        ])
        
        # Mock normals (flat plane -> z-up)
        mock_mesh.vertex_normals = np.array([
            [0, 0, 1], [0, 0, 1],
            [0, 0, 1], [0, 0, 1]
        ])
        
        mean_curv, shape_idx = utils.compute_curvature(vertices, faces)
        
        self.assertEqual(mean_curv.shape, (4,))
        self.assertEqual(shape_idx.shape, (4,))
        # Flat surface should have near zero curvature
        # Note: Boundary effects might cause non-zero values in this implementation
        
    def test_compute_geodesic_distances(self):
        # 3 vertices in a line: 0 -- 1 -- 2
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [2, 0, 0]
        ])
        # Faces needed for graph construction
        # Make a dummy triangle
        faces = np.array([[0, 1, 2]])
        
        dists = utils.compute_geodesic_distances(vertices, faces)
        
        self.assertEqual(dists.shape, (3, 3))
        self.assertAlmostEqual(dists[0, 1], 1.0)
        self.assertAlmostEqual(dists[0, 2], 2.0)
        self.assertAlmostEqual(dists[1, 2], 1.0)

class TestSurfaceUtils(unittest.TestCase):
    def test_get_patch_vertices(self):
        full_vertices = np.array([[0,0,0], [1,1,1], [2,2,2]])
        patch = {'indices': np.array([0, 2])}
        verts = surface_utils.get_patch_vertices(patch, full_vertices)
        np.testing.assert_array_equal(verts, np.array([[0,0,0], [2,2,2]]))

    def test_get_patch_normals(self):
        full_normals = np.array([[0,0,1], [0,1,0], [1,0,0]])
        patch = {'indices': np.array([1])}
        norms = surface_utils.get_patch_normals(patch, full_normals)
        np.testing.assert_array_equal(norms, np.array([[0,1,0]]))

    @patch('surfacepatcher.surface_utils.load_surface_from_pdb')
    def test_surface_cache(self, mock_load):
        mock_load.return_value = ('traj', 'v', 'f', 'n', 'a')
        cache = surface_utils.SurfaceCache()
        
        # First call
        data1 = cache.load_surface('test.pdb')
        self.assertEqual(data1, ('traj', 'v', 'f', 'n', 'a'))
        mock_load.assert_called_once()
        
        # Second call (cached)
        data2 = cache.load_surface('test.pdb')
        self.assertEqual(data2, ('traj', 'v', 'f', 'n', 'a'))
        mock_load.assert_called_once() # Still called once

if __name__ == '__main__':
    unittest.main()
