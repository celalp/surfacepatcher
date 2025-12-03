
import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Ensure we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from surfacepatcher.pointcloud_comparison import PointCloudComparison
from surfacepatcher.geodesic_patcher import ProteinPatches

class TestPointCloudComparison(unittest.TestCase):
    def setUp(self):
        self.patches1 = MagicMock(spec=ProteinPatches)
        self.patches1.pdb_file = 'p1.pdb'
        self.patches1.patches = {
            0: {'indices': np.array([0, 1]), 'features': {
                'shape_index': np.zeros(2), 'mean_curvature': np.zeros(2),
                'electrostatic': np.zeros(2), 'h_bond_donor': np.zeros(2),
                'h_bond_acceptor': np.zeros(2), 'hydrophobicity': np.zeros(2)
            }}
        }
        
        self.patches2 = MagicMock(spec=ProteinPatches)
        self.patches2.pdb_file = 'p2.pdb'
        self.patches2.patches = {
            10: {'indices': np.array([0, 1]), 'features': {
                'shape_index': np.zeros(2), 'mean_curvature': np.zeros(2),
                'electrostatic': np.zeros(2), 'h_bond_donor': np.zeros(2),
                'h_bond_acceptor': np.zeros(2), 'hydrophobicity': np.zeros(2)
            }}
        }
        
        self.mock_cache = MagicMock()
        # Mock load_surface to return vertices and normals
        self.mock_cache.load_surface.return_value = (None, np.zeros((10, 3)), None, np.zeros((10, 3)), None)

    @patch('surfacepatcher.pointcloud_comparison.o3d')
    def test_patch_to_pointcloud(self, mock_o3d):
        pc = PointCloudComparison(self.patches1, self.patches2, surface_cache=self.mock_cache)
        
        patch = self.patches1.patches[0]
        vertices = np.zeros((10, 3))
        
        pcd, features = pc._patch_to_pointcloud(patch, vertices)
        
        self.assertEqual(features.shape, (2, 6))
        mock_o3d.geometry.PointCloud.assert_called()

    @patch('surfacepatcher.pointcloud_comparison.o3d')
    def test_compute_fpfh(self, mock_o3d):
        pc = PointCloudComparison(self.patches1, self.patches2, surface_cache=self.mock_cache)
        
        mock_pcd = MagicMock()
        mock_fpfh = MagicMock()
        mock_fpfh.data = np.zeros((33, 5)) # 5 points, 33 dims
        mock_o3d.pipelines.registration.compute_fpfh_feature.return_value = mock_fpfh
        
        desc = pc._compute_fpfh(mock_pcd, radius=1.0)
        
        self.assertEqual(desc.shape, (5, 33))

    def test_aggregate_descriptors(self):
        pc = PointCloudComparison(self.patches1, self.patches2, surface_cache=self.mock_cache)
        
        geom_desc = np.random.rand(10, 33)
        biochem = np.random.rand(10, 6)
        
        agg = pc._aggregate_descriptors(geom_desc, biochem)
        
        # 33*4 + 6*4 = 132 + 24 = 156
        self.assertEqual(len(agg), 156)

    @patch('surfacepatcher.pointcloud_comparison.o3d')
    def test_compute_patch_descriptor_fpfh(self, mock_o3d):
        pc = PointCloudComparison(self.patches1, self.patches2, descriptor_type='fpfh', surface_cache=self.mock_cache)
        
        # Mock FPFH computation
        mock_fpfh = MagicMock()
        mock_fpfh.data = np.zeros((33, 2))
        mock_o3d.pipelines.registration.compute_fpfh_feature.return_value = mock_fpfh
        
        # Mock point cloud points
        mock_pcd = MagicMock()
        mock_pcd.points = np.zeros((2, 3))
        mock_o3d.geometry.PointCloud.return_value = mock_pcd
        
        patch = self.patches1.patches[0]
        vertices = np.zeros((10, 3))
        
        desc = pc._compute_patch_descriptor(patch, vertices)
        
        self.assertEqual(len(desc), 156)

    @patch('surfacepatcher.pointcloud_comparison.o3d')
    def test_compute_patch_descriptor_zernike(self, mock_o3d):
        # This requires mocking Zernike3D import inside the method or patching it
        with patch('surfacepatcher.zernike_descriptors.Zernike3D') as MockZernike:
            mock_zernike_instance = MockZernike.return_value
            # Mock compute_descriptors to return a vector of size N
            mock_zernike_instance.compute_descriptors.return_value = np.zeros(50)
            
            pc = PointCloudComparison(self.patches1, self.patches2, descriptor_type='zernike', surface_cache=self.mock_cache)
            
            # Mock point cloud points
            mock_pcd = MagicMock()
            mock_pcd.points = np.zeros((2, 3))
            mock_o3d.geometry.PointCloud.return_value = mock_pcd
            
            protein_patch = self.patches1.patches[0]
            vertices = np.zeros((10, 3))
            
            desc = pc._compute_patch_descriptor(protein_patch, vertices)
            
            # 50 (zernike) + 24 (biochem stats) = 74
            self.assertEqual(len(desc), 74)

    @patch('surfacepatcher.pointcloud_comparison.o3d')
    def test_compute(self, mock_o3d):
        # Mock _compute_patch_descriptor to return fixed size vector
        with patch.object(PointCloudComparison, '_compute_patch_descriptor', return_value=np.zeros(10)):
            pc = PointCloudComparison(self.patches1, self.patches2, surface_cache=self.mock_cache)
            
            dists, k1, k2 = pc.compute()
            
            self.assertEqual(dists.shape, (1, 1))
            self.assertEqual(k1, [0])
            self.assertEqual(k2, [10])

if __name__ == '__main__':
    unittest.main()
