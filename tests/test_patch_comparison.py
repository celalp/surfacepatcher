
import unittest
import numpy as np
import sys
import os
import torch
from unittest.mock import MagicMock

# Ensure we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from surfacepatcher.patch_comparison import PatchComparison
from surfacepatcher.geodesic_patcher import ProteinPatches

class TestPatchComparison(unittest.TestCase):
    def setUp(self):
        # Create dummy ProteinPatches
        self.patches1 = MagicMock(spec=ProteinPatches)
        self.patches2 = MagicMock(spec=ProteinPatches)
        
        # Mock descriptors: (M_rot, M_ang, K, P)
        # Let's say M_rot=4, M_ang=4, K=2, P=6
        M_rot, M_ang, K, P = 4, 4, 2, 6
        
        # 2 patches for protein 1
        self.patches1.descriptors = {
            0: torch.rand(M_rot, M_ang, K, P),
            1: torch.rand(M_rot, M_ang, K, P)
        }
        
        # 3 patches for protein 2
        self.patches2.descriptors = {
            10: torch.rand(M_rot, M_ang, K, P),
            11: torch.rand(M_rot, M_ang, K, P),
            12: torch.rand(M_rot, M_ang, K, P)
        }
        
        self.comparison = PatchComparison(self.patches1, self.patches2)

    def test_init(self):
        self.assertTrue(hasattr(self.comparison, 'feature_weights'))
        self.assertEqual(len(self.comparison.feature_weights), 6)

    def test_compute(self):
        dists, idxs, k1, k2 = self.comparison.compute(batch_size=2)
        
        self.assertEqual(dists.shape, (2, 3)) # 2 patches in p1, 3 in p2
        self.assertEqual(idxs.shape, (2, 3))
        self.assertEqual(len(k1), 2)
        self.assertEqual(len(k2), 3)
        self.assertEqual(k1, [0, 1])
        self.assertEqual(k2, [10, 11, 12])

    def test_get_preset_weights(self):
        w = PatchComparison.get_preset_weights('epitope')
        self.assertEqual(len(w), 6)
        
        with self.assertRaises(ValueError):
            PatchComparison.get_preset_weights('invalid')

    def test_get_feature_weights(self):
        w_dict = self.comparison.get_feature_weights()
        self.assertIn('shape_index', w_dict)
        self.assertEqual(len(w_dict), 6)

if __name__ == '__main__':
    unittest.main()
