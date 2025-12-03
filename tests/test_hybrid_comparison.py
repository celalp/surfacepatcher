
import unittest
import numpy as np
import sys
import os

# Ensure we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from surfacepatcher.hybrid_comparison import HybridComparison

class TestHybridComparison(unittest.TestCase):
    def setUp(self):
        # Mock results: (distances, keys1, keys2) or (distances, indices, keys1, keys2)
        self.geo_res = (np.array([[0.1, 0.8], [0.5, 0.2]]), np.zeros((2, 2)), ['a', 'b'], ['x', 'y'])
        self.pc_res = (np.array([[0.2, 0.7], [0.6, 0.3]]), ['a', 'b'], ['x', 'y'])
        self.topo_res = (np.array([[0.3, 0.6], [0.4, 0.1]]), ['a', 'b'], ['x', 'y'])
        
        self.hybrid = HybridComparison(
            geodesic_results=self.geo_res,
            pointcloud_results=self.pc_res,
            topological_results=self.topo_res
        )

    def test_init(self):
        self.assertEqual(self.hybrid.fusion_method, 'weighted_sum')
        self.assertAlmostEqual(sum(self.hybrid.weights.values()), 1.0)

    def test_get_preset_weights(self):
        w = HybridComparison.get_preset_weights('epitope')
        self.assertIn('geodesic', w)
        
        with self.assertRaises(ValueError):
            HybridComparison.get_preset_weights('invalid')

    def test_normalize_distances(self):
        d = np.array([[0, 10], [5, 5]])
        norm = self.hybrid._normalize_distances(d)
        self.assertEqual(np.min(norm), 0.0)
        self.assertEqual(np.max(norm), 1.0)
        self.assertEqual(norm[0, 1], 1.0)

    def test_rank_distances(self):
        d = np.array([[0.1, 0.4], [0.2, 0.3]])
        # Flatten: 0.1, 0.4, 0.2, 0.3
        # Ranks: 0, 3, 1, 2
        # Norm ranks: 0/3, 3/3, 1/3, 2/3
        ranks = self.hybrid._rank_distances(d)
        self.assertAlmostEqual(ranks[0, 0], 0.0)
        self.assertAlmostEqual(ranks[0, 1], 1.0)

    def test_compute_weighted_sum(self):
        combined, k1, k2 = self.hybrid.compute()
        self.assertEqual(combined.shape, (2, 2))
        self.assertEqual(k1, ['a', 'b'])
        self.assertEqual(k2, ['x', 'y'])

    def test_compute_rank_fusion(self):
        self.hybrid.fusion_method = 'rank_fusion'
        combined, k1, k2 = self.hybrid.compute()
        self.assertEqual(combined.shape, (2, 2))

    def test_compute_product(self):
        self.hybrid.fusion_method = 'product'
        combined, k1, k2 = self.hybrid.compute()
        self.assertEqual(combined.shape, (2, 2))

    def test_get_method_agreement(self):
        corr = self.hybrid.get_method_agreement()
        self.assertIn('geodesic_vs_pointcloud', corr)
        self.assertIn('geodesic_vs_topological', corr)
        self.assertIn('pointcloud_vs_topological', corr)

    def test_get_top_matches(self):
        matches = self.hybrid.get_top_matches(n=2)
        self.assertEqual(len(matches), 2)
        # Check structure: (k1, k2, dist)
        self.assertEqual(len(matches[0]), 3)

if __name__ == '__main__':
    unittest.main()
