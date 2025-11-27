
import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Ensure we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from surfacepatcher.topological_comparison import TopologicalComparison
from surfacepatcher.geodesic_patcher import ProteinPatches

class TestTopologicalComparison(unittest.TestCase):
    def setUp(self):
        self.patches1 = MagicMock(spec=ProteinPatches)
        self.patches1.pdb_file = 'p1.pdb'
        self.patches1.patches = {
            0: {'indices': np.array([0, 1, 2, 3]), 'features': {
                'electrostatic': np.zeros(4),
                'hydrophobicity': np.zeros(4)
            }}
        }
        
        self.patches2 = MagicMock(spec=ProteinPatches)
        self.patches2.pdb_file = 'p2.pdb'
        self.patches2.patches = {
            10: {'indices': np.array([0, 1, 2, 3]), 'features': {
                'electrostatic': np.zeros(4),
                'hydrophobicity': np.zeros(4)
            }}
        }
        
        self.mock_cache = MagicMock()
        self.mock_cache.load_surface.return_value = (None, np.zeros((10, 3)), None, np.zeros((10, 3)), None)

    @patch('surfacepatcher.topological_comparison.VietorisRipsPersistence')
    def test_init(self, mock_vr):
        tc = TopologicalComparison(self.patches1, self.patches2, surface_cache=self.mock_cache)
        mock_vr.assert_called()
        self.assertIsNotNone(tc.vr_persistence)

    @patch('surfacepatcher.topological_comparison.VietorisRipsPersistence')
    def test_compute_persistence_diagram(self, mock_vr):
        tc = TopologicalComparison(self.patches1, self.patches2, surface_cache=self.mock_cache)
        
        mock_vr_instance = mock_vr.return_value
        # Mock fit_transform to return a diagram
        # Shape: (n_samples, n_features, 3)
        mock_vr_instance.fit_transform.return_value = np.zeros((1, 5, 3))
        
        points = np.zeros((4, 3))
        diagram = tc._compute_persistence_diagram(points)
        
        self.assertEqual(diagram.shape, (5, 3))

    @patch('surfacepatcher.topological_comparison.VietorisRipsPersistence')
    @patch('surfacepatcher.topological_comparison.PersistenceEntropy')
    @patch('surfacepatcher.topological_comparison.Amplitude')
    @patch('surfacepatcher.topological_comparison.NumberOfPoints')
    def test_vectorize_persistence_diagram(self, mock_nop, mock_amp, mock_ent, mock_vr):
        tc = TopologicalComparison(self.patches1, self.patches2, surface_cache=self.mock_cache)
        
        # Mock transformers
        mock_ent.return_value.fit_transform.return_value = np.zeros((1, 1))
        mock_amp.return_value.fit_transform.return_value = np.zeros((1, 1))
        mock_nop.return_value.fit_transform.return_value = np.zeros((1, 3)) # 3 dims
        
        diagram = np.zeros((5, 3))
        # Set some birth/death/dim values
        diagram[:, 2] = np.array([0, 0, 1, 1, 2])
        
        vec = tc._vectorize_persistence_diagram(diagram)
        
        # Length check:
        # Entropy: 1
        # Amplitude: 5 metrics * 1 = 5
        # Num points: 3
        # Stats: 3 dims * 11 stats = 33
        # Total = 1 + 5 + 3 + 33 = 42
        self.assertEqual(len(vec), 42)

    @patch('surfacepatcher.topological_comparison.VietorisRipsPersistence')
    def test_compute_patch_descriptor(self, mock_vr):
        tc = TopologicalComparison(self.patches1, self.patches2, surface_cache=self.mock_cache)
        
        # Mock vectorize to return fixed size
        with patch.object(tc, '_vectorize_persistence_diagram', return_value=np.zeros(10)):
            # Mock compute_diagram to return dummy
            with patch.object(tc, '_compute_persistence_diagram', return_value=np.zeros((5, 3))):
                # Mock feature weighted persistence
                with patch.object(tc, '_compute_feature_weighted_persistence', return_value=np.zeros((5, 3))):
                    
                    patch_data = self.patches1.patches[0]
                    vertices = np.zeros((10, 3))
                    
                    desc = tc._compute_patch_descriptor(patch_data, vertices)
                    
                    # 3 descriptors (geom, elec, hydro) * 10 = 30
                    self.assertEqual(len(desc), 30)

    @patch('surfacepatcher.topological_comparison.VietorisRipsPersistence')
    def test_compute(self, mock_vr):
        tc = TopologicalComparison(self.patches1, self.patches2, surface_cache=self.mock_cache)
        
        with patch.object(tc, '_compute_patch_descriptor', return_value=np.zeros(10)):
            dists, k1, k2 = tc.compute()
            
            self.assertEqual(dists.shape, (1, 1))
            self.assertEqual(k1, [0])
            self.assertEqual(k2, [10])

if __name__ == '__main__':
    unittest.main()
