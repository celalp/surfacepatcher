import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata


class HybridComparison:
    def __init__(self, geodesic_results=None, pointcloud_results=None, topological_results=None,
                 weights=None, fusion_method='weighted_sum'):
        """
        Combine multiple surface comparison methods into a unified similarity metric.
        
        :param geodesic_results: Tuple (distances, indices, keys1, keys2) from PatchComparison
        :param pointcloud_results: Tuple (distances, keys1, keys2) from PointCloudComparison
        :param topological_results: Tuple (distances, keys1, keys2) from TopologicalComparison
        :param weights: Dict with keys 'geodesic', 'pointcloud', 'topological' (default: equal weights)
        :param fusion_method: 'weighted_sum', 'rank_fusion', or 'product'
        """
        self.geodesic_results = geodesic_results
        self.pointcloud_results = pointcloud_results
        self.topological_results = topological_results
        self.fusion_method = fusion_method
        
        # Set default weights
        if weights is None:
            weights = {}
        self.weights = {
            'geodesic': weights.get('geodesic', 1.0),
            'pointcloud': weights.get('pointcloud', 1.0),
            'topological': weights.get('topological', 1.0)
        }
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
    @staticmethod
    def get_preset_weights(preset='balanced'):
        """
        Get preset weight configurations for different use cases.
        
        :param preset: One of 'balanced', 'epitope', 'shape_focused', 'topology_focused'
        :return: Dict of weights
        """
        presets = {
            'balanced': {
                'geodesic': 1.0,
                'pointcloud': 1.0,
                'topological': 1.0
            },
            'epitope': {
                'geodesic': 1.5,  # High-resolution features important
                'pointcloud': 1.0,  # Geometric shape
                'topological': 1.3  # Cavity/protrusion patterns critical
            },
            'shape_focused': {
                'geodesic': 1.0,
                'pointcloud': 2.0,  # Emphasize geometric descriptors
                'topological': 0.5
            },
            'topology_focused': {
                'geodesic': 0.8,
                'pointcloud': 0.5,
                'topological': 2.0  # Emphasize topological features
            }
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(presets.keys())}")
        
        return presets[preset]
    
    def _normalize_distances(self, distances):
        """
        Normalize distance matrix to [0, 1] range using min-max scaling.
        
        :param distances: (N1, N2) distance matrix
        :return: Normalized distances
        """
        min_val = np.min(distances)
        max_val = np.max(distances)
        
        if max_val - min_val < 1e-10:
            return np.zeros_like(distances)
        
        return (distances - min_val) / (max_val - min_val)
    
    def _rank_distances(self, distances):
        """
        Convert distances to ranks (lower rank = more similar).
        
        :param distances: (N1, N2) distance matrix
        :return: Rank matrix
        """
        # Flatten, rank, reshape
        flat_dists = distances.flatten()
        ranks = rankdata(flat_dists, method='average')
        rank_matrix = ranks.reshape(distances.shape)
        
        # Normalize ranks to [0, 1]
        return (rank_matrix - 1) / (len(flat_dists) - 1)
    
    def compute(self):
        """
        Compute hybrid distance matrix combining all available methods.
        
        :return: (combined_distances, keys1, keys2)
        """
        available_methods = []
        distance_matrices = []
        method_weights = []
        
        # Collect available results
        if self.geodesic_results is not None:
            dists, indices, keys1, keys2 = self.geodesic_results
            available_methods.append('geodesic')
            distance_matrices.append(dists)
            method_weights.append(self.weights['geodesic'])
        
        if self.pointcloud_results is not None:
            dists, k1, k2 = self.pointcloud_results
            if 'keys1' not in locals():
                keys1, keys2 = k1, k2
            available_methods.append('pointcloud')
            distance_matrices.append(dists)
            method_weights.append(self.weights['pointcloud'])
        
        if self.topological_results is not None:
            dists, k1, k2 = self.topological_results
            if 'keys1' not in locals():
                keys1, keys2 = k1, k2
            available_methods.append('topological')
            distance_matrices.append(dists)
            method_weights.append(self.weights['topological'])
        
        if len(available_methods) == 0:
            raise ValueError("No comparison results provided")
        
        # Normalize weights for available methods
        total_weight = sum(method_weights)
        method_weights = [w / total_weight for w in method_weights]
        
        # Fusion
        if self.fusion_method == 'weighted_sum':
            # Normalize each distance matrix and combine
            normalized_matrices = [self._normalize_distances(d) for d in distance_matrices]
            combined = sum(w * d for w, d in zip(method_weights, normalized_matrices))
            
        elif self.fusion_method == 'rank_fusion':
            # Rank-based fusion (Borda count style)
            rank_matrices = [self._rank_distances(d) for d in distance_matrices]
            combined = sum(w * r for w, r in zip(method_weights, rank_matrices))
            
        elif self.fusion_method == 'product':
            # Multiplicative fusion (geometric mean)
            normalized_matrices = [self._normalize_distances(d) + 1e-10 for d in distance_matrices]
            # Weighted geometric mean
            log_sum = sum(w * np.log(d) for w, d in zip(method_weights, normalized_matrices))
            combined = np.exp(log_sum)
            
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")
        
        return combined, keys1, keys2
    
    def get_method_agreement(self):
        """
        Compute correlation between different methods to assess agreement.
        
        :return: Dict of pairwise correlations
        """
        methods = {}
        
        if self.geodesic_results is not None:
            methods['geodesic'] = self.geodesic_results[0].flatten()
        
        if self.pointcloud_results is not None:
            methods['pointcloud'] = self.pointcloud_results[0].flatten()
        
        if self.topological_results is not None:
            methods['topological'] = self.topological_results[0].flatten()
        
        # Compute pairwise correlations
        correlations = {}
        method_names = list(methods.keys())
        
        for i, name1 in enumerate(method_names):
            for name2 in method_names[i+1:]:
                corr = np.corrcoef(methods[name1], methods[name2])[0, 1]
                correlations[f"{name1}_vs_{name2}"] = corr
        
        return correlations
    
    def get_top_matches(self, n=10, method='hybrid'):
        """
        Get top N matches across all patch pairs.
        
        :param n: Number of top matches to return
        :param method: 'hybrid', 'geodesic', 'pointcloud', or 'topological'
        :return: List of (patch1_idx, patch2_idx, distance) tuples
        """
        if method == 'hybrid':
            distances, keys1, keys2 = self.compute()
        elif method == 'geodesic' and self.geodesic_results is not None:
            distances = self.geodesic_results[0]
            keys1, keys2 = self.geodesic_results[2], self.geodesic_results[3]
        elif method == 'pointcloud' and self.pointcloud_results is not None:
            distances, keys1, keys2 = self.pointcloud_results
        elif method == 'topological' and self.topological_results is not None:
            distances, keys1, keys2 = self.topological_results
        else:
            raise ValueError(f"Method '{method}' not available")
        
        # Find top N matches
        flat_dists = distances.flatten()
        flat_indices = np.argsort(flat_dists)[:n]
        
        matches = []
        for idx in flat_indices:
            i = idx // len(keys2)
            j = idx % len(keys2)
            matches.append((keys1[i], keys2[j], distances[i, j]))
        
        return matches

    #TODO add pickling
    
