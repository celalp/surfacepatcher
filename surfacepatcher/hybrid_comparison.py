import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
from surfacepatcher.gpu_utils import GPUManager

class HybridComparison:
    def __init__(self, geodesic_results=None, pointcloud_results=None, topological_results=None,
                 weights=None, fusion_method='weighted_sum', use_gpu=True):
        """
        Combine multiple surface comparison methods into a unified similarity metric.
        
        :param geodesic_results: Tuple (distances, indices, keys1, keys2) from PatchComparison
        :param pointcloud_results: Tuple (distances, keys1, keys2) from PointCloudComparison
        :param topological_results: Tuple (distances, keys1, keys2) from TopologicalComparison
        :param weights: Dict with keys 'geodesic', 'pointcloud', 'topological' (default: equal weights)
        :param fusion_method: 'weighted_sum', 'rank_fusion', or 'product'
        :param use_gpu: Whether to use GPU acceleration for fusion operations (default: True)
        """
        self.geodesic_results = geodesic_results
        self.pointcloud_results = pointcloud_results
        self.topological_results = topological_results
        self.fusion_method = fusion_method
        self.use_gpu = use_gpu
        
        # Initialize GPU manager

        self.gpu_manager = GPUManager(use_gpu=use_gpu) if use_gpu else None
        
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
        
        :param preset: One of 'balanced', 'epitope', 'shape_focused', 'topology_focused' I completely made up these
        numbers in the sense that I am not sure what epiptote centric would look like. I figured it has more to do with shape and 
        electrostatics/hydrophobicity than anything else. 
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
        Uses GPU acceleration when enabled.
        
        :param distances: (N1, N2) distance matrix
        :return: Normalized distances
        """
        if self.use_gpu and self.gpu_manager is not None:
            from surfacepatcher.gpu_utils import normalize_matrix_gpu
            return normalize_matrix_gpu(distances, device=self.gpu_manager.device)
        
        # CPU path
        min_val = np.min(distances)
        max_val = np.max(distances)
        
        if max_val - min_val < 1e-10:
            return np.zeros_like(distances)
        
        return (distances - min_val) / (max_val - min_val)
    
    def _rank_distances(self, distances):
        """
        Convert distances to ranks (lower rank = more similar).
        Uses GPU acceleration when enabled.
        
        :param distances: (N1, N2) distance matrix
        :return: Rank matrix
        """
        if self.use_gpu and self.gpu_manager is not None:
            from surfacepatcher.gpu_utils import rank_matrix_gpu
            return rank_matrix_gpu(distances, device=self.gpu_manager.device)
        
        # CPU path
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
                 combined_distances: np.ndarray of shape (N1, N2) with fused distances
                 keys1: list of keys for patches1 corresponding to rows
                 keys2: list of keys for patches2 corresponding to columns
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
            
        elif self.fusion_method == 'reciprocal_rank_fusion':
            # Reciprocal rank fusion - better for preserving magnitude information
            # Convert distances to reciprocal ranks and weight by magnitude
            combined = np.zeros_like(distance_matrices[0])
            for d, w in zip(distance_matrices, method_weights):
                # Rank distances (lower rank = smaller distance = more similar)
                flat_dists = d.flatten()
                ranks = np.argsort(np.argsort(flat_dists)) + 1  # Ranks start from 1
                rank_matrix = ranks.reshape(d.shape).astype(np.float32)
                
                # Reciprocal rank with weight
                reciprocal_scores = w / (rank_matrix + 60)  # k=60 is common for RRF
                combined += reciprocal_scores
            
            # Normalize to distance (invert scores)
            combined = 1.0 / (combined + 1e-10)
            
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
    
    def save_distances(self, path, results):
        """
        Save the hybrid comparison results.
        
        :param path: Output file path
        :param results: Tuple (distances, keys1, keys2) from compare()
        """
        dists, k1, k2 = results
        data = {
            "distances": dists,
            "keys1": k1,
            "keys2": k2,
            "weights": self.weights,
            "fusion_method": self.fusion_method,
            "available_methods": []
        }
        
        # Record which methods were used
        if self.geodesic_results is not None:
            data["available_methods"].append("geodesic")
        if self.pointcloud_results is not None:
            data["available_methods"].append("pointcloud")
        if self.topological_results is not None:
            data["available_methods"].append("topological")
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
