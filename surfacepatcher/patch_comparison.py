import pickle
import numpy as np
from surfacepatcher.utils import presets
from surfacepatcher.gpu_utils import GPUManager

class PatchComparison:
    def __init__(self, patches1, patches2, feature_weights=None, use_gpu=True):
        """
        Perform pairwise comparisons on all patches and their rotations using GPU acceleration.
        
        :param patches1: geodesic_patcher.geodesic_patcher.ProteinPatches for protein A
        :param patches2: geodesic_patcher.geodesic_patcher.ProteinPatches for protein B
        :param feature_weights: Optional custom weights for features (shape_index, mean_curvature, 
                               electrostatic, h_bond_donor, h_bond_acceptor, hydrophobicity)
        :param use_gpu: Whether to use GPU acceleration (default: True)
        """
        self.patches1 = patches1
        self.patches2 = patches2
        self.use_gpu = use_gpu
        
        # Initialize GPU manager and device
        if use_gpu:
            import torch
            self.gpu_manager = GPUManager(use_gpu=True)
            self.device = self.gpu_manager.device
            self.torch_available = True
        else:
            self.gpu_manager = None
            self.device = None
            self.torch_available = False

        #TODO need to get the weights from the static method
        # Set feature weights
        if feature_weights is not None:
            feature_weights_array = feature_weights
        else:
            # Epitope-optimized weights based on antibody-antigen binding importance
            # Order: shape_index, mean_curvature, electrostatic, h_bond_donor, h_bond_acceptor, hydrophobicity
            feature_weights_array = presets["epitope"]
        
        # Store as numpy array first, convert to tensor if using GPU
        self.feature_weights_np = np.array(feature_weights_array, dtype=np.float32)
        
        if use_gpu:
            import torch
            self.feature_weights = torch.tensor(self.feature_weights_np, dtype=torch.float32).to(self.device)
        else:
            self.feature_weights = None

    def compute(self, batch_size=100):
        """
        Compute pairwise distances between all patches in patches1 and patches2.
        For each pair (p1, p2), finds the minimum distance across all rotations of p2.
        Feature weights are applied to prioritize epitope-relevant properties.
        
        :param batch_size: Number of patches from patches1 to process at once.
        :return: (distances, indices, keys1, keys2)
                 distances: np.ndarray of shape (N1, N2) with min weighted distances
                 indices: np.ndarray of shape (N1, N2) with index of best rotation
                 keys1: list of keys for patches1 corresponding to rows
                 keys2: list of keys for patches2 corresponding to columns
        """
        if not self.torch_available:
            raise RuntimeError(
                "PyTorch is required for PatchComparison but is not available. "
                "Please install PyTorch: pip install torch"
            )
        
        import torch
        
        # 1. Prepare Data for Protein 1 (Query)
        # We only need one reference rotation (e.g., index 0) for P1
        keys1 = sorted(list(self.patches1.descriptors.keys()))
        # Shape: (N1, M_ang, K, P) -> Flatten to (N1, F)
        # descriptors[k] is (M_rot, M_ang, K, P). We take [0].
        list1 = [self.patches1.descriptors[k][0] for k in keys1]
        if not list1:
            return np.array([]), np.array([]), keys1, []
        
        # Stack: (N1, M_ang, K, P)
        tensor1 = torch.stack(list1).float().to(self.device)
        N1, M_ang, K, P = tensor1.shape
        
        # Flatten first: (N1, M_ang * K * P)
        # Structure: each row is [all M_ang*K points for feature_0, all M_ang*K points for feature_1, ...]
        tensor1_flat = tensor1.view(N1, -1)
        
        # 2. Prepare Data for Protein 2 (Target)
        # We keep all rotations to search against
        keys2 = sorted(list(self.patches2.descriptors.keys()))
        # Shape: (N2, M_rot, M_ang, K, P) -> Flatten to (N2 * M_rot, F)
        list2 = [self.patches2.descriptors[k] for k in keys2]
        if not list2:
            return np.array([]), np.array([]), keys1, keys2
            
        tensor2 = torch.stack(list2).float().to(self.device)
        N2, M_rot, M_ang2, K2, P2 = tensor2.shape
        
        # Flatten: (N2 * M_rot, M_ang * K * P)
        F = tensor1_flat.shape[1]
        tensor2_flat = tensor2.view(-1, F)
        
        # --- Feature Standardization ---
        # Reshape to (Total_Points, P) to compute stats per feature
        # Combine samples from both proteins for robust statistics
        # tensor1: (N1, M_ang, K, P)
        # tensor2: (N2, M_rot, M_ang, K, P)
        
        t1_for_stats = tensor1.view(-1, P)
        t2_for_stats = tensor2.view(-1, P)
        all_features = torch.cat([t1_for_stats, t2_for_stats], dim=0)
        
        # Compute mean and std per feature
        feat_mean = all_features.mean(dim=0)
        feat_std = all_features.std(dim=0)
        
        # Avoid division by zero
        feat_std[feat_std < 1e-6] = 1.0
        
        # Normalize tensors
        # tensor1_flat: (N1, M_ang*K*P) -> View as (N1, M_ang*K, P)
        points_per_sample = M_ang * K
        
        t1_normalized = (tensor1.view(-1, P) - feat_mean) / feat_std
        tensor1_flat = t1_normalized.view(N1, -1)
        
        t2_normalized = (tensor2.view(-1, P) - feat_mean) / feat_std
        tensor2_flat = t2_normalized.view(-1, F)
        
        # -------------------------------
        
        # Apply feature weights to compute weighted norm per feature channel
        # Reshape to separate features: (N, M_ang*K, P)
        points_per_feature = M_ang * K
        tensor1_by_feat = tensor1_flat.view(N1, points_per_feature, P)
        tensor2_by_feat = tensor2_flat.view(-1, points_per_feature, P2)
        
        # Weight each feature channel and compute norms
        # For each sample, compute ||feat_i * weight_i||^2 for each feature i
        weighted_norm1_sq = torch.zeros(N1, device=self.device)
        weighted_norm2_sq = torch.zeros(tensor2_flat.shape[0], device=self.device)
        
        for feat_idx in range(P):
            feat1 = tensor1_by_feat[:, :, feat_idx]  # (N1, points_per_feature)
            feat2 = tensor2_by_feat[:, :, feat_idx]  # (N2*M_rot, points_per_feature)
            weight = self.feature_weights[feat_idx]
            
            weighted_norm1_sq += weight * weight * (feat1 ** 2).sum(dim=1)
            weighted_norm2_sq += weight * weight * (feat2 ** 2).sum(dim=1)
        
        
        # 3. Compute Distances in Batches with feature weighting
        # ||A - B||^2 with weights = sum_i w_i^2 * ||A_i - B_i||^2
        # = sum_i w_i^2 * (||A_i||^2 + ||B_i||^2 - 2<A_i, B_i>)
        
        all_min_dists = []
        all_min_idxs = []
        
        # Process P1 in batches
        for i in range(0, N1, batch_size):
            end = min(i + batch_size, N1)
            batch1_by_feat = tensor1_by_feat[i:end]  # (B, points_per_feature, P)
            batch_norm1 = weighted_norm1_sq[i:end].unsqueeze(1)  # (B, 1)
            
            # Compute weighted dot product across all features
            # For each sample pair, compute sum_i w_i^2 * <A_i, B_i>
            weighted_dot = torch.zeros(end - i, tensor2_flat.shape[0], device=self.device)
            
            for feat_idx in range(P):
                feat1_batch = batch1_by_feat[:, :, feat_idx]  # (B, points_per_feature)
                feat2_all = tensor2_by_feat[:, :, feat_idx]    # (N2*M_rot, points_per_feature)
                weight = self.feature_weights[feat_idx]
                
                # Dot product for this feature
                dot_feat = torch.matmul(feat1_batch, feat2_all.T)  # (B, N2*M_rot)
                weighted_dot += weight * weight * dot_feat
            
            # Distance squared: ||A||^2 + ||B||^2 - 2<A, B>
            dist_sq = batch_norm1 + weighted_norm2_sq.unsqueeze(0) - 2 * weighted_dot
            
            # Clamp to 0 to avoid negative due to precision
            dist_sq = torch.clamp(dist_sq, min=0.0)
            
            # Reshape to separate rotations: (B, N2, M_rot)
            dist_sq = dist_sq.view(-1, N2, M_rot)
            
            # Find min over rotations
            min_dist_sq, min_rot_idx = torch.min(dist_sq, dim=2)
            
            # Sqrt
            min_dist = torch.sqrt(min_dist_sq)
            
            all_min_dists.append(min_dist.cpu().numpy())
            all_min_idxs.append(min_rot_idx.cpu().numpy())
            
        # Concatenate results
        final_dists = np.concatenate(all_min_dists, axis=0) # (N1, N2)
        final_idxs = np.concatenate(all_min_idxs, axis=0)   # (N1, N2)
        
        return final_dists, final_idxs, keys1, keys2

    def save_distances(self, path, results):
        """
        Save the comparison results.
        :param path: Output file path
        :param results: Tuple (distances, indices, keys1, keys2)
        """
        dists, idxs, k1, k2 = results
        data = {
            "distances": dists,
            "indices": idxs,
            "keys1": k1,
            "keys2": k2,
            "feature_weights": self.feature_weights.cpu().numpy() if self.torch_available else self.feature_weights_np
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

        return None




