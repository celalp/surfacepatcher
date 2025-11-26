import pickle
import torch
import numpy as np
from tqdm import tqdm

class PatchComparison:
    def __init__(self, patches1, patches2):
        """
        Perform pairwise comparisons on all patches and their rotations using GPU acceleration.
        :param patches1: surfacepatcher.surfacepatcher.ProteinPatches for protein A
        :param patches2: surfacepatcher.surfacepatcher.ProteinPatches for protein B
        """
        self.patches1 = patches1
        self.patches2 = patches2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute(self, batch_size=100):
        """
        Compute pairwise distances between all patches in patches1 and patches2.
        For each pair (p1, p2), finds the minimum distance across all rotations of p2.
        
        :param batch_size: Number of patches from patches1 to process at once.
        :return: (distances, indices, keys1, keys2)
                 distances: np.ndarray of shape (N1, N2) with min distances
                 indices: np.ndarray of shape (N1, N2) with index of best rotation
                 keys1: list of keys for patches1 corresponding to rows
                 keys2: list of keys for patches2 corresponding to columns
        """
        # 1. Prepare Data for Protein 1 (Query)
        # We only need one reference rotation (e.g., index 0) for P1
        keys1 = sorted(list(self.patches1.descriptors.keys()))
        # Shape: (N1, M_ang, K, P) -> Flatten to (N1, F)
        # descriptors[k] is (M_rot, M_ang, K, P). We take [0].
        list1 = [self.patches1.descriptors[k][0] for k in keys1]
        if not list1:
            return np.array([]), np.array([]), keys1, []
        
        # Stack and flatten: (N1, M_ang * K * P)
        tensor1 = torch.stack(list1).float().to(self.device)
        N1 = tensor1.shape[0]
        tensor1 = tensor1.view(N1, -1) 
        
        # 2. Prepare Data for Protein 2 (Target)
        # We keep all rotations to search against
        keys2 = sorted(list(self.patches2.descriptors.keys()))
        # Shape: (N2, M_rot, M_ang, K, P) -> Flatten to (N2 * M_rot, F)
        list2 = [self.patches2.descriptors[k] for k in keys2]
        if not list2:
            return np.array([]), np.array([]), keys1, keys2
            
        tensor2 = torch.stack(list2).float().to(self.device)
        N2, M_rot = tensor2.shape[0], tensor2.shape[1]
        F = tensor1.shape[1]
        
        # Flatten: (N2 * M_rot, F)
        tensor2_flat = tensor2.view(-1, F)
        
        # 3. Compute Distances in Batches
        # ||A - B||^2 = ||A||^2 + ||B||^2 - 2<A, B>
        
        # Precompute norms
        norm1_sq = (tensor1 ** 2).sum(dim=1, keepdim=True) # (N1, 1)
        norm2_sq = (tensor2_flat ** 2).sum(dim=1) # (N2 * M_rot)
        
        all_min_dists = []
        all_min_idxs = []
        
        # Process P1 in batches
        for i in tqdm(range(0, N1, batch_size), desc="Computing distances"):
            end = min(i + batch_size, N1)
            batch1 = tensor1[i:end] # (B, F)
            batch_norm1 = norm1_sq[i:end] # (B, 1)
            
            # Dot product: (B, F) @ (F, N2*M_rot) -> (B, N2*M_rot)
            dot = torch.matmul(batch1, tensor2_flat.T)
            
            # Distance squared: (B, 1) + (N2*M_rot) - (B, N2*M_rot)
            # Broadcasting works: (B, 1) + (1, N2*M_rot)
            dist_sq = batch_norm1 + norm2_sq.unsqueeze(0) - 2 * dot
            
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
            "keys2": k2
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

