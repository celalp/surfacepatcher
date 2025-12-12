import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple
from geomloss import SamplesLoss

class ProteinPatchComparator:
    """
    Calculates pairwise similarity between a large collection of ProteinPatch objects
    using a two-stage filtering process:
    1. Coarse Filter: L2 distance on mean-pooled FPFH features.
    2. Fine Comparison: Sinkhorn-Wasserstein distance on full FPFH distributions.

    Assumes a CUDA-enabled GPU is available and PyTorch is configured.
    """

    def __init__(self, fpfh_dim: int = 33, coarse_threshold: float = 0.5,
                 sinkhorn_blur: float = 0.05, sinkhorn_reach: float = None):
        """
        Initializes the comparator with distance parameters.

        :param fpfh_dim: Dimension of the FPFH feature vector (default: 33).
        :param coarse_threshold: Max L2 distance for a pair to pass the coarse filter.
        :param sinkhorn_blur: The entropy regularization parameter (lambda) for Sinkhorn.
        :param sinkhorn_reach: Optional parameter for unbalanced OT (not used here).
        :param device: The PyTorch device ('cuda' or 'cpu'). Defaults to 'cuda' if available.
        """
        self.fpfh_dim = fpfh_dim
        self.coarse_threshold = coarse_threshold

        # 1. Initialize the Sinkhorn-Wasserstein loss (for fine comparison)
        # The 'p=2' specifies the 2-Wasserstein distance (Earth Mover's Distance)
        # 'reduction='none'' returns the distance for each pair.
        self.sinkhorn_loss_fn = SamplesLoss(
            loss="sinkhorn", p=2, blur=sinkhorn_blur, reach=sinkhorn_reach,
            backend="tensorized", potentials=False, verbose=False
        )

        # 2. Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _get_global_features(self, patches: List['ProteinPatch']) -> torch.Tensor:
        """
        Calculates the mean-pooled global FPFH feature vector for each patch.
        (Stage 1 preparation).
        """
        global_features = []
        for patch in patches:
            # Convert FPFH to a torch tensor and move to device
            fpfh_tensor = torch.from_numpy(patch.fpfh_features).float().to(self.device)
            # Compute the mean (centroid) of the distribution
            mean_fpfh = fpfh_tensor.mean(dim=0)
            global_features.append(mean_fpfh)

        # Stack into a single tensor (N_patches, fpfh_dim)
        return torch.stack(global_features)

    def coarse_filter(self, global_features: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Stage 1: Computes the pairwise L2 distance between global features
        and filters pairs based on the coarse_threshold.
        """
        N = global_features.size(0)

        # Compute the pairwise squared Euclidean distance (L2 distance squared)
        # Using torch.cdist and squaring is cleaner and often faster than manual broadcasting
        dist_matrix = torch.cdist(global_features, global_features, p=2)

        # Identify pairs that pass the filter
        # We only need the upper triangle (i < j) for unique pairs
        passing_indices = (dist_matrix < self.coarse_threshold).triu(diagonal=1).nonzero(as_tuple=False)

        # Convert tensor indices to a list of (i, j) tuples for easy lookup
        passing_pairs = [(i.item(), j.item()) for i, j in passing_indices]

        return dist_matrix, passing_pairs

    def fine_comparison(self, patches: List['ProteinPatch'],
                        passing_pairs: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Stage 2: Computes the Sinkhorn-Wasserstein distance for the
        filtered pairs.
        """

        if not passing_pairs:
            # Return an empty tensor if no pairs passed the filter
            return torch.empty(0, 3)

            # We need to map the (i, j) indices to the final distance result
        results = []

        # Process pairs sequentially (for simplicity and memory management)
        for idx_i, idx_j in passing_pairs:
            patch_i = patches[idx_i]
            patch_j = patches[idx_j]

            # Convert FPFH descriptors to PyTorch tensors (N_points, 33)
            fpfh_i = torch.from_numpy(patch_i.fpfh_features).float().to(self.device)
            fpfh_j = torch.from_numpy(patch_j.fpfh_features).float().to(self.device)

            # The SamplesLoss expects the distributions/measures, which are the
            # feature tensors themselves. Weights (masses) are implicitly assumed
            # to be uniform (1/N) if not provided. We rely on uniform mass here.

            # Note: The output is W_2^2. We take the square root for W_2.
            # GeomLoss SampleLoss returns W_p^p by default, so we use the final
            # power=1 parameter to adjust for W_2.
            # *Self-correction: W_p(., .) is the distance. The function computes it.*
            # *The parameter p=2 means we compute the L2 norm on the ground cost.*

            # Distance is W_2(FPFH_i, FPFH_j)
            distance_W2 = self.sinkhorn_loss_fn(fpfh_i, fpfh_j)

            # Append (index_i, index_j, W_2 distance)
            results.append((idx_i, idx_j, distance_W2.item()))

        # Convert the results list to a (N_pairs, 3) tensor
        return torch.tensor(results, dtype=torch.float32)

    def compute_similarity_matrix(self, patches: List['ProteinPatch']) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main function to execute the two-stage similarity computation.

        :param patches: A list of ProteinPatch objects.
        :return: A tuple: (coarse_distance_matrix, fine_comparison_results)
                 - coarse_distance_matrix: (N, N) matrix of L2 distances on global features.
                 - fine_comparison_results: (N_filtered_pairs, 3) tensor of (i, j, W2_dist).
        """

        if not patches:
            return torch.empty(0), torch.empty(0)

        # --- Stage 1: Coarse Filtering ---
        # 1.1 Compute global features
        global_features = self._get_global_features(patches)

        # 1.2 Compute L2 matrix and filter pairs
        coarse_matrix, passing_pairs = self.coarse_filter(global_features)

        # --- Stage 2: Fine Comparison (GPU-accelerated Sinkhorn) ---
        fine_results = self.fine_comparison(patches, passing_pairs)

        return coarse_matrix.cpu(), fine_results.cpu()

