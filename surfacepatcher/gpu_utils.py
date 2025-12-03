import numpy as np
from typing import Optional, Union


#TODO get rid of all the printing and assume GPU is available if use_gpu is True
class GPUManager:
    """Manages GPU device selection."""
    
    def __init__(self, use_gpu=True):
        """
        Initialize GPU manager.
        
        :param use_gpu: Whether to use GPU (assumes GPU is available if True)
        """
        self.use_gpu = use_gpu
        
        if use_gpu:
            try:
                import torch
                self.device = torch.device("cuda")
                self.torch_available = True
            except ImportError:
                raise ImportError("PyTorch is required for GPU acceleration. Install with: pip install torch")
        else:
            self.device = None
            self.torch_available = False


def compute_pairwise_distances_gpu(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = 'euclidean',
    device: Optional[str] = None,
    batch_size: int = 1000
) -> np.ndarray:
    """
    Compute pairwise distances between two sets of vectors using GPU.
    
    :param X: First set of vectors (N, D)
    :param Y: Second set of vectors (M, D)
    :param metric: Distance metric ('euclidean', 'cosine', 'chi2')
    :param device: PyTorch device to use
    :param batch_size: Batch size for large computations
    :return: Distance matrix (N, M)
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for GPU acceleration")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to tensors
    X_tensor = torch.from_numpy(X).float().to(device)
    Y_tensor = torch.from_numpy(Y).float().to(device)
    
    N, D = X_tensor.shape
    M = Y_tensor.shape[0]
    
    if metric == 'euclidean':
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
        X_norm_sq = (X_tensor ** 2).sum(dim=1, keepdim=True)  # (N, 1)
        Y_norm_sq = (Y_tensor ** 2).sum(dim=1)  # (M,)
        
        # Process in batches to avoid memory issues
        distances = []
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            X_batch = X_tensor[i:end]
            X_batch_norm = X_norm_sq[i:end]
            
            # Dot product
            dot = torch.mm(X_batch, Y_tensor.T)  # (batch, M)
            
            # Distance squared
            dist_sq = X_batch_norm + Y_norm_sq.unsqueeze(0) - 2 * dot
            dist_sq = torch.clamp(dist_sq, min=0.0)
            
            # Take square root
            dist = torch.sqrt(dist_sq)
            distances.append(dist.cpu().numpy())
        
        return np.concatenate(distances, axis=0)
    
    # there must be built in way to calculate cosine distance
    elif metric == 'cosine':
        # Normalize vectors
        X_norm = X_tensor / (torch.norm(X_tensor, dim=1, keepdim=True) + 1e-10)
        Y_norm = Y_tensor / (torch.norm(Y_tensor, dim=1, keepdim=True) + 1e-10)
        
        # Cosine similarity
        distances = []
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            X_batch = X_norm[i:end]
            
            # Cosine similarity
            similarity = torch.mm(X_batch, Y_norm.T)
            
            # Convert to distance (1 - similarity)
            dist = 1.0 - similarity
            distances.append(dist.cpu().numpy())
        
        return np.concatenate(distances, axis=0)
    
    elif metric == 'chi2':
        # Chi-square distance for histogram comparison
        # Ensure non-negative and normalize
        X_pos = torch.abs(X_tensor)
        Y_pos = torch.abs(Y_tensor)
        
        X_norm = X_pos / (X_pos.sum(dim=1, keepdim=True) + 1e-10)
        Y_norm = Y_pos / (Y_pos.sum(dim=1, keepdim=True) + 1e-10)
        
        distances = []
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            X_batch = X_norm[i:end]  # (batch, D)
            
            # Expand for broadcasting: (batch, 1, D) and (1, M, D)
            X_expanded = X_batch.unsqueeze(1)  # (batch, 1, D)
            Y_expanded = Y_norm.unsqueeze(0)  # (1, M, D)
            
            # Chi-square distance: sum of (x - y)^2 / (x + y)
            numerator = (X_expanded - Y_expanded) ** 2
            denominator = X_expanded + Y_expanded + 1e-10
            chi2_dist = (numerator / denominator).sum(dim=2)
            
            distances.append(chi2_dist.cpu().numpy())
        
        return np.concatenate(distances, axis=0)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def aggregate_descriptors_gpu(
    descriptors: np.ndarray,
    device: Optional[str] = None
) -> np.ndarray:
    """
    Aggregate point-wise descriptors to patch-level using GPU.
    Computes mean, std, max, min statistics.
    
    :param descriptors: Point-wise descriptors (N, D)
    :param device: PyTorch device to use
    :return: Aggregated descriptor (4*D,)
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for GPU acceleration")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to tensor
    desc_tensor = torch.from_numpy(descriptors).float().to(device)
    
    # Compute statistics
    desc_mean = torch.mean(desc_tensor, dim=0)
    desc_std = torch.std(desc_tensor, dim=0)
    desc_max = torch.max(desc_tensor, dim=0)[0]
    desc_min = torch.min(desc_tensor, dim=0)[0]
    
    # Concatenate and return
    aggregated = torch.cat([desc_mean, desc_std, desc_max, desc_min])
    
    return aggregated.cpu().numpy()


def batch_process_gpu(
    data: np.ndarray,
    process_fn,
    batch_size: int = 100,
    device: Optional[str] = None
) -> np.ndarray:
    """
    Process data in batches on GPU to avoid memory issues.
    
    :param data: Input data (N, ...)
    :param process_fn: Function to apply to each batch (takes tensor, returns tensor)
    :param batch_size: Size of each batch
    :param device: PyTorch device to use
    :return: Processed data
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for GPU acceleration")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    N = len(data)
    results = []
    
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        batch = torch.from_numpy(data[i:end]).float().to(device)
        
        # Process batch
        result = process_fn(batch)
        
        # Store result
        results.append(result.cpu().numpy())
    
    return np.concatenate(results, axis=0)


def normalize_matrix_gpu(
    matrix: np.ndarray,
    device: Optional[str] = None
) -> np.ndarray:
    """
    Normalize matrix to [0, 1] range using GPU (min-max normalization).
    
    :param matrix: Input matrix
    :param device: PyTorch device to use
    :return: Normalized matrix
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for GPU acceleration")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to tensor
    matrix_tensor = torch.from_numpy(matrix).float().to(device)
    
    # Normalize
    min_val = matrix_tensor.min()
    max_val = matrix_tensor.max()
    
    if max_val - min_val < 1e-10:
        return torch.zeros_like(matrix_tensor).cpu().numpy()
    
    normalized = (matrix_tensor - min_val) / (max_val - min_val)
    
    return normalized.cpu().numpy()


def rank_matrix_gpu(
    matrix: np.ndarray,
    device: Optional[str] = None
) -> np.ndarray:
    """
    Convert matrix values to ranks using GPU.
    
    :param matrix: Input matrix
    :param device: PyTorch device to use
    :return: Rank matrix normalized to [0, 1]
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for GPU acceleration")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to tensor
    matrix_tensor = torch.from_numpy(matrix).float().to(device)
    
    # Flatten
    flat = matrix_tensor.flatten()
    
    # Compute ranks (argsort twice trick)
    ranks = torch.argsort(torch.argsort(flat)).float()
    
    # Reshape and normalize
    ranks = ranks.reshape(matrix_tensor.shape)
    ranks_normalized = ranks / (len(flat) - 1) if len(flat) > 1 else ranks
    
    return ranks_normalized.cpu().numpy()


def compute_feature_distances_gpu(
    points: np.ndarray,
    feature_values: np.ndarray,
    device: Optional[str] = None,
    feature_weight: float = 2.0,
    skip_normalization: bool = False
) -> np.ndarray:
    """
    Compute combined geometric and feature-weighted distance matrix on GPU.
    Used for feature-weighted persistence in topological comparison.
    
    :param points: Point coordinates (N, 3)
    :param feature_values: Feature values at each point (N,)
    :param device: PyTorch device to use
    :param feature_weight: Weight for feature distance contribution (default: 2.0)
    :param skip_normalization: If True, assume feature_values are already normalized [0,1]
    :return: Combined distance matrix (N, N)
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for GPU acceleration")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to tensors
    points_tensor = torch.from_numpy(points).float().to(device)
    feat_values_tensor = torch.from_numpy(feature_values).float().to(device)
    
    # Geometric distances (Euclidean)
    # (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = points_tensor.unsqueeze(1) - points_tensor.unsqueeze(0)
    geom_dists = torch.sqrt(torch.sum(diff ** 2, dim=2))
    
    # Normalize feature values
    if skip_normalization:
        feat_norm = feat_values_tensor
    else:
        feat_min = feat_values_tensor.min()
        feat_max = feat_values_tensor.max()
        feat_norm = (feat_values_tensor - feat_min) / (feat_max - feat_min + 1e-10)
    
    # Feature distances
    feat_dists = torch.abs(feat_norm.unsqueeze(1) - feat_norm.unsqueeze(0))
    
    # Combined distance with configurable weight
    combined_dists = geom_dists + feature_weight * feat_dists
    
    return combined_dists.cpu().numpy()
