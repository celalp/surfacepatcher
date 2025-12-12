import os
import pickle
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any

from surfacepatcher.geodesic_patcher import ProteinPatch, ProteinPatches

# Try imports for optional acceleration
try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None
    _FAISS_AVAILABLE = False

try:
    from pytorch3d.loss import chamfer_distance as pyt3d_chamfer
    _PYT3D_AVAILABLE = True
except Exception:
    pyt3d_chamfer = None
    _PYT3D_AVAILABLE = False

class PatchAccessor:
    """
    Provides unified access to ProteinPatch objects, regardless of whether they
    come from memory or shelve-backed ProteinPatches containers.

    Provides:
      - get_ids()
      - get_patch(idx)
      - get_points(idx)      → (N,3) ndarray
      - get_fpfh(idx)        → 1D ndarray
      - get_biochem(idx)     → dict
    """

    def __init__(self, container: ProteinPatches):
        self.container = container
        self.ids = sorted(container.get_all_patch_ids(container.shelve_path)
                          if container.patches == {} else
                          list(container.patches.keys()))
        self.n = len(self.ids)

    def __len__(self):
        return self.n

    def get_ids(self):
        return self.ids

    def get_patch(self, i: int) -> ProteinPatch:
        """i is an index into ids, NOT a patch_id."""
        return self.container[self.ids[i]]

    def get_points(self, i: int) -> np.ndarray:
        p = self.get_patch(i)
        return p.vertices.astype(np.float32)

    def get_normals(self, i: int) -> np.ndarray:
        p = self.get_patch(i)
        return p.normals.astype(np.float32)

    def get_fpfh(self, i: int) -> np.ndarray:
        p = self.get_patch(i)
        # p.fpfh_features is (D, K) column-major for Open3D.
        # FAISS needs (D,) or (1,D). Take mean or flatten.
        # Typical FPFH in O3D is shape (33, N)
        return p.fpfh_features.mean(axis=1).astype(np.float32)

    def get_biochem(self, i: int) -> dict:
        return self.get_patch(i).biochem_features or {}




class FAISS_PyTorch3D_Comparator:
    """
    GPU-accelerated comparator using FAISS for candidate selection (on GPU if available)
    and PyTorch3D (or torch.cdist fallback) for Chamfer distance + SVD Procrustes alignment.

    Required inputs:
      - patches1, patches2: lists of patch objects (same as before)
      - fpfh1, fpfh2: numpy arrays of precomputed descriptors for each patch:
            fpfh1.shape == (len(patches1), D)
            fpfh2.shape == (len(patches2), D)

    Main parameters:
      - k_candidates: FAISS top-k neighbors to consider for each patch in patches1
      - num_points: number of points sampled/padded per patch for alignment/chamfer
      - device: "cuda" or "cpu"
      - use_faiss_gpu: attempt to use FAISS GPU if available
      - use_pytorch3d: attempt to use pytorch3d chamfer if available (falls back to torch.cdist)
      - pair_block_size: how many candidate pairs to process per GPU batch
    """

    def __init__(
        self,
        patches1: List,
        patches2: List,
        device: str = "cuda",
        k_candidates: int = 50,
        num_points: int = 512,
        feat_sample_len: int = 128,
        pair_block_size: int = 128,
        use_faiss_gpu: bool = True,
        use_pytorch3d: bool = True,
        seed: Optional[int] = None,
    ):
        self.p1 = PatchAccessor(patches1)
        self.p2 = PatchAccessor(patches2)

        # Build FAISS feature matrices
        self.fpfh1 = np.vstack([self.p1.get_fpfh(i) for i in range(len(self.p1))])
        self.fpfh2 = np.vstack([self.p2.get_fpfh(i) for i in range(len(self.p2))])

        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
        self.k_candidates = int(k_candidates)
        self.num_points = int(num_points)
        self.feat_sample_len = int(feat_sample_len)
        self.pair_block_size = int(pair_block_size)
        self.use_faiss_gpu = bool(use_faiss_gpu and _FAISS_AVAILABLE)
        self.use_pytorch3d = bool(use_pytorch3d and _PYT3D_AVAILABLE and self.device.type == "cuda")
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Preprocess points and biochemical features (same helpers as earlier)
        self._prepare_points()
        self._prepare_biochem()

        # Build FAISS index for fpfh2
        self._build_faiss_index()

    # -------------------------
    # Point extraction & sampling
    # -------------------------
    def _extract_points_numpy(self, patch) -> np.ndarray:
        try:
            pts = np.asarray(patch[1].pcd.points)
        except Exception:
            try:
                pts = np.asarray(patch.pcd.points)
            except Exception:
                raise ValueError("Patch does not expose .pcd.points as expected")
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("Points must be shape (N,3)")
        return pts

    def _sample_or_pad_points(self, pts: np.ndarray) -> torch.Tensor:
        n = pts.shape[0]
        if n == 0:
            res = np.zeros((self.num_points, 3), dtype=np.float32)
            return torch.from_numpy(res).to(self.device)
        if n >= self.num_points:
            idx = np.random.choice(n, self.num_points, replace=False)
            sampled = pts[idx]
        else:
            pad_needed = self.num_points - n
            if n >= pad_needed:
                extra_idx = np.random.choice(n, pad_needed, replace=False)
            else:
                extra_idx = np.random.choice(n, pad_needed, replace=True)
            sampled = np.vstack([pts, pts[extra_idx]])
        return torch.from_numpy(sampled.astype(np.float32)).to(self.device)

    def _prepare_points(self):
        pts_list_1 = [self._sample_or_pad_points(self._extract_points_numpy(p)) for p in self.patches1]
        pts_list_2 = [self._sample_or_pad_points(self._extract_points_numpy(p)) for p in self.patches2]

        self.pts1 = torch.stack(pts_list_1, dim=0).to(self.device) if pts_list_1 else torch.empty((0, self.num_points, 3), device=self.device)
        self.pts2 = torch.stack(pts_list_2, dim=0).to(self.device) if pts_list_2 else torch.empty((0, self.num_points, 3), device=self.device)

        self.pts1_centered = self.pts1 - self.pts1.mean(dim=1, keepdim=True) if self.pts1.shape[0] else self.pts1
        self.pts2_centered = self.pts2 - self.pts2.mean(dim=1, keepdim=True) if self.pts2.shape[0] else self.pts2

    # -------------------------
    # Biochemical features (GPU Wasserstein-like)
    # -------------------------
    def _extract_biochem_dict(self, patch) -> Dict:
        try:
            d = patch[1].biochem_features
        except Exception:
            try:
                d = patch.biochem_features
            except Exception:
                d = {}
        return d or {}

    def _sample_or_pad_feature(self, arr: np.ndarray) -> torch.Tensor:
        arr = np.asarray(arr).astype(np.float32).ravel()
        if arr.size == 0:
            res = np.zeros((self.feat_sample_len,), dtype=np.float32)
            return torch.from_numpy(res).to(self.device)
        n = arr.size
        if n >= self.feat_sample_len:
            idx = np.random.choice(n, self.feat_sample_len, replace=False)
            sampled = arr[idx]
        else:
            pad_needed = self.feat_sample_len - n
            if n >= pad_needed:
                extra_idx = np.random.choice(n, pad_needed, replace=False)
            else:
                extra_idx = np.random.choice(n, pad_needed, replace=True)
            sampled = np.concatenate([arr, arr[extra_idx]])
        return torch.from_numpy(sampled).to(self.device)

    def _prepare_biochem(self):
        keys = set()
        for p in self.patches1:
            keys.update(self._extract_biochem_dict(p).keys())
        for p in self.patches2:
            keys.update(self._extract_biochem_dict(p).keys())
        keys = sorted(keys)
        self.feature_keys = keys
        K = len(keys)
        N1 = len(self.patches1)
        N2 = len(self.patches2)
        L = self.feat_sample_len

        if K == 0:
            self.bf1 = torch.empty((N1, 0, L), device=self.device)
            self.bf2 = torch.empty((N2, 0, L), device=self.device)
            return

        bf1_list = []
        for p in self.patches1:
            d = self._extract_biochem_dict(p)
            per_patch = [self._sample_or_pad_feature(d.get(k, np.array([]))) for k in keys]
            bf1_list.append(torch.stack(per_patch, dim=0))
        bf2_list = []
        for p in self.patches2:
            d = self._extract_biochem_dict(p)
            per_patch = [self._sample_or_pad_feature(d.get(k, np.array([]))) for k in keys]
            bf2_list.append(torch.stack(per_patch, dim=0))

        self.bf1 = torch.stack(bf1_list, dim=0).to(self.device) if bf1_list else torch.empty((0, K, L), device=self.device)
        self.bf2 = torch.stack(bf2_list, dim=0).to(self.device) if bf2_list else torch.empty((0, K, L), device=self.device)

    @staticmethod
    def _wasserstein_1d_batch_sorted(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_s = torch.sort(a, dim=-1).values
        b_s = torch.sort(b, dim=-1).values
        return torch.mean(torch.abs(a_s - b_s), dim=-1)

    # -------------------------
    # FAISS index building & candidate selection
    # -------------------------
    def _build_faiss_index(self):
        if not _FAISS_AVAILABLE:
            self.faiss_index = None
            return
        d = self.fpfh2.shape[1]
        index = faiss.IndexFlatL2(d)
        if self.use_faiss_gpu and self.device.type == "cuda":
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = False
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, index, co)
        else:
            self.faiss_index = index
        self.faiss_index.add(self.fpfh2)  # add descriptors for set2

    def _select_candidates_with_faiss(self) -> List[Tuple[int, int]]:
        """
        For every patch in patches1, retrieve top-k neighbors in patches2.
        Return unique list of (i,j) candidate pairs.
        """
        n1 = self.fpfh1.shape[0]
        if self.faiss_index is None:
            # fallback: all pairs
            return [(i, j) for i in range(n1) for j in range(self.fpfh2.shape[0])]
        # search in blocks to avoid large memory
        batch = 1024
        pairs = []
        for start in range(0, n1, batch):
            end = min(n1, start + batch)
            q = self.fpfh1[start:end]
            distances, indices = self.faiss_index.search(q, min(self.k_candidates, self.fpfh2.shape[0]))
            for local_i, neigh in enumerate(indices):
                i = start + local_i
                for j in neigh:
                    pairs.append((i, int(j)))
        # unique while preserving order
        seen = set()
        uniq = []
        for p in pairs:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq

    # -------------------------
    # Procrustes + Chamfer (PyTorch3D if available)
    # -------------------------
    @staticmethod
    def _batched_procrustes(X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        H = torch.einsum("bpi,bpj->bij", X, Y)
        U, S, Vt = torch.linalg.svd(H)
        R = torch.matmul(U, Vt)
        det = torch.linalg.det(R)
        if (det < 0).any():
            mask = det < 0
            U[mask, :, -1] *= -1.0
            R = torch.matmul(U, Vt)
        X_aligned = torch.einsum("bij,bpj->bpi", R, X)
        return X_aligned, R

    def _compute_chamfer_block(self, X_block: torch.Tensor, Y_block: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        X_block, Y_block: (B,P,3), centered
        Returns: chamfer (B,), symmetric_rmse (B,), mean_src_to_tgt (B,), mean_tgt_to_src (B,)
        """
        X_al, _R = self._batched_procrustes(X_block, Y_block)
        if self.use_pytorch3d:
            # pytorch3d chamfer returns (dist, (idx1, idx2)) where dist is mean of squared distances
            # to get symmetric distances use direct cdist as fallback because pytorch3d's return is L2 squared mean
            try:
                # attempt to use pytorch3d for potentially faster kernels
                # pyt3d_chamfer expects (B,P,3) and (B,Q,3)
                # It returns (torch.Tensor (B,), (idx1, idx2)) in newer releases; handle gracefully
                d, _ = pyt3d_chamfer(X_al, Y_block)
                # d is average squared distance; convert to approximate chamfer by doubling sqrt
                chamfer = torch.sqrt(d) * 2.0
                # For more precise per-direction means, fallback to cdist
                dmat = torch.cdist(X_al, Y_block)
            except Exception:
                dmat = torch.cdist(X_al, Y_block)
        else:
            dmat = torch.cdist(X_al, Y_block)
        d_src_to_tgt = dmat.min(dim=2).values
        d_tgt_to_src = dmat.min(dim=1).values
        mean_src = d_src_to_tgt.mean(dim=1)
        mean_tgt = d_tgt_to_src.mean(dim=1)
        chamfer = mean_src + mean_tgt
        symmetric_rmse = torch.sqrt(0.5 * ((d_src_to_tgt ** 2).mean(dim=1) + (d_tgt_to_src ** 2).mean(dim=1)))
        return chamfer, symmetric_rmse, mean_src, mean_tgt

    # -------------------------
    # High-level compute
    # -------------------------
    def compute_all_metrics(
        self,
        filter_threshold: Optional[float] = None,
        weights: Tuple[float, ...] = (0.8, 0.1, 0.1, 0.2, 0.2),
    ) -> List[Dict[str, Any]]:
        """
        Workflow:
          1) Use FAISS to get candidate pairs (top-k per patch1).
          2) Optionally run biochem filter (vectorized) and remove pairs above threshold.
          3) For remaining candidate pairs, compute batched Procrustes + Chamfer.
          4) Return list of dicts with same structure as prior implementation.

        Returns:
          list of {i, j, distance, registration (or None), biochem (np.ndarray)}
        """
        # 1) candidate selection
        candidate_pairs = self._select_candidates_with_faiss()
        if len(candidate_pairs) == 0:
            return []

        # 2) compute biochem distances for candidate pairs (batched)
        K = len(self.feature_keys)
        total_pairs = len(candidate_pairs)
        biochem_per_pair = np.zeros((total_pairs, K), dtype=np.float32) if K > 0 else np.zeros((total_pairs, 0), dtype=np.float32)

        # process in blocks to conserve memory
        blk = self.pair_block_size
        for start in range(0, total_pairs, blk):
            end = min(total_pairs, start + blk)
            idx_block = candidate_pairs[start:end]
            idx_i = torch.tensor([p[0] for p in idx_block], dtype=torch.long, device=self.device)
            idx_j = torch.tensor([p[1] for p in idx_block], dtype=torch.long, device=self.device)
            if K > 0:
                A = self.bf1[idx_i]  # (B,K,L)
                B = self.bf2[idx_j]  # (B,K,L)
                dist_block = self._wasserstein_1d_batch_sorted(A, B)  # (B,K)
                biochem_per_pair[start:end, :] = dist_block.cpu().numpy()

        # scalar biochem metric for filtering
        if K > 0:
            biochem_scalar = biochem_per_pair.mean(axis=1)
        else:
            biochem_scalar = np.zeros((total_pairs,), dtype=np.float32)

        if filter_threshold is not None:
            keep_mask = biochem_scalar < float(filter_threshold)
        else:
            keep_mask = np.ones((total_pairs,), dtype=bool)

        # separate pairs to compute registration (kept) vs skipped
        kept_pairs = [candidate_pairs[i] for i in range(total_pairs) if keep_mask[i]]
        skipped_pairs = [candidate_pairs[i] for i in range(total_pairs) if not keep_mask[i]]

        results: List[Dict[str, Any]] = []

        # 3) compute registration + chamfer for kept pairs in GPU blocks
        total_kept = len(kept_pairs)
        for start in range(0, total_kept, blk):
            end = min(total_kept, start + blk)
            block = kept_pairs[start:end]
            idx_i = torch.tensor([p[0] for p in block], dtype=torch.long, device=self.device)
            idx_j = torch.tensor([p[1] for p in block], dtype=torch.long, device=self.device)

            X_block = self.pts1_centered[idx_i]  # (B,P,3)
            Y_block = self.pts2_centered[idx_j]  # (B,P,3)

            chamfer_vals, sym_rmse, mean_src, mean_tgt = self._compute_chamfer_block(X_block, Y_block)

            # collect per pair
            for local_idx in range(end - start):
                i_idx = int(idx_i[local_idx].item())
                j_idx = int(idx_j[local_idx].item())
                global_index = None  # need to find position in candidate_pairs
                # find original pair index to index biochem_per_pair
                # Use dict mapping for speed
                results.append({
                    "i": i_idx,
                    "j": j_idx,
                    "distance": float(chamfer_vals[local_idx].cpu().item()),  # placeholder; final weighted below
                    "registration": {
                        "chamfer_distance": float(chamfer_vals[local_idx].cpu().item()),
                        "symmetric_rmse": float(sym_rmse[local_idx].cpu().item()),
                        "mean_src_to_tgt": float(mean_src[local_idx].cpu().item()),
                        "mean_tgt_to_src": float(mean_tgt[local_idx].cpu().item()),
                    },
                    "biochem": None  # fill later
                })

        # 4) assemble skipped results (registration=None)
        for (i_idx, j_idx) in skipped_pairs:
            results.append({
                "i": i_idx,
                "j": j_idx,
                "distance": None,
                "registration": None,
                "biochem": None
            })

        # 5) Fill biochem arrays and compute final weighted distance for each result
        # Build a mapping from pair -> index in candidate_pairs to pull biochem vector
        pair_to_idx = {candidate_pairs[i]: i for i in range(len(candidate_pairs))}
        for r in results:
            key = (r["i"], r["j"])
            idx = pair_to_idx.get(key, None)
            if idx is None:
                r["biochem"] = np.array([], dtype=np.float32)
            else:
                r["biochem"] = biochem_per_pair[idx]
            # compute weighted distance: if registration present, use weight[0]*chamfer + mean(biochem)*(rest)
            if r["registration"] is None:
                # use only biochem (average)
                if r["biochem"].size:
                    r["distance"] = float(r["biochem"].mean())
                else:
                    r["distance"] = 0.0
            else:
                reg_weight = float(weights[0])
                if r["biochem"].size:
                    biochem_weight = float(r["biochem"].mean())
                else:
                    biochem_weight = 0.0
                r["distance"] = float(reg_weight * r["registration"]["chamfer_distance"] + biochem_weight)

        return results

    @staticmethod
    def save_distances(path: str, results: List[Dict[str, Any]]):
        with open(path, "wb") as f:
            pickle.dump(results, f)
