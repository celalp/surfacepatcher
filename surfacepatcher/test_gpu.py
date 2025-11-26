
import torch
import numpy as np
from surfacepatcher.patchcomparison import PatchComparison
from dataclasses import dataclass

@dataclass(frozen=True)
class ProteinPatches:
    descriptors: dict

def test_patch_comparison():
    # Mock data
    N1 = 100
    N2 = 200
    M = 36
    K = 5
    P = 6
    
    # Create random descriptors
    # Shape (M, M, K, P)
    # We simulate rotations by rolling
    
    desc1 = {}
    for i in range(N1):
        base = torch.randn(M, K, P)
        # Create rotations
        rotations = torch.stack([torch.roll(base, shifts=r, dims=0) for r in range(M)])
        desc1[i] = rotations
        
    desc2 = {}
    for i in range(N2):
        base = torch.randn(M, K, P)
        rotations = torch.stack([torch.roll(base, shifts=r, dims=0) for r in range(M)])
        desc2[i] = rotations
        
    patches1 = ProteinPatches(descriptors=desc1)
    patches2 = ProteinPatches(descriptors=desc2)
    
    pc = PatchComparison(patches1, patches2)
    
    print("Running compute...")
    dists, idxs, k1, k2 = pc.compute(batch_size=10)
    
    print(f"Output shape: {dists.shape}")
    assert dists.shape == (N1, N2)
    assert idxs.shape == (N1, N2)
    
    # Sanity check: Distance of patch to itself (if identical)
    # Let's add an identical patch to patches2
    desc2[999] = desc1[0] # Same as patch 0 in P1
    patches2_mod = ProteinPatches(descriptors=desc2)
    pc_mod = PatchComparison(patches1, patches2_mod)
    dists_mod, idxs_mod, _, k2_mod = pc_mod.compute(batch_size=10)
    
    # Find index of key 999 in k2_mod
    idx_999 = k2_mod.index(999)
    dist_0_999 = dists_mod[0, idx_999]
    print(f"Distance between identical patches: {dist_0_999}")
    assert dist_0_999 < 1e-4, "Distance should be near zero for identical patches"
    
    # Check rotation invariance
    # Create a rotated version of patch 0
    rotated_base = torch.roll(desc1[0][0], shifts=10, dims=0)
    rotations_rotated = torch.stack([torch.roll(rotated_base, shifts=r, dims=0) for r in range(M)])
    desc2[1000] = rotations_rotated
    patches2_rot = ProteinPatches(descriptors=desc2)
    pc_rot = PatchComparison(patches1, patches2_rot)
    dists_rot, idxs_rot, _, k2_rot = pc_rot.compute(batch_size=10)
    
    idx_1000 = k2_rot.index(1000)
    dist_0_1000 = dists_rot[0, idx_1000]
    best_rot = idxs_rot[0, idx_1000]
    print(f"Distance to rotated patch: {dist_0_1000}, Best Rot Index: {best_rot}")
    
    assert dist_0_1000 < 1e-4, "Distance should be near zero for rotated patches"
    # The best rotation index should align them. 
    # If P2 is rotated by 10 relative to P1, then we need to rotate P2 by -10 (or 26) to match?
    # Or maybe the index tells us which rotation of P2 matched P1[0].
    # P2[0] is Rot(10) of P1[0].
    # P2[k] is Rot(10+k) of P1[0].
    # We want P2[k] == P1[0].
    # Rot(10+k) = Rot(0) -> 10+k = 0 (mod 36) -> k = -10 = 26.
    # So best_rot should be 26.
    print(f"Expected rotation index: {(-10) % M}")
    assert best_rot == (-10) % M

if __name__ == "__main__":
    test_patch_comparison()
