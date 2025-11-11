import pickle
import torch

from surfacepatcher.surfacepatcher import ProteinPatches


class PatchComparison:
    def __init__(self, patches1: ProteinPatches, patches2:ProteinPatches):
        """
        take the resuls of genodesic patcher and perform pairwise comparisons on all the patches and their rotations
        :param patches1: surfacepatcher.surfacepatcher.ProteinPatches for protein A
        :param patches2: surfacepatcher.surfacepatcher.ProteinPatches for protein B
        """
        self.patches1 = patches1
        self.patches2 = patches2

    #TODO need to re-write this to compare one by one 
    @property
    def distances(self):
        patch_distances={}
        keys1=list(self.patches1.descriptors.keys())
        keys2 = list(self.patches2.descriptors.keys())
        for key1 in keys1:
            patch1_fixed = self.patches1.descriptors[key1].unsqueeze(0)
            for key2 in keys2:
                diff = patch1_fixed - self.patches2.descriptors[key2]  # shape [25, 25, 5, 6]
                dist = torch.sqrt(torch.sum(diff ** 2, dim=(1, 2, 3)))  # shape [25]
                min_dist=dist.min().item()
                min_index=dist.argmin().item()
                patch_distances[f"{key1}-{key2}"]=(min_dist, min_index)
        return patch_distances

    def save_distances(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.distances, f)
