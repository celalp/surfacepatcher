from dataclasses import dataclass
import pickle
import numpy as np
import open3d as o3d
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from multiprocessing import Pool

@dataclass(frozen=True)
class PatchAlignment:
    """
    patch alignment results, this takes 2 patches that are not supposed to be skipped and then calculates a bunch of
    aligment metrics see below methods for descriptions
    """
    p1_index: int
    p2_index: int
    chamfer_distance: float
    symmetric_rmse: float
    mean_src_to_tgt: float
    mean_tgt_to_src: float
    registration_fitness: float
    registration_inlier_rmse: float
    biochem_distances: list #this is in order, I probably should change it to a dict
    weighted_alignmentd: float

class PointCloudComparison:
    def __init__(self, patches1, patches2):
        """
        initialize the comparison object
        :param patches1: output from geodesic patcher
        :param patches2: output from geodesic patcher
        """
        self.patches1 = patches1
        self.patches2 = patches2

    def _get_diag(self, pcd):
        """
        get the diagonal length of a point cloud, this will be used to set the _ransac alignment voxel size
        :param pcd: point cloud
        :return: diag: float
        """
        min_b = pcd.get_min_bound()
        max_b = pcd.get_max_bound()
        diag = np.linalg.norm(max_b - min_b)
        return diag

    def _ransac(self, patch1, patch2, voxel_multiplier=1.5, n_iter=3):
        """
        ransac alignment of protein point clouds
        :param patch1: patch from ProteinPatches
        :param patch2: patch from ProteinPatches
        :param voxel_multiplier: size of the voxel for cutoff
        :param n_iter: number of ransac iterations
        :return: aligned point clouds and the distance threshold, the latter will be used for icp
        """
        pc1=patch1[1].pcd
        pc2=patch2[1].pcd
        fpfh1=patch1[1].fpfh
        fpfh2=patch2[1].fpfh
        diag1=self._get_diag(pc1)
        diag2=self._get_diag(pc2)

        voxel_size = max(diag1, diag2) / 50
        distance_threshold = voxel_size * voxel_multiplier

        ransac_aligned=o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pc1,
            pc2,
            fpfh1,
            fpfh2,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=n_iter,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return ransac_aligned, distance_threshold

    def _icp(self, patch1, patch2, ransac_aligned, distance_threshold):
        """
        iterative point cloud registration after ransac alignment
        :param patch1: patch from ProteinPatches
        :param patch2: patch from ProteinPatches
        :param ransac_aligned: alignment from ransac, we need the rotation matrix
        :param distance_threshold: distances threshold from ransac for icp registration
        :return:
        """
        pc1 = patch1.pcd
        pc2 = patch2.pcd

        icp_aligned = o3d.pipelines.registration.registration_icp(
            pc1,
            pc2,
            distance_threshold,
            ransac_aligned.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        return icp_aligned

    def _register(self, patch1, patch2, voxel_multiplier=1.5, n_iter=3):
        """
        run the full registration pipeline to get some metrics
        :param patch1: patch from ProteinPatches
        :param patch2: patch from ProteinPatches
        :param voxel_multiplier: see above
        :param n_iter: see above
        :return: a dictionary of metrics
        results={
            "p1_index":patch1[1].center,
            "p2_index":patch2[1].center,
            "chamfer_distance": float(chamfer),
            "symmetric_rmse": float(symmetric_rmse),
            "mean_src_to_tgt": float(d_src_to_tgt.mean()),
            "mean_tgt_to_src": float(d_tgt_to_src.mean()),
            "registration_fitness": icp_aligned.fitness,
            "registration_inlier_rmse": icp_aligned.inlier_rmse,
        }
        """
        ransac_aligned, distance_threshold = self._ransac(patch1, patch2, voxel_multiplier, n_iter)
        icp_aligned = self._icp(patch1[1], patch2[1], ransac_aligned, distance_threshold)
        pc1_aligned=patch1[1].pcd.transform(icp_aligned.transformation)
        tgt_tree = o3d.geometry.KDTreeFlann(patch2[1].pcd)
        src_tree = o3d.geometry.KDTreeFlann(pc1_aligned)

        src_pts = np.asarray(pc1_aligned.points)
        tgt_pts = np.asarray(patch2[1].pcd.points)

        # --- One-way: src → tgt ---
        d_src_to_tgt = []
        for p in src_pts:
            _, _, dist2 = tgt_tree.search_knn_vector_3d(p, 1)
            d_src_to_tgt.append(np.sqrt(dist2[0]))

        # --- One-way: tgt → src ---
        d_tgt_to_src = []
        for p in tgt_pts:
            _, _, dist2 = src_tree.search_knn_vector_3d(p, 1)
            d_tgt_to_src.append(np.sqrt(dist2[0]))

        d_src_to_tgt = np.array(d_src_to_tgt)
        d_tgt_to_src = np.array(d_tgt_to_src)

        # Symmetric metrics
        chamfer = d_src_to_tgt.mean() + d_tgt_to_src.mean()
        symmetric_rmse = np.sqrt(
            0.5 * ((d_src_to_tgt ** 2).mean() + (d_tgt_to_src ** 2).mean())
        )
        results={
            "chamfer_distance": float(chamfer),
            "symmetric_rmse": float(symmetric_rmse),
            "mean_src_to_tgt": float(d_src_to_tgt.mean()),
            "mean_tgt_to_src": float(d_tgt_to_src.mean()),
            "registration_fitness": icp_aligned.fitness,
            "registration_inlier_rmse": icp_aligned.inlier_rmse,
        }

        return results

    def _biochem_alignment(self, patch1, patch2):
        """
        simple wasserstein distance between the biochemical features
        :param patch1: patches from ProteinPatches
        :param patch2: patches from ProteinPatches
        :return: a list of distances for each metric in the order of h bond donor, h bond acceptor, hydrophobicity, and electrostatic
        charge from apbs
        """
        p1_feat=[]
        for _, value in patch1[1].biochem_features.items():
            p1_feat.append(value)
        p2_feat = []
        for _, value in patch2[1].biochem_features.items():
            p2_feat.append(value)
        distances=[]
        for feat1, feat2 in zip(p1_feat, p2_feat):
            distances.append(wasserstein_distance(feat1, feat2))
        return distances

       #assuming that we calculated the same features for both patches

    def _weigh_alignment(self, registration_distance, biochem_distances, metric="chamfer_distance",
                         weights=(0.8, 0.1, 0.1, 0.2, 0.2)) -> float:
        """
        get weighted distance after registration
        :param registration_distances: registration distances from ransac and icp
        :param metric: which metric to use for registration default is chamfer distance
        :param biochem_distances: biochem wasserstein distances
        :param weights: weight for each, the order is important
        :return: a single metric of weighted distance
        """
        biochem_weighted=[item*weight for item, weight in zip(biochem_distances, weights[1:])]
        reg_dist_weighted=registration_distance[metric]*weights[0]
        global_dist=reg_dist_weighted+sum(biochem_weighted)/5
        return global_dist

    def _compute_alignment(self, patch1, patch2, voxel_multiplier=1.5, n_iter=3, metric="chamfer_distance",
                           weights=(0.8, 0.1, 0.1, 0.2, 0.2)):
        """
        full alignment pipeline from start to finish for 2 patches
        :param patch1: see above
        :param patch2: see above
        :param voxel_multiplier: see above
        :param n_iter: see above
        :param metric: see above
        :param weights: see above
        :return: a PatchAlignment dataclass
        """
        registration_distances=self._register(patch1, patch2, voxel_multiplier, n_iter)
        biochem_distances=self._biochem_alignment(patch1, patch2)
        weighted_distances=self._weigh_alignment(registration_distances, biochem_distances, metric, weights)
        results=PatchAlignment(patch1[1].center, patch2[1].center,
                               registration_distances["chamfer_distance"],
                               registration_distances["symmetric_rmse"],
                               registration_distances["mean_src_to_tgt"],
                               registration_distances["mean_tgt_to_src"],
                               registration_distances["registration_fitness"],
                               registration_distances["registration_inlier_rmse"],
                               biochem_distances,
                               weighted_distances)
        return results

    #TODO add multiprocessing
    def compute_all_metrics(self):
        results=[]
        for patch1 in tqdm(self.patches1):
            if not patch1[1].skip:
                for patch2 in self.patches2:
                    if not patch2[1].skip:
                        distance=self._compute_alignment(patch1, patch2)
                        results.append(distance)
        return results

    def save_distances(self, path, results):
        with open(path, 'wb') as f:
            pickle.dump(results, f)








