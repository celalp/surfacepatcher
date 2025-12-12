import pickle
import numpy as np
import open3d as o3d
from scipy.stats import wasserstein_distance
from tqdm import tqdm

class PointCloudComparison:
    def __init__(self, patches1, patches2):
        self.patches1 = patches1
        self.patches2 = patches2

    def _get_diag(self, pcd):
        min_b = pcd.get_min_bound()
        max_b = pcd.get_max_bound()
        diag = np.linalg.norm(max_b - min_b)
        return diag

    def _ransac(self, patch1, patch2, voxel_multiplier=1.5, n_iter=3):
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

    def _compute_alignment(self):
        results=[]
        for patch1 in self.patches1:
            for patch2 in tqdm(self.patches2):
                results.append(self._register(patch1, patch2))

        return results

    #assuming that we calculated the same features for both patches
    def _biochem_alignment(self):
        results=[]
        for patch1 in self.patches1:
            p1_feat=[]
            for _, value in patch1[1].biochem_features.items():
                p1_feat.append(value)
            for patch2 in self.patches2:
                p2_feat = []
                for _, value in patch2[1].biochem_features.items():
                    p2_feat.append(value)
            distances=[]
            for feat1, feat2 in zip(p1_feat, p2_feat):
                distances.append(wasserstein_distance(feat1, feat2))

        results.append(distances)
        return results

    def _weigh_alignment(self, registration_distances, biochem_distances, weights=(0.8, 0.1, 0.1, 0.2, 0.2)):
        global_distances=[]
        for reg_dist, biochem_dist in zip(registration_distances, biochem_distances):
            biochem_weighted=[item*weight for item, weight in zip(biochem_dist, weights[1:])]
            reg_dist_weighted=registration_distances["chamfer_distance"]*weights[0]
            global_dist=reg_dist_weighted+sum(biochem_weighted)/5
            global_distances.append(global_dist)

        return global_distances

    def compute_all_metrics(self):
        registration_distances=self._compute_alignment()
        biochem_distances=self._biochem_alignment()
        return self._weigh_alignment(registration_distances, biochem_distances)

    def save_distances(self, path, results):
        with open(path, 'wb') as f:
            pickle.dump(results, f)








