import pickle
import numpy as np
import open3d as o3d


class PointCloudComparison:
    def __init__(self, patches1, patches2):
        self.patches1 = patches1
        self.patches2 = patches2

    def _get_diag(self, patch):
        min_b = patch["pcd"].get_min_bound()
        max_b = patch["pcd"].get_max_bound()
        diag = np.linalg.norm(max_b - min_b)
        return diag

    def _ransac(self, patch1, patch2, voxel_multiplier=1.5, n_iter=3):
        pc1=patch1.pcd
        pc2=patch2.pcd
        fpfh1=patch1.fpfh_features
        fpfh2=patch2.fpfh_features
        diag1=self._get_diag(patch1)
        diag2=self._get_diag(patch2)

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
        fpfh1 = patch1.fpfh_features
        fpfh2 = patch2.fpfh_features
        icp_aligned = o3d.pipelines.registration.registration_icp(
            pc1,
            pc2,
            fpfh1,
            fpfh2,
            distance_threshold,
            ransac_aligned.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        return icp_aligned

    def _register(self, patch1, patch2, voxel_multiplier=1.5, n_iter=3):
        ransac_aligned, distance_threshold = self._ransac(patch1, patch2, voxel_multiplier, n_iter)
        icp_aligned = self._icp(patch1, patch2, ransac_aligned, distance_threshold)
        pc1_aligned=patch1["pcd"].transform(icp_aligned.transformation)
        tgt_tree = o3d.geometry.KDTreeFlann(patch2["pcd"])
        src_tree = o3d.geometry.KDTreeFlann(pc1_aligned)

        src_pts = np.asarray(pc1_aligned.points)
        tgt_pts = np.asarray(patch2["pcd"].points)

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
        return {
            "chamfer_distance": float(chamfer),
            "symmetric_rmse": float(symmetric_rmse),
            "mean_src_to_tgt": float(d_src_to_tgt.mean()),
            "mean_tgt_to_src": float(d_tgt_to_src.mean()),
            "registration_fitness": icp_aligned.fitness,
            "registration_inlier_rmse": icp_aligned.inlier_rmse,
        }

    def _compute_alignment(self):
        results=[]
        for patch1 in self.patches1:
            for patch2 in self.patches2:
                results.append(self._register(patch1, patch2))

        return results

    def _biochem_alignment(self):
        results=[]
        for patch1 in self.patches1:
            for patch2 in self.patches2:
                results.append(np.linalg.norm(patch1["biochemical_features"] -
                                              patch2["biochemical_features"]))

        return results

    def _weigh_alignment(self, registration_distances, biochem_distances, weights=(0.8, 0.2)):
        global_distances=[]
        for reg_dist, biochem_dist in zip(registration_distances, biochem_distances):
            global_dist=reg_dist["chamfer_distance"]*weights[0]+biochem_dist*weights[1]
            global_distances.append(global_dist)

        return global_distances

    def compute_all_metrics(self):
        registration_distances=self._compute_alignment()
        biochem_distances=self._biochem_alignment()
        return self._weigh_alignment(registration_distances, biochem_distances)

    def save_distances(self, path, results):
        with open(path, 'wb') as f:
            pickle.dump(results, f)








