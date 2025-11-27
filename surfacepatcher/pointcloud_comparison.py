import pickle
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import chi2_kernel
from geodesic_patcher.surface_utils import SurfaceCache


class PointCloudComparison:
    def __init__(self, patches1, patches2, descriptor_type='fpfh', 
                 fpfh_radius_multiplier=2.0, shot_radius_multiplier=3.0,
                 zernike_order=10, zernike_grid_size=32,
                 surface_cache=None):
        """
        Compare protein surface patches using handcrafted 3D point cloud descriptors.
        
        :param patches1: geodesic_patcher.geodesic_patcher.ProteinPatches for protein A
        :param patches2: geodesic_patcher.geodesic_patcher.ProteinPatches for protein B
        :param descriptor_type: 'fpfh', 'shot', 'zernike', or 'all'
        :param fpfh_radius_multiplier: Multiplier for FPFH search radius (relative to avg point spacing)
        :param shot_radius_multiplier: Multiplier for SHOT search radius
        :param zernike_order: Order of Zernike polynomials (default: 10)
        :param zernike_grid_size: Voxel grid size for Zernike (default: 32)
        :param surface_cache: Optional SurfaceCache instance (will create if None)
        """
        self.patches1 = patches1
        self.patches2 = patches2
        self.descriptor_type = descriptor_type
        self.fpfh_radius_mult = fpfh_radius_multiplier
        self.shot_radius_mult = shot_radius_multiplier
        self.zernike_order = zernike_order
        self.zernike_grid_size = zernike_grid_size
        
        # Surface cache for accessing vertices/normals
        self.surface_cache = surface_cache if surface_cache is not None else SurfaceCache()
        
        # Load surface data
        self.vertices1, self.normals1 = self._load_surface_data(patches1)
        self.vertices2, self.normals2 = self._load_surface_data(patches2)
        
    def _load_surface_data(self, patches):
        """
        Load surface vertices and normals for a ProteinPatches object.
        
        :param patches: ProteinPatches instance
        :return: (vertices, normals)
        """
        pdb_file = patches.pdb_file
        # Extract chain_id if stored (may not be in current implementation)
        # For now, assume chain_id=0 or None
        _, vertices, _, normals, _ = self.surface_cache.load_surface(pdb_file, chain_id=None)
        return vertices, normals
        
    def _patch_to_pointcloud(self, patch, vertices):
        """
        Convert a patch to an Open3D point cloud with normals and features.
        
        :param patch: Single patch dict from ProteinPatches
        :param vertices: Full vertex array
        :return: o3d.geometry.PointCloud, feature_array (N, 6)
        """
        patch_indices = patch['indices']
        patch_vertices = vertices[patch_indices]
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(patch_vertices)
        
        # Estimate normals if not already available
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
        )
        
        # Extract biochemical features
        features = np.stack([
            patch['features']['shape_index'],
            patch['features']['mean_curvature'],
            patch['features']['electrostatic'],
            patch['features']['h_bond_donor'],
            patch['features']['h_bond_acceptor'],
            patch['features']['hydrophobicity']
        ], axis=1)  # Shape: (N, 6)
        
        return pcd, features
    
    def _compute_fpfh(self, pcd, radius):
        """
        Compute Fast Point Feature Histogram (FPFH) descriptors.
        
        :param pcd: Open3D point cloud with normals
        :param radius: Search radius for feature computation
        :return: FPFH features (N, 33) - 33-dimensional histogram
        """
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
        )
        return np.asarray(fpfh.data).T  # Transpose to (N, 33)
    
    def _compute_shot(self, pcd, radius):
        """
        Compute SHOT (Signature of Histograms of Orientations) descriptors.
        Note: Open3D doesn't have native SHOT, so we use FPFH as a proxy.
        For true SHOT, consider using PCL bindings or custom implementation.
        
        :param pcd: Open3D point cloud with normals
        :param radius: Search radius for feature computation
        :return: SHOT-like features
        """
        # Using FPFH as a robust alternative (both are histogram-based local descriptors)
        # For production, consider integrating PCL's SHOT implementation
        return self._compute_fpfh(pcd, radius)
    
    def _aggregate_descriptors(self, geometric_desc, biochem_features):
        """
        Aggregate point-wise descriptors to a single patch-level descriptor.
        
        :param geometric_desc: (N, D_geom) geometric descriptors (FPFH/SHOT)
        :param biochem_features: (N, 6) biochemical features
        :return: Patch-level descriptor vector
        """
        # Statistical aggregation of geometric descriptors
        geom_mean = np.mean(geometric_desc, axis=0)
        geom_std = np.std(geometric_desc, axis=0)
        geom_max = np.max(geometric_desc, axis=0)
        geom_min = np.min(geometric_desc, axis=0)
        
        # Statistical aggregation of biochemical features
        biochem_mean = np.mean(biochem_features, axis=0)
        biochem_std = np.std(biochem_features, axis=0)
        biochem_max = np.max(biochem_features, axis=0)
        biochem_min = np.min(biochem_features, axis=0)
        
        # Concatenate all statistics
        patch_descriptor = np.concatenate([
            geom_mean, geom_std, geom_max, geom_min,
            biochem_mean, biochem_std, biochem_max, biochem_min
        ])
        
        return patch_descriptor
    
    def _compute_patch_descriptor(self, patch, vertices):
        """
        Compute a single patch descriptor combining geometric and biochemical features.
        
        :param patch: Patch dict
        :param vertices: Full vertex array
        :return: Patch descriptor vector
        """
        pcd, biochem_features = self._patch_to_pointcloud(patch, vertices)
        
        # Estimate appropriate radius based on point density
        points = np.asarray(pcd.points)
        if len(points) < 2:
            # Degenerate patch, return zeros
            if self.descriptor_type == 'zernike':
                # Zernike descriptor size depends on order
                n_zernike = sum(1 for n in range(self.zernike_order + 1) 
                               for l in range(n + 1) if (n - l) % 2 == 0)
                return np.zeros(n_zernike + 6 * 4)
            elif self.descriptor_type in ['fpfh', 'shot']:
                return np.zeros(33 * 4 + 6 * 4)  # FPFH stats + biochem stats
            elif self.descriptor_type == 'all':
                n_zernike = sum(1 for n in range(self.zernike_order + 1) 
                               for l in range(n + 1) if (n - l) % 2 == 0)
                return np.zeros(33 * 4 * 2 + n_zernike + 6 * 4)
            else:
                return np.zeros(33 * 4 * 2 + 6 * 4)  # Both FPFH/SHOT
        
        # Compute average nearest neighbor distance
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        avg_spacing = np.mean(np.min(dists, axis=1))
        
        # Compute geometric descriptors based on type
        if self.descriptor_type == 'fpfh':
            fpfh_radius = avg_spacing * self.fpfh_radius_mult
            geometric_desc = self._compute_fpfh(pcd, fpfh_radius)
            patch_desc = self._aggregate_descriptors(geometric_desc, biochem_features)
            
        elif self.descriptor_type == 'shot':
            shot_radius = avg_spacing * self.shot_radius_mult
            geometric_desc = self._compute_shot(pcd, shot_radius)
            patch_desc = self._aggregate_descriptors(geometric_desc, biochem_features)
            
        elif self.descriptor_type == 'zernike':
            # Compute Zernike descriptors
            from surfacepatcher.zernike_descriptors import Zernike3D
            
            zernike = Zernike3D(order=self.zernike_order, grid_size=self.zernike_grid_size)
            # Use electrostatic as primary feature for voxelization
            electrostatic = patch['features']['electrostatic']
            zernike_desc = zernike.compute_descriptors(points, features=electrostatic[:, None])
            
            # Aggregate biochemical features
            biochem_mean = np.mean(biochem_features, axis=0)
            biochem_std = np.std(biochem_features, axis=0)
            biochem_max = np.max(biochem_features, axis=0)
            biochem_min = np.min(biochem_features, axis=0)
            biochem_agg = np.concatenate([biochem_mean, biochem_std, biochem_max, biochem_min])
            
            # Combine Zernike with biochemical features
            patch_desc = np.concatenate([zernike_desc, biochem_agg])
            
        elif self.descriptor_type == 'both':
            fpfh_radius = avg_spacing * self.fpfh_radius_mult
            shot_radius = avg_spacing * self.shot_radius_mult
            
            fpfh_desc = self._compute_fpfh(pcd, fpfh_radius)
            shot_desc = self._compute_shot(pcd, shot_radius)
            
            fpfh_agg = self._aggregate_descriptors(fpfh_desc, biochem_features)
            shot_agg = self._aggregate_descriptors(shot_desc, biochem_features)
            
            # Concatenate both (remove duplicate biochem features)
            patch_desc = np.concatenate([
                fpfh_agg[:-24],  # FPFH geometric only
                shot_agg[:-24],  # SHOT geometric only
                fpfh_agg[-24:]   # Biochem features (once)
            ])
            
        elif self.descriptor_type == 'all':
            # Compute all three descriptor types
            from surfacepatcher.zernike_descriptors import Zernike3D
            
            fpfh_radius = avg_spacing * self.fpfh_radius_mult
            shot_radius = avg_spacing * self.shot_radius_mult
            
            fpfh_desc = self._compute_fpfh(pcd, fpfh_radius)
            shot_desc = self._compute_shot(pcd, shot_radius)
            
            zernike = Zernike3D(order=self.zernike_order, grid_size=self.zernike_grid_size)
            electrostatic = patch['features']['electrostatic']
            zernike_desc = zernike.compute_descriptors(points, features=electrostatic[:, None])
            
            fpfh_agg = self._aggregate_descriptors(fpfh_desc, biochem_features)
            shot_agg = self._aggregate_descriptors(shot_desc, biochem_features)
            
            # Concatenate all (remove duplicate biochem features)
            patch_desc = np.concatenate([
                fpfh_agg[:-24],    # FPFH geometric only
                shot_agg[:-24],    # SHOT geometric only
                zernike_desc,      # Zernike descriptors
                fpfh_agg[-24:]     # Biochem features (once)
            ])
        else:
            raise ValueError(f"Unknown descriptor_type: {self.descriptor_type}")
        
        return patch_desc
    
    def compute(self, distance_metric='euclidean'):
        """
        Compute pairwise distances between all patches using point cloud descriptors.
        
        :param distance_metric: 'euclidean', 'cosine', or 'chi2'
        :return: (distances, keys1, keys2)
                 distances: np.ndarray of shape (N1, N2) with descriptor distances
                 keys1: list of keys for patches1
                 keys2: list of keys for patches2
        """
        keys1 = sorted(list(self.patches1.patches.keys()))
        keys2 = sorted(list(self.patches2.patches.keys()))
        
        print("Computing descriptors for protein 1...")
        descriptors1 = []
        for key in tqdm(keys1):
            patch = self.patches1.patches[key]
            desc = self._compute_patch_descriptor(patch, self.vertices1)
            descriptors1.append(desc)
        
        print("Computing descriptors for protein 2...")
        descriptors2 = []
        for key in tqdm(keys2):
            patch = self.patches2.patches[key]
            desc = self._compute_patch_descriptor(patch, self.vertices2)
            descriptors2.append(desc)
        
        descriptors1 = np.array(descriptors1)
        descriptors2 = np.array(descriptors2)
        
        # Compute pairwise distances
        print("Computing pairwise distances...")
        if distance_metric == 'euclidean':
            distances = cdist(descriptors1, descriptors2, metric='euclidean')
        elif distance_metric == 'cosine':
            distances = cdist(descriptors1, descriptors2, metric='cosine')
        elif distance_metric == 'chi2':
            # Chi-square distance for histogram comparison
            # Ensure non-negative values
            desc1_pos = np.abs(descriptors1)
            desc2_pos = np.abs(descriptors2)
            # Normalize
            desc1_norm = desc1_pos / (desc1_pos.sum(axis=1, keepdims=True) + 1e-10)
            desc2_norm = desc2_pos / (desc2_pos.sum(axis=1, keepdims=True) + 1e-10)
            distances = cdist(desc1_norm, desc2_norm, metric='chi2')
        else:
            raise ValueError(f"Unknown distance_metric: {distance_metric}")
        
        return distances, keys1, keys2
    
    def _get_vertices_from_patch(self, patch, vertices_full):
        """
        Get vertex positions for a specific patch.
        
        :param patch: Patch dict with 'indices'
        :param vertices_full: Full surface vertices array
        :return: (N, 3) vertex array for patch
        """
        indices = patch['indices']
        return vertices_full[indices]
    
    def save_distances(self, path, results):
        """
        Save the comparison results.
        
        :param path: Output file path
        :param results: Tuple (distances, keys1, keys2)
        """
        dists, k1, k2 = results
        data = {
            "distances": dists,
            "keys1": k1,
            "keys2": k2,
            "descriptor_type": self.descriptor_type
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
