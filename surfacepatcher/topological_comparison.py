import pickle
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from surfacepatcher.surface_utils import SurfaceCache

from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, Amplitude, NumberOfPoints
from gtda.diagrams import PersistenceImage, PersistenceLandscape
from gtda.diagrams import Scaler

class TopologicalComparison:
    def __init__(self, patches1, patches2, homology_dimensions=(0, 1, 2),
                 max_edge_length=15.0, use_feature_filtration=True,
                 surface_cache=None):
        """
        Compare protein surface patches using persistent homology and topological descriptors.
        
        :param patches1: surfacepatcher.surfacepatcher.ProteinPatches for protein A
        :param patches2: surfacepatcher.surfacepatcher.ProteinPatches for protein B
        :param homology_dimensions: Tuple of homology dimensions to compute (0=components, 1=loops, 2=voids)
        :param max_edge_length: Maximum edge length for Vietoris-Rips filtration
        :param use_feature_filtration: If True, also compute feature-weighted filtrations
        :param surface_cache: Optional SurfaceCache instance (will create if None)
        """
        
        self.patches1 = patches1
        self.patches2 = patches2
        self.homology_dimensions = homology_dimensions
        self.max_edge_length = max_edge_length
        self.use_feature_filtration = use_feature_filtration
        
        # Surface cache for accessing vertices
        self.surface_cache = surface_cache if surface_cache is not None else SurfaceCache()
        
        # Load surface data
        self.vertices1, _ = self._load_surface_data(patches1)
        self.vertices2, _ = self._load_surface_data(patches2)
        
        # Initialize persistence computer
        self.vr_persistence = VietorisRipsPersistence(
            homology_dimensions=homology_dimensions,
            max_edge_length=max_edge_length,
            n_jobs=-1  # Use all cores
        )
    
    def _load_surface_data(self, patches):
        """
        Load surface vertices for a ProteinPatches object.
        
        :param patches: ProteinPatches instance
        :return: (vertices, normals)
        """
        pdb_file = patches.pdb_file
        _, vertices, _, normals, _ = self.surface_cache.load_surface(pdb_file, chain_id=None)
        return vertices, normals
        
    def _get_patch_points(self, patch, vertices_full):
        """
        Extract point coordinates from patch using actual surface vertices.
        
        :param patch: Patch dict with 'indices'
        :param vertices_full: Full surface vertices array
        :return: (N, 3) array of point coordinates
        """
        indices = patch['indices']
        return vertices_full[indices]
    
    def _compute_persistence_diagram(self, points):
        """
        Compute persistence diagram for a point cloud.
        
        :param points: (N, 3) array of point coordinates
        :return: Persistence diagram
        """
        # Reshape for giotto-tda (expects batch dimension)
        points_batch = points[np.newaxis, :, :]  # (1, N, 3)
        
        # Compute persistence
        diagrams = self.vr_persistence.fit_transform(points_batch)
        
        return diagrams[0]  # Return single diagram
    
    def _compute_feature_weighted_persistence(self, points, feature_values, feature_name):
        """
        Compute persistence using a feature as the filtration function.
        This captures topology of level sets of the feature.
        
        :param points: (N, 3) point coordinates
        :param feature_values: (N,) feature values at each point
        :param feature_name: Name of the feature (for debugging)
        :return: Modified persistence diagram
        """
        # Create distance matrix weighted by feature similarity
        geom_dists = cdist(points, points, metric='euclidean')
        
        # Normalize feature values
        feat_norm = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min() + 1e-10)
        
        # Feature distance matrix
        feat_dists = np.abs(feat_norm[:, None] - feat_norm[None, :])
        
        # Combined distance: geometric + feature
        # Weight feature contribution
        combined_dists = geom_dists + 2.0 * feat_dists
        
        # Use lower-star filtration on feature values
        # This is a simplified approach; for true sublevel set persistence,
        # we'd need a different filtration
        
        # For now, return standard geometric persistence
        # TODO: Implement proper feature-based filtration
        return self._compute_persistence_diagram(points)
    
    def _vectorize_persistence_diagram(self, diagram):
        """
        Convert persistence diagram to a fixed-size vector representation.
        Uses multiple topological statistics.
        
        :param diagram: Persistence diagram from giotto-tda
        :return: Feature vector
        """
        # Reshape for giotto-tda transformers
        diagram_batch = diagram[np.newaxis, :, :]  # (1, N_points, 3)
        
        features = []
        
        # 1. Persistence Entropy
        entropy_computer = PersistenceEntropy()
        entropy = entropy_computer.fit_transform(diagram_batch)
        features.append(entropy.flatten())
        
        # 2. Amplitude (various norms)
        for metric in ['bottleneck', 'wasserstein', 'landscape', 'betti', 'heat']:
            try:
                amp_computer = Amplitude(metric=metric)
                amplitude = amp_computer.fit_transform(diagram_batch)
                features.append(amplitude.flatten())
            except:
                # Some metrics might not be available
                pass
        
        # 3. Number of points in each homology dimension
        n_points_computer = NumberOfPoints()
        n_points = n_points_computer.fit_transform(diagram_batch)
        features.append(n_points.flatten())
        
        # 4. Persistence statistics (birth, death, persistence)
        for hom_dim in self.homology_dimensions:
            # Filter diagram by homology dimension
            dim_mask = diagram[:, 2] == hom_dim
            dim_diagram = diagram[dim_mask]
            
            if len(dim_diagram) > 0:
                births = dim_diagram[:, 0]
                deaths = dim_diagram[:, 1]
                persistences = deaths - births
                
                # Statistical features
                stats = [
                    np.mean(births), np.std(births), np.max(births) if len(births) > 0 else 0,
                    np.mean(deaths), np.std(deaths), np.max(deaths) if len(deaths) > 0 else 0,
                    np.mean(persistences), np.std(persistences), np.max(persistences),
                    np.sum(persistences),  # Total persistence
                    len(persistences)  # Number of features
                ]
                features.append(np.array(stats))
            else:
                # No features in this dimension
                features.append(np.zeros(11))
        
        # Concatenate all features
        feature_vector = np.concatenate(features)
        
        return feature_vector
    
    def _compute_patch_descriptor(self, patch, vertices_full):
        """
        Compute topological descriptor for a single patch.
        
        :param patch: Patch dict
        :param vertices_full: Full surface vertices array
        :return: Topological feature vector
        """
        points = self._get_patch_points(patch, vertices_full)
        
        if len(points) < 4:
            # Too few points for meaningful topology
            # Return zero vector
            return np.zeros(100)  # Placeholder size
        
        # 1. Standard geometric persistence
        diagram_geom = self._compute_persistence_diagram(points)
        features_geom = self._vectorize_persistence_diagram(diagram_geom)
        
        if not self.use_feature_filtration:
            return features_geom
        
        # 2. Feature-weighted persistence (electrostatic)
        electrostatic = patch['features']['electrostatic']
        diagram_elec = self._compute_feature_weighted_persistence(
            points, electrostatic, 'electrostatic'
        )
        features_elec = self._vectorize_persistence_diagram(diagram_elec)
        
        # 3. Feature-weighted persistence (hydrophobicity)
        hydrophobic = patch['features']['hydrophobicity']
        diagram_hydro = self._compute_feature_weighted_persistence(
            points, hydrophobic, 'hydrophobicity'
        )
        features_hydro = self._vectorize_persistence_diagram(diagram_hydro)
        
        # Concatenate all topological features
        combined_features = np.concatenate([
            features_geom,
            features_elec,
            features_hydro
        ])
        
        return combined_features
    
    def compute(self, distance_metric='euclidean'):
        """
        Compute pairwise distances between patches using topological descriptors.
        
        :param distance_metric: 'euclidean', 'cosine', or 'wasserstein' (for diagrams directly)
        :return: (distances, keys1, keys2)
        """
        keys1 = sorted(list(self.patches1.patches.keys()))
        keys2 = sorted(list(self.patches2.patches.keys()))
        
        print("Computing topological descriptors for protein 1...")
        descriptors1 = []
        for key in tqdm(keys1):
            patch = self.patches1.patches[key]
            desc = self._compute_patch_descriptor(patch, self.vertices1)
            descriptors1.append(desc)
        
        print("Computing topological descriptors for protein 2...")
        descriptors2 = []
        for key in tqdm(keys2):
            patch = self.patches2.patches[key]
            desc = self._compute_patch_descriptor(patch, self.vertices2)
            descriptors2.append(desc)
        
        # Ensure all descriptors have same length (pad if necessary)
        max_len = max(max(len(d) for d in descriptors1), max(len(d) for d in descriptors2))
        descriptors1 = np.array([np.pad(d, (0, max_len - len(d))) for d in descriptors1])
        descriptors2 = np.array([np.pad(d, (0, max_len - len(d))) for d in descriptors2])
        
        # Compute pairwise distances
        print("Computing pairwise distances...")
        if distance_metric == 'euclidean':
            distances = cdist(descriptors1, descriptors2, metric='euclidean')
        elif distance_metric == 'cosine':
            distances = cdist(descriptors1, descriptors2, metric='cosine')
        else:
            raise ValueError(f"Unknown distance_metric: {distance_metric}")
        
        return distances, keys1, keys2
    
    def save_distances(self, path, results):
        """
        Save comparison results.
        
        :param path: Output file path
        :param results: Tuple (distances, keys1, keys2)
        """
        dists, k1, k2 = results
        data = {
            "distances": dists,
            "keys1": k1,
            "keys2": k2,
            "homology_dimensions": self.homology_dimensions,
            "max_edge_length": self.max_edge_length
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
