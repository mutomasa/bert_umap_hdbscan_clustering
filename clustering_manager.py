"""
Clustering Manager Module
Handles UMAP dimensionality reduction and HDBSCAN clustering.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import streamlit as st


class ClusteringManager:
    """UMAP and HDBSCAN clustering manager"""
    
    def __init__(self):
        """Initialize the clustering manager"""
        self.umap_reducer = None
        self.hdbscan_clusterer = None
        self.scaler = StandardScaler()
    
    def apply_umap(
        self, 
        embeddings: np.ndarray, 
        n_neighbors: int = 15, 
        min_dist: float = 0.1, 
        n_components: int = 2,
        metric: str = 'cosine',
        random_state: int = 42
    ) -> np.ndarray:
        """
        Apply UMAP dimensionality reduction
        
        Args:
            embeddings: Input embeddings
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance between points
            n_components: Number of output dimensions
            metric: Distance metric
            random_state: Random seed
            
        Returns:
            UMAP reduced embeddings
        """
        # Standardize embeddings
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Apply UMAP
        self.umap_reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=random_state,
            verbose=True
        )
        
        umap_embeddings = self.umap_reducer.fit_transform(embeddings_scaled)
        
        return umap_embeddings
    
    def apply_hdbscan(
        self, 
        embeddings: np.ndarray, 
        min_cluster_size: int = 5, 
        min_samples: int = 3,
        cluster_selection_epsilon: float = 0.0,
        alpha: float = 1.0
    ) -> np.ndarray:
        """
        Apply HDBSCAN clustering
        
        Args:
            embeddings: Input embeddings (UMAP reduced)
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples in neighborhood
            cluster_selection_epsilon: Cluster selection epsilon
            alpha: Alpha parameter for outlier detection
            
        Returns:
            Cluster labels
        """
        # Apply HDBSCAN
        self.hdbscan_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            alpha=alpha,
            prediction_data=True
        )
        
        cluster_labels = self.hdbscan_clusterer.fit_predict(embeddings)
        
        return cluster_labels
    
    def get_clustering_statistics(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """
        Get clustering statistics
        
        Args:
            cluster_labels: Cluster labels from HDBSCAN
            
        Returns:
            Dictionary with clustering statistics
        """
        # Basic statistics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        n_clustered = len(cluster_labels) - n_noise
        total_points = len(cluster_labels)
        
        # Cluster sizes
        cluster_sizes = {}
        for label in set(cluster_labels):
            if label != -1:
                cluster_sizes[label] = list(cluster_labels).count(label)
        
        # Calculate metrics if we have clusters
        metrics = {}
        if n_clusters > 1 and self.umap_reducer is not None:
            try:
                # Get UMAP embeddings for metrics calculation
                umap_embeddings = self.umap_reducer.embedding_
                
                # Filter out noise points for metrics
                clustered_mask = cluster_labels != -1
                if np.sum(clustered_mask) > 1:
                    clustered_embeddings = umap_embeddings[clustered_mask]
                    clustered_labels = cluster_labels[clustered_mask]
                    
                    if len(set(clustered_labels)) > 1:
                        metrics['silhouette_score'] = silhouette_score(
                            clustered_embeddings, clustered_labels
                        )
                        metrics['calinski_harabasz_score'] = calinski_harabasz_score(
                            clustered_embeddings, clustered_labels
                        )
                        metrics['davies_bouldin_score'] = davies_bouldin_score(
                            clustered_embeddings, clustered_labels
                        )
            except Exception as e:
                st.warning(f"Could not calculate clustering metrics: {str(e)}")
        
        statistics = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'n_clustered': n_clustered,
            'total_points': total_points,
            'clustering_ratio': n_clustered / total_points if total_points > 0 else 0,
            'cluster_sizes': cluster_sizes,
            'avg_cluster_size': np.mean(list(cluster_sizes.values())) if cluster_sizes else 0,
            'min_cluster_size': min(cluster_sizes.values()) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes.values()) if cluster_sizes else 0,
            'metrics': metrics
        }
        
        return statistics
    
    def predict_clusters(self, new_embeddings: np.ndarray) -> np.ndarray:
        """
        Predict clusters for new embeddings
        
        Args:
            new_embeddings: New embeddings to cluster
            
        Returns:
            Predicted cluster labels
        """
        if self.hdbscan_clusterer is None:
            raise ValueError("HDBSCAN model not fitted. Run clustering first.")
        
        # Scale embeddings
        new_embeddings_scaled = self.scaler.transform(new_embeddings)
        
        # Apply UMAP transformation
        if self.umap_reducer is not None:
            new_umap_embeddings = self.umap_reducer.transform(new_embeddings_scaled)
        else:
            new_umap_embeddings = new_embeddings_scaled
        
        # Predict clusters
        cluster_labels, strengths = hdbscan.approximate_predict(
            self.hdbscan_clusterer, new_umap_embeddings
        )
        
        return cluster_labels
    
    def get_outlier_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get outlier scores for embeddings
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Outlier scores
        """
        if self.hdbscan_clusterer is None:
            raise ValueError("HDBSCAN model not fitted. Run clustering first.")
        
        # Scale embeddings
        embeddings_scaled = self.scaler.transform(embeddings)
        
        # Apply UMAP transformation
        if self.umap_reducer is not None:
            umap_embeddings = self.umap_reducer.transform(embeddings_scaled)
        else:
            umap_embeddings = embeddings_scaled
        
        # Get outlier scores
        outlier_scores = hdbscan.outlier_scores(self.hdbscan_clusterer)
        
        return outlier_scores
    
    def get_cluster_hierarchies(self) -> Optional[Dict[str, Any]]:
        """
        Get cluster hierarchy information
        
        Returns:
            Dictionary with hierarchy information or None
        """
        if self.hdbscan_clusterer is None:
            return None
        
        try:
            # Get condensed tree
            condensed_tree = self.hdbscan_clusterer.condensed_tree_
            
            # Get cluster tree
            cluster_tree = self.hdbscan_clusterer.single_linkage_tree_
            
            hierarchy_info = {
                'n_clusters': len(self.hdbscan_clusterer.labels_),
                'cluster_persistence': self.hdbscan_clusterer.cluster_persistence_,
                'condensed_tree_size': len(condensed_tree._raw_tree),
                'single_linkage_tree_size': len(cluster_tree._raw_tree)
            }
            
            return hierarchy_info
            
        except Exception as e:
            st.warning(f"Could not extract hierarchy information: {str(e)}")
            return None
    
    def optimize_parameters(
        self, 
        embeddings: np.ndarray, 
        param_grid: Dict[str, List] = None
    ) -> Dict[str, Any]:
        """
        Optimize clustering parameters using grid search
        
        Args:
            embeddings: Input embeddings
            param_grid: Parameter grid for optimization
            
        Returns:
            Dictionary with best parameters and scores
        """
        if param_grid is None:
            param_grid = {
                'min_cluster_size': [3, 5, 10, 15],
                'min_samples': [1, 2, 3, 5]
            }
        
        best_score = -1
        best_params = {}
        best_labels = None
        
        results = []
        
        # Grid search
        for min_cluster_size in param_grid['min_cluster_size']:
            for min_samples in param_grid['min_samples']:
                try:
                    # Apply clustering
                    labels = self.apply_hdbscan(
                        embeddings, 
                        min_cluster_size=min_cluster_size, 
                        min_samples=min_samples
                    )
                    
                    # Calculate score (silhouette score)
                    if len(set(labels)) > 1 and -1 in labels:
                        # Filter out noise for scoring
                        clustered_mask = labels != -1
                        if np.sum(clustered_mask) > 1:
                            clustered_embeddings = embeddings[clustered_mask]
                            clustered_labels = labels[clustered_mask]
                            
                            if len(set(clustered_labels)) > 1:
                                score = silhouette_score(clustered_embeddings, clustered_labels)
                                
                                results.append({
                                    'min_cluster_size': min_cluster_size,
                                    'min_samples': min_samples,
                                    'silhouette_score': score,
                                    'n_clusters': len(set(labels)) - 1,
                                    'n_noise': list(labels).count(-1)
                                })
                                
                                if score > best_score:
                                    best_score = score
                                    best_params = {
                                        'min_cluster_size': min_cluster_size,
                                        'min_samples': min_samples
                                    }
                                    best_labels = labels.copy()
                
                except Exception as e:
                    continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_labels': best_labels,
            'all_results': results
        } 