import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import seaborn as sns
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import umap
import copy

__all__ = ['BaseClusteringModel', 'UMAPClusteringModel', 'GMMClusteringModel', 
           'ClusteringEvaluator', 'ClusteringVisualizer', 
           'ModelRegistry', 'ClusteringPipeline', 'PCATransformer', 'radar_chart']




class BaseClusteringModel(ABC):
    """Abstract base class for clustering models."""
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseClusteringModel':
        """Fit the clustering model to data."""
        pass
        
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict cluster labels for data."""
        pass
        
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        pass

class UMAPClusteringModel:
    """
    UMAP Clustering Model for dimensionality reduction and visualization.

    This class provides an interface to generate UMAP embeddings over a grid
    of parameters (n_neighbors and min_dist) and displays them in a subplot grid.
    """

    def __init__(self, n_neighbors_options=None, min_dist_options=None, sample_size=5000, 
                 random_state=42, figsize=(20, 20), title='UMAP Parameter Grid'):
        # Set default parameter options if not provided
        if n_neighbors_options is None:
            n_neighbors_options = [5, 10, 25, 50, 100]
        if min_dist_options is None:
            min_dist_options = [0.01, 0.025, 0.05, 0.1, 0.25]

        self.n_neighbors_options = n_neighbors_options
        self.min_dist_options = min_dist_options
        self.sample_size = sample_size
        self.random_state = random_state
        self.figsize = figsize
        self.title = title

    def plot_umap_parameter_grid(self, scaled_data, features, 
                                 n_neighbors_options=None, min_dist_options=None, 
                                 sample_size=None, random_state=None, 
                                 figsize=None, title=None):
        """
        Plot a grid of UMAP embeddings with different parameter combinations.
        """
        # Use parameters provided to the function or fall back to instance attributes
        n_neighbors_options = n_neighbors_options or self.n_neighbors_options
        min_dist_options = min_dist_options or self.min_dist_options
        sample_size = sample_size or self.sample_size
        random_state = random_state or self.random_state
        figsize = figsize or self.figsize
        title = title or self.title

        # Use only the specified features
        umap_data = scaled_data[features]

        # Sample the data if needed
        if sample_size and sample_size < len(umap_data):
            umap_data = umap_data.sample(sample_size, random_state=random_state)

        # Create a grid of subplots
        n_rows = len(n_neighbors_options)
        n_cols = len(min_dist_options)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        # Dictionary to store embeddings
        embeddings = {}

        # Iterate through all combinations of parameters
        for i, n_neighbors in enumerate(n_neighbors_options):
            for j, min_dist in enumerate(min_dist_options):
                # Configure UMAP for this combination
                umap_transformer = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric='euclidean',
                    random_state=random_state
                )

                # Fit and transform the data
                embedding = umap_transformer.fit_transform(umap_data)

                # Store embedding in dictionary
                embeddings[(n_neighbors, min_dist)] = embedding

                # Plot on the corresponding subplot
                ax = axes[i, j]
                ax.scatter(embedding[:, 0], embedding[:, 1], s=3, alpha=0.5)
                ax.set_title(f'n={n_neighbors}, d={min_dist}', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

        # Add row and column labels
        for i, n_neighbors in enumerate(n_neighbors_options):
            axes[i, 0].set_ylabel(f'n_neighbors={n_neighbors}', fontsize=12)
        for j, min_dist in enumerate(min_dist_options):
            axes[0, j].set_xlabel(f'min_dist={min_dist}', fontsize=12)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

        return embeddings
    
class GMMClusteringModel(BaseClusteringModel):
    """Gaussian Mixture Model clustering implementation."""
    
    def __init__(self, n_components: int, random_state: int = 42, **kwargs):
        self.n_components = n_components
        self.random_state = random_state
        self.model_params = kwargs
        self.model = None
        self.labels_ = None
        
    def fit(self, data: pd.DataFrame) -> 'GMMClusteringModel':
        """Fit GMM to data."""
        self.model = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
            **self.model_params
        )
        self.model.fit(data)
        self.labels_ = self.model.predict(data)
        return self
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict cluster labels for data."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
        return self.model.predict(data)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
        return {
            'n_components': self.n_components,
            'random_state': self.random_state,
            **self.model_params
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
        
        return {
            'model_type': 'GMM',
            'n_components': self.n_components,
            'means': self.model.means_,
            'weights': self.model.weights_,
            'covariances': self.model.covariances_,
            'converged': self.model.converged_,
            'n_iter': self.model.n_iter_,
            'log_likelihood': self.model.lower_bound_,
            'cluster_sizes': np.bincount(self.labels_, minlength=self.n_components)
        }

class ClusteringEvaluator:
    """Evaluates clustering models with consistent metrics."""
    
    def evaluate(self, model: BaseClusteringModel, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate common clustering metrics.
        """
        labels = model.predict(data)
        
        # Handle the case where all points might be noise in HDBSCAN
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
            return {
                'silhouette_score': float('nan'),
                'calinski_harabasz_score': float('nan'),
                'davies_bouldin_score': float('nan'),
                'n_clusters': 0 if len(unique_labels) <= 1 else 1,
                'noise_percentage': np.mean(labels == -1) * 100 if -1 in labels else 0
            }
        
        # For silhouette score calculation, exclude noise points (-1 labels)
        if -1 in labels:
            valid_indices = labels != -1
            if np.sum(valid_indices) > 1:
                silhouette = silhouette_score(data.iloc[valid_indices], labels[valid_indices])
            else:
                silhouette = float('nan')
            noise_percentage = np.mean(labels == -1) * 100
        else:
            silhouette = silhouette_score(data, labels)
            noise_percentage = 0
            
        # For other metrics that don't handle noise well, also filter
        if -1 in labels and np.sum(labels != -1) > 1:
            valid_indices = labels != -1
            ch_score = calinski_harabasz_score(data.iloc[valid_indices], labels[valid_indices])
            db_score = davies_bouldin_score(data.iloc[valid_indices], labels[valid_indices])
        else:
            ch_score = calinski_harabasz_score(data, labels)
            db_score = davies_bouldin_score(data, labels)
            
        metrics = {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': ch_score,
            'davies_bouldin_score': db_score,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'noise_percentage': noise_percentage
        }
        
        # Add model-specific metrics
        if isinstance(model, GMMClusteringModel) and hasattr(model.model, 'bic'):
            metrics['bic'] = model.model.bic(data)
            
        if isinstance(model, GMMClusteringModel) and hasattr(model.model, 'aic'):
            metrics['aic'] = model.model.aic(data)
            
        return metrics

class ClusteringVisualizer:
    """Visualization tools for clustering results."""
    
    def plot_metrics(self, metrics_dict: Dict[int, Dict[str, float]], 
                    metric_names: List[str] = None,
                    title: str = '',
                    figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot multiple metrics across parameter values.
        """
        if not metrics_dict:
            raise ValueError("No metrics data provided")
            
        # Get parameter values (e.g., number of clusters/components)
        param_values = sorted(metrics_dict.keys())
        
        # If no metric names specified, use all metrics from the first entry
        if metric_names is None:
            metric_names = list(metrics_dict[param_values[0]].keys())
        
        # Separate metrics that should be maximized from those that should be minimized
        maximize_metrics = ['silhouette_score', 'calinski_harabasz_score']
        minimize_metrics = ['davies_bouldin_score', 'bic', 'aic', 'inertia']
        
        # Separate metrics into two groups
        max_metrics = [m for m in metric_names if m in maximize_metrics]
        min_metrics = [m for m in metric_names if m in minimize_metrics]
        
        # Plot metrics to maximize
        if max_metrics:
            plt.figure(figsize=figsize)
            for metric in max_metrics:
                if all(metric in metrics_dict[p] for p in param_values):
                    values = [metrics_dict[p][metric] for p in param_values]
                    plt.plot(param_values, values, marker='o', label=metric)
            
            plt.xlabel('Number of Clusters/Components')
            plt.ylabel('Score (Higher is Better)')
            plt.title(f'{title} Metrics to Maximize')
            plt.grid(True)
            plt.legend()
            plt.show()
        
        # Plot metrics to minimize
        if min_metrics:
            plt.figure(figsize=figsize)
            for metric in min_metrics:
                if all(metric in metrics_dict[p] for p in param_values):
                    values = [metrics_dict[p][metric] for p in param_values]
                    plt.plot(param_values, values, marker='x', label=metric)
            
            plt.xlabel('Number of Clusters/Components')
            plt.ylabel('Score (Lower is Better)')
            plt.title(f'{title} Metrics to Minimize')
            plt.grid(True)
            plt.legend()
            plt.show()
    
    def plot_cluster_profiles(self, model: BaseClusteringModel, data: pd.DataFrame, **kwargs):
        """
        Plot cluster profiles showing feature distributions by cluster.
        """
        labels = model.predict(data)
        
        # Add cluster labels to data
        labeled_data = data.copy()
        labeled_data['cluster'] = labels
        
        # Get number of clusters (excluding noise points)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters == 0:
            print("No clusters found, all points are classified as noise.")
            return
            
        # Create consistent color palette (with darker color for noise points if present)
        if -1 in labels:
            colors = sns.color_palette("colorblind", n_clusters)
            colors = [(0.2, 0.2, 0.2)] + list(colors)  # Dark gray for noise
        else:
            colors = sns.color_palette("colorblind", n_clusters)
            
        # Plot cluster sizes
        plt.figure(figsize=(15, 6))
        
        if -1 in labels:
            # Include noise points in a separate bar
            cluster_indices = np.concatenate([[-1], np.arange(n_clusters)])
            cluster_sizes = np.array([np.sum(labels == -1)] + [np.sum(labels == i) for i in range(n_clusters)])
            labels_text = ['Noise'] + [f'Cluster {i}' for i in range(n_clusters)]
        else:
            cluster_indices = np.arange(n_clusters)
            cluster_sizes = np.array([np.sum(labels == i) for i in range(n_clusters)])
            labels_text = [f'Cluster {i}' for i in range(n_clusters)]
            
        plt.bar(range(len(cluster_sizes)), cluster_sizes, color=colors)
        plt.title(f'Cluster Sizes (n={n_clusters} clusters)')
        plt.xlabel('Cluster')
        plt.ylabel('Size')
        plt.xticks(range(len(cluster_sizes)), labels_text)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        
        # Get model information for additional context
        model_info = model.get_model_info()
        model_type = model_info.get('model_type', 'Unknown')

        # Plot feature distributions by cluster
        # Only include non-noise points for the feature plots if desired
        if -1 in labels and kwargs.get('exclude_noise', True):
            plot_data = labeled_data[labeled_data['cluster'] != -1].copy()
        else:
            plot_data = labeled_data.copy()
            
        # Skip if no valid clusters
        if len(plot_data) == 0:
            return
            
        feature_names = data.columns
        melted_data = plot_data.melt(
            id_vars='cluster', 
            value_vars=feature_names, 
            var_name='Feature', 
            value_name='Value'
        )
        
        plt.figure(figsize=(15, 10))
        sns.violinplot(
            x='Feature', 
            y='Value', 
            hue='cluster', 
            data=melted_data, 
            split=False, 
            palette=colors[1:] if -1 in labels and kwargs.get('exclude_noise', True) else colors,
            inner=None
        )
        plt.title(f'{model_type} Cluster Feature Distributions')
        plt.xlabel('Features')
        plt.ylabel('Value')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

class ModelRegistry:
    """Stores, retrieves, and compares multiple clustering models."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def register(self, model_id: str, model: BaseClusteringModel, 
                data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Register a model and its evaluation results."""
        self.models[model_id] = model
        self.results[model_id] = results
        
    def get_model(self, model_id: str) -> Optional[BaseClusteringModel]:
        """Retrieve a model by ID."""
        return self.models.get(model_id)
    
    def get_results(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve model results by ID."""
        return self.results.get(model_id)
    
    def list_models(self) -> List[str]:
        """List all registered model IDs."""
        return list(self.models.keys())
    
    def compare_models(self, metric_name: str) -> pd.DataFrame:
        """
        Compare models based on a specific metric.
        """
        comparison = []
        for model_id, results in self.results.items():
            if metric_name in results:
                comparison.append({
                    'model_id': model_id,
                    metric_name: results[metric_name]
                })
        
        return pd.DataFrame(comparison).sort_values(by=metric_name, ascending=False)

class ClusteringPipeline:
    """Orchestrates the end-to-end clustering workflow."""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.evaluator = ClusteringEvaluator()
        self.visualizer = ClusteringVisualizer()
        
    def run_experiment(self, model_class: type, param_grid: Dict[str, List], 
                      data: pd.DataFrame, param_name: str = 'n_components',
                      base_model_id: str = 'model') -> Dict[int, str]:
        """
        Run an experiment with a model class and parameter grid.
        """
        # Track metrics across parameter values
        all_metrics = {}
        model_ids = {}
        
        # Extract the parameter values to vary
        param_values = param_grid.pop(param_name, [])
        if not param_values:
            raise ValueError(f"No values provided for parameter '{param_name}'")
        
        # Run experiments for each parameter value
        for param_value in param_values:
            # Create model ID
            model_id = f"{base_model_id}_{param_name}_{param_value}"
            model_ids[param_value] = model_id
            
            # Create and fit model
            # Extract the first value from each list in param_grid
            fixed_params = {k: v[0] if isinstance(v, list) else v for k, v in param_grid.items()}
            model_params = {param_name: param_value, **fixed_params}
            model = model_class(**model_params)
            model.fit(data)
            
            # Evaluate model
            metrics = self.evaluator.evaluate(model, data)
            all_metrics[param_value] = metrics
            
            # Register model and results
            self.registry.register(model_id, model, data, metrics)
        
        # Visualize metrics
        self.visualizer.plot_metrics(all_metrics)
        
        return model_ids
    
    def examine_model(self, model_id: str, data: pd.DataFrame) -> None:
        """
        Examine a specific model in detail.
        """
        model = self.registry.get_model(model_id)
        if model is None:
            raise ValueError(f"Model '{model_id}' not found in registry")
            
        # Get evaluation results
        results = self.registry.get_results(model_id)
        
        # Print metrics
        print(f"Model: {model_id}")
        print("Metrics:")
        for metric, value in results.items():
            print(f"  {metric}: {value}")
        
        # Visualize cluster profiles
        self.visualizer.plot_cluster_profiles(model, data)

class PCATransformer:
    def __init__(self, data: pd.DataFrame, feature_weights=None):
        self.data = data
        self.feature_weights = feature_weights if feature_weights else {}
        self.pca = None
        self.transformed_data = None
        self.scaler = None

    def fit_transform(self, n_components=8, drop_missing=True, scale=True):
        """Fit PCA to the data."""
        # Handle missing values
        if drop_missing:
            self.data = self.data.dropna()
        
        # Standardize the data if requested
        if scale:
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(self.data)
        else:
            scaled_data = self.data.values

        # Fit PCA
        self.pca = PCA(n_components=n_components)
        self.transformed_data = self.pca.fit_transform(scaled_data)
        return self.transformed_data

    def transform(self, data, scale=True):
        """Transform new data using the fitted PCA."""
        if self.pca is None:
            raise ValueError("PCA has not been fitted yet")
        
        # Scale the data if requested and a scaler exists
        if scale and self.scaler is not None:
            data = self.scaler.transform(data)
            
        return self.pca.transform(data)

@ staticmethod
def radar_chart(df, title):
    """
    Create a radar chart to visualize cluster profiles.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with clusters as index and features as columns, typically created with df.groupby('cluster').mean()
    title : str
        Title for the radar chart
        
    Returns:
    --------
    fig, ax : tuple
        Matplotlib figure and axes objects
    """
    # Number of variables
    categories = list(df.columns)
    N = len(categories)
    
    # Create angles for each feature
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], size=10)
    plt.ylim(0, 1)
    
    # Plot data
    for i, cluster in enumerate(df.index):
        values = df.loc[cluster].values.tolist()
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Cluster {cluster}")
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title, size=15, pad=20)
    
    return fig, ax