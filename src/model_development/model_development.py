import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class ModelTrainer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.gmm_model = None
        self.kmeans_model = None
        self.n_components = None

    def fit_gmm(self, n_components: int):
        """Fit a Gaussian Mixture Model to the data."""
        self.n_components = n_components
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(self.data)
        self.gmm_model = gmm
        return gmm

    def fit_kmeans(self, n_clusters: int):
        """Fit a K-Means model using the GMM centroids."""
        if self.gmm_model is None:
            raise ValueError("GMM model must be fitted before K-Means.")
        
        kmeans = KMeans(n_clusters=n_clusters, init=self.gmm_model.means_, random_state=42)
        kmeans.fit(self.data)
        self.kmeans_model = kmeans
        return kmeans

class ModelOptimizer:
    def __init__(self, gmm_model: GaussianMixture):
        self.gmm_model = gmm_model

    def assess_feature_importance(self):
        """Assess feature importance based on GMM covariance matrices."""
        covariances = self.gmm_model.covariances_
        # Analyze covariances to determine feature importance
        # (Implementation details can be added here)
        return covariances

class PCATransformer:
    def __init__(self, data: pd.DataFrame, feature_weights=None):
        self.data = data
        self.feature_weights = feature_weights if feature_weights else {}
        self.pca = None
        self.transformed_data = None

    def fit_transform(self, n_components=8, drop_missing=True):
        """Fit PCA to the data."""
        # Handle missing values and apply weights
        if drop_missing:
            self.data = self.data.dropna()
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)

        # Fit PCA
        self.pca = PCA(n_components=n_components)
        self.transformed_data = self.pca.fit_transform(scaled_data)
        return self.transformed_data

    def plot_cumulative_variance(self):
        """Plot cumulative explained variance ratio."""
        if self.pca is None:
            raise ValueError("PCA has not been fitted yet.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_), marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance vs Number of Components')
        plt.grid()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Load your data
    data = pd.read_csv('your_data.csv')  # Replace with your actual data source

    # Initialize and fit models
    trainer = ModelTrainer(data)
    gmm = trainer.fit_gmm(n_components=3)  # Example number of components
    kmeans = trainer.fit_kmeans(n_clusters=3)  # Example number of clusters

    # Optimize and assess feature importance
    optimizer = ModelOptimizer(gmm)
    feature_importance = optimizer.assess_feature_importance()

    # Apply PCA
    pca_transformer = PCATransformer(data)
    pca_data = pca_transformer.fit_transform(n_components=2)  # Example PCA components
    pca_transformer.plot_cumulative_variance()