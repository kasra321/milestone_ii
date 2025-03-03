# MIMICDF Project Reproduction Guide

## Overview
This guide outlines the steps to reproduce the clustering models in the MIMICDF project, which analyzes MIMIC-IV Emergency Department data.

## Step 1: Data Preparation
First, run the `data_preparation.ipynb` notebook:
- Load demo data using:
- The notebook will:
  - Perform exploratory data analysis
  - Clean the data
  - Engineer features
  - Apply transformations (robust and aggressive)
  - Split data into training/validation sets
  - Export transformed data to `/data/cached` directory

## Step 2: Model Development
Next, run the `model_development.ipynb` notebook:
- The notebook automatically loads the transformed data from `/data/cached`
- The modeling workflow includes:
  - UMAP visualization with parameter exploration
  - GMM clustering with different component counts
  - Model evaluation using silhouette scores and other metrics
  - Cluster profile visualization and interpretation
  - Validation on held-out data

## Key Components
- **Data Source**: Uses local demo data by default
- **Preprocessing**: Handles missing values, outliers, and feature engineering
- **Transformations**: Offers both robust scaling and aggressive transformations
- **Modeling**: Focuses on Gaussian Mixture Models with UMAP visualization
- **Evaluation**: Uses standard clustering metrics and visual inspection

## Output
The notebooks produce visualizations and model artifacts that help understand ED patient clustering patterns.


# MIMICDF: MIMIC-IV ED Data Interface

## Overview
The `mimicdf.py` module provides an interface for accessing MIMIC-IV Emergency Department data through:
- Google BigQuery (GCP) for complete dataset access
- Locally stored data (Default)


## Key Methods

### Core Data
- `edstays()`: Base cohort with admission times and demographics
- `demographics()`: Subject demographics with standardized race categories
- `vitals()`: Vital signs measurements
- `triage()`: Triage assessments including acuity scores
- `diagnosis()`: ED visit diagnosis codes

### Medications
- `pyxis()`: Medication dispensing records
- `medications()`: Combined medication records

### Preprocessed Data
- `ed_data()`: Comprehensive dataset combining demographics, vitals, and time features

## Important Notes
- **GCP Access**: Requires PhysioNet credentialing and proper GCP authentication
- **Demo Data**: local CSV files in `data/demo` directory
- **Memory Management**: Use `clear_cache()` to free memory when needed


# Data Preprocessor Module Documentation

## Overview
The `data_preprocessor.py` module provides a comprehensive toolkit for preparing emergency department (ED) data for analysis and modeling. It contains four main classes that handle different aspects of the data preparation pipeline.

## Classes

### DataExplorer
Performs exploratory data analysis and visualization.

**Key Methods:**
- `summarize_missing_data()`: Analyzes and visualizes missing data patterns
- `analyze_missing_correlations()`: Creates correlation matrix of missingness patterns
- `plot_missingness_distribution()`: Visualizes missing data across arrival modes and dispositions
- `plot_features_distribution()`: Creates violin plots showing feature distributions
- `plot_qq_plot()`: Generates QQ plots to assess normality

### DataCleaner
Handles data cleaning and validation operations.

**Key Methods:**
- `prepare_data()`: Main method that orchestrates the cleaning process
- `_clean_categorical()`: Filters categorical variables to valid values
- `_clean_vitals()`: Applies physiological range checks to vital signs
- `_clean_blood_pressure()`: Special handling for blood pressure validation

### FeatureEngineer
Creates new features from raw data.

**Key Methods:**
- `engineer_features()`: Main method that creates all feature types
- `_add_clinical_metrics()`: Calculates MAP, pulse pressure, shock index, etc.
- `_add_vital_categories()`: Creates categorical features from vital signs
- `_add_demographic_features()`: Adds age groups and other demographic features
- `_add_clinical_scores()`: Calculates clinical scores like SIRS criteria
- `_add_temporal_features()`: Creates cyclical time features

### DataTransformer
Transforms features to improve statistical properties.

**Key Methods:**
- `robust_transformer_fit()`: Applies robust scaling to features
- `aggressive_transformer_fit()`: Applies more aggressive transformations including Box-Cox
- `_boxcox_transform()`: Applies Box-Cox transformation to normalize features
- `_cyclical_transform()`: Converts time features to sine/cosine components
- `_validate_transformations()`: Checks normality of transformed features

## Feature Sets
The module defines several standard feature sets:
- **aggressive_gmm_features**: Features with aggressive transformations
- **robust_gmm_features**: Features with robust scaling
- **pca_features**: Features suitable for PCA
- **metadata**: ID columns and metadata

## Usage in Notebook
The `data_preperation.ipynb` notebook demonstrates the complete workflow:

1. Data loading and initial exploration
2. Missing data analysis
3. Data cleaning
4. Feature engineering
5. Data transformation (both robust and aggressive)
6. Data splitting and export

# Model Development Module Documentation

## Overview
The `model_development.py` module provides a comprehensive framework for developing, evaluating, and visualizing clustering models for emergency department data analysis. It implements a modular, object-oriented approach to clustering analysis.

## Classes

### BaseClusteringModel
Abstract base class that defines the interface for all clustering models.

**Key Methods:**
- `fit()`: Fits the model to input data
- `predict()`: Predicts cluster assignments for new data
- `get_params()`: Returns model parameters
- `get_model_info()`: Provides detailed model information

### UMAPClusteringModel
Implements dimensionality reduction and visualization using UMAP.

**Key Methods:**
- `plot_umap_parameter_grid()`: Creates a grid of UMAP visualizations with different parameters
- `fit_umap()`: Fits UMAP to input data with specified parameters

### GMMClusteringModel
Implements Gaussian Mixture Model clustering.

**Key Methods:**
- `fit()`: Fits GMM to input data
- `predict()`: Assigns cluster labels to new data
- `get_model_info()`: Returns detailed model information including means, weights, and covariances

### ClusteringEvaluator
Evaluates clustering models using standard metrics.

**Key Methods:**
- `evaluate()`: Calculates silhouette score, Calinski-Harabasz index, and Davies-Bouldin index
- `plot_metrics()`: Visualizes evaluation metrics across parameter values

### ClusteringVisualizer
Creates visualizations for clustering results.

**Key Methods:**
- `plot_cluster_profiles()`: Creates violin plots showing feature distributions by cluster
- `plot_cluster_scatter()`: Generates scatter plots of clusters in 2D space
- `plot_cluster_heatmap()`: Creates heatmaps showing cluster centroids

### ModelRegistry
Stores and manages multiple clustering models.

**Key Methods:**
- `register()`: Adds a model and its evaluation results to the registry
- `get_model()`: Retrieves a model by ID
- `compare_models()`: Compares models based on a specific metric

### ClusteringPipeline
Orchestrates the entire clustering workflow.

**Key Methods:**
- `run_experiment()`: Runs a clustering experiment with different parameter values
- `examine_model()`: Provides detailed analysis of a specific model
- `plot_parameter_grid()`: Visualizes model performance across parameter values

### PCATransformer
Performs PCA dimensionality reduction.

**Key Methods:**
- `fit_transform()`: Fits PCA to data and returns transformed result
- `transform()`: Applies fitted PCA to new data

## Utility Functions
- `radar_chart()`: Creates radar charts for visualizing cluster profiles

## Usage in Notebook
The `model_development.ipynb` notebook demonstrates the complete workflow:

1. **Data Preparation:**
   - Loading transformed data from preprocessing pipeline
   - Defining feature sets for different experiments

2. **UMAP Exploration:**
   - Parameter grid search for UMAP visualization
   - Comparison of different feature sets and transformations

3. **GMM Clustering:**
   - Systematic evaluation of GMM models with different numbers of components
   - Comparison of aggressive vs. robust transformations
   - Comparison of full feature set vs. cardiovascular-focused features

4. **Model Evaluation:**
   - Silhouette score comparison across models
   - Detailed examination of selected models

5. **Validation:**
   - Application of selected model to validation data
   - Visualization of cluster assignments in PCA space
   - Analysis of cluster profiles using radar charts

The notebook provides a comprehensive demonstration of how to use the module to develop, evaluate, and interpret clustering models for emergency department data.