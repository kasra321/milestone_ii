# Hospital Admission Prediction Project

This repository contains a machine learning project focused on predicting hospital admissions from emergency department data. The project implements multiple modeling approaches and combines them into an ensemble for improved prediction accuracy.

## Project Structure

### Logistic Regression Analysis
1. [`logistic_regression_model.py`](./logistic_regression_model.py) - Core implementation of the logistic regression model with comprehensive feature processing, model training, and evaluation capabilities.
2. [`logistic_regression_analysis.ipynb`](./logistic_regression_analysis.ipynb) - Detailed analysis of the logistic regression model's performance, including feature importance and parameter tuning.
3. [`logistic_regression_experiment.md`](./logistic_regression_experiment.md) - Documentation of experiments and extensions to the logistic regression model, including:
   - Feature importance analysis
   - Ablation studies
   - Sampling strategy comparison
   - Sensitivity analysis
   - Model retraining recommendations
4. [`failure_analysis.ipynb`](./failure_analysis.ipynb) - In-depth analysis of cases where the logistic regression model fails to make correct predictions.

### Model Experiments and Comparisons
1. [`experiments/xgboost.ipynb`](./experiments/xgboost.ipynb) - Implementation and analysis of XGBoost model using text embeddings from chief complaints.
2. [`experiments/randomforest.ipynb`](./experiments/randomforest.ipynb) - Random Forest model implementation using complexity scores and structured features.
3. [`experiments/ensemble.py`](./experiments/ensemble.py) - Implementation of the ensemble model combining multiple approaches.
4. [`experiments/pca.ipynb`](./experiments/pca.ipynb) - PCA dimensionality reduction experiments for feature optimization.
5. [`experiments/supervised_learning.ipynb`](./experiments/supervised_learning.ipynb) - Comprehensive evaluation of all models, including:
   - Individual model performance metrics
   - ROC curve analysis
   - Feature importance comparisons
   - Ensemble model validation

### Data
The project uses emergency department visit data with various features including:
- Vital signs (temperature, heart rate, blood pressure, etc.)
- Patient demographics
- Chief complaint text
- Acuity scores
- Clinical indicators (SIRS, shock index)

## Model Components

### 1. Logistic Regression
- Feature engineering with both numeric and categorical variables
- Hyperparameter optimization via grid search
- Comprehensive evaluation metrics
- Feature importance analysis
- Sampling strategy optimization for class imbalance

### 2. XGBoost
- Text embedding features from chief complaints
- Advanced feature engineering
- Hyperparameter tuning
- Performance analysis

### 3. Random Forest
- Structured feature utilization
- Complexity score incorporation
- Feature importance analysis
- Cross-validation performance

### 4. Ensemble Model
- Combines predictions from all base models
- Weighted voting mechanism
- Performance comparison with individual models
- Robust evaluation framework

## Utility Modules

The `utils` directory contains essential helper modules that support the main functionality of the project:

### Core Data Processing
- [`data_utils.py`](./utils/data_utils.py) - Core data processing functions including:
  - Data loading and preprocessing
  - Feature scaling and encoding
  - Processing pipelines for different models (XGBoost, Random Forest, Logistic Regression)
  - Caching mechanisms for efficient data handling

### Feature Engineering
- [`embeddings.py`](./utils/embeddings.py) - Text embedding generation using transformer models:
  - Uses BAAI/bge-m3 model for generating embeddings
  - Caching system for efficient embedding storage and retrieval
  - GPU acceleration support
  
- [`complexity.py`](./utils/complexity.py) - Text complexity analysis:
  - Entropy calculation
  - Lexical complexity metrics
  - Medical entity recognition
  - POS tagging complexity

### Model Support
- [`evaluation.py`](./utils/evaluation.py) - Model evaluation utilities:
  - Performance metrics calculation
  - ROC curve generation
  - Confusion matrix visualization
  - Model comparison tools

- [`lr_model.py`](./utils/lr_model.py) - Extended logistic regression implementation:
  - Custom model training
  - Feature importance analysis
  - Model evaluation and visualization

### Helper Utilities
- [`check_gpu.py`](./utils/check_gpu.py) - GPU availability and configuration checking:
  - CUDA availability verification
  - GPU memory monitoring
  - Driver and version compatibility checks

- [`helper_functions.py`](./utils/helper_functions.py) - General utility functions
- [`pain_utils.py`](./utils/pain_utils.py) - Pain score processing utilities

## Requirements

- imblearn >= 0.0
- ipykernel >= 6.29.5
- ipywidgets >= 8.1.5
- joblib >= 1.4.2
- matplotlib >= 3.10.0
- numpy >= 2.2.3
- pandas >= 2.2.3
- pip >= 25.0.1
- psutil >= 7.0.0
- scikit-learn >= 1.6.1
- seaborn >= 0.13.2
- spacy >= 3.8.4
- tqdm >= 4.67.1
- transformers >= 4.49.0
- xgboost >= 2.1.4

## Usage

1. Start with the logistic regression analysis notebooks to understand baseline performance
2. Explore individual model experiments in the `experiments` directory
3. Review the failure analysis to understand model limitations
4. Examine the ensemble model implementation and final results in `supervised_learning.ipynb`

## Results

The ensemble model demonstrates improved performance over individual models, with:
- Higher accuracy and F1 scores
- Better handling of edge cases
- More robust predictions across different patient populations

Detailed performance metrics and comparisons can be found in `experiments/supervised_learning.ipynb`.

## Acknowledgments

This project was developed with assistance from:
- [Windsurf IDE](https://www.codeium.com/windsurf) - Agentic IDE providing intelligent code assistance and pair programming capabilities
- [Claude 3.5 Sonnet](https://www.anthropic.com/claude) - Advanced AI model used for code generation, debugging, and technical documentation
- [Grammarly](https://www.grammarly.com) - Writing assistant used for documentation proof reading, grammar checking, clarity and readability


These tools significantly enhanced development efficiency and code quality through:
- Intelligent code suggestions and completion
- Automated debugging assistance
- Code organization and refactoring recommendations



# Supervised Learning Project Setup

## Initial Setup Instructions

### 1\. Dataset Preparation

First, download and unzip the datasets to a directory named "data" in the project root:

```
# Create data directory if it doesn't exist
mkdir -p data

# unzip data.zip -d data/
```

This will unzip three data files including:

- edstays.csv
- patients.csv
- triage.csv

### 2\. Environment Setup

#### Install uv (Faster Python Package Installer)

```
pip install uv
```

#### Initialize Project with uv

```
# Initialize a new project
uv init

# Create a virtual environment
uv venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
# source .venv/bin/activate
```


#### Install PyTorch
```
uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

#### Install Required Packages

```
# Install dependencies from requirements.txt (if available)
uv pip install -r requirements.txt

# Or install individual packages
# uv pip install pandas numpy scikit-learn matplotlib seaborn
```

#### Install spaCy and Download Language Model

```
# Install spaCy
uv pip install spacy

# Download the English language model
python -m spacy download en_core_web_sm
```

## Running the Project

After completing the setup steps above, you can run the data processing pipeline:

```
# Run the data handler to process the dataset to load, merge, clean, and create
python -m utils.data_handler
```

This will:

1.  Load and merge the datasets
2.  Clean the data
3.  Create features (including complexity features for chief complaints)
4.  Split into training and validation sets
5.  Save the processed datasets to the data directory