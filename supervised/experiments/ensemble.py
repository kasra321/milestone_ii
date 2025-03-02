"""
Instructions to run this script:

1. Run with all rows and tuning step:
    python -m ensemble
2. Run with a subset of rows (1000 in this example) and tuning step:
    python -m ensemble -num_rows 1000
3. Run with all rows and skip tuning step:
    python -m ensemble -no_tune
4. Run with a subset of rows (1000 in this example) and skip tuning step:
    python -m ensemble -num_rows 1000 -no_tune
"""

import sys
import os
import argparse

# Add parent directory to path temporarily for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression

from utils.data_utils import process_data_for_rf, process_data_for_lr, process_data_for_xgboost
from utils.evaluation import evaluate_model
from utils.embeddings import EmbeddingExtractor

# Remove parent directory from path after imports
sys.path.remove(parent_dir)

# Suppress specific sklearn warnings
warnings.filterwarnings('ignore', message='Setting penalty=None will ignore the C and l1_ratio parameters')

def load_data(file_path, num_rows=None):
    """Load data from CSV file with optional row limit"""
    df = pd.read_csv(file_path)
    if num_rows is not None:
        # Randomly sample num_rows if specified
        df = df.sample(n=min(num_rows, len(df)), random_state=42)
    return df


def load_base_models():
    """Load all three base models and their preprocessors"""
    models = {}
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Check if models exist
    if not all(os.path.exists(f) for f in [
        'models/random_forest_model.pkl',
        'models/logistic_regression_model.pkl',
        'models/xgboost_model.pkl'
    ]):
        print("Pre-trained models not found. Please run training first.")
        return None

    # Load Random Forest
    models['rf'] = {
        'model': joblib.load('models/random_forest_model.pkl'),
        'scaler': joblib.load('models/rf_scaler.pkl'),
        'encoder': joblib.load('models/rf_encoder.pkl'),
        'process_func': process_data_for_rf
    }

    # Load Logistic Regression
    models['lr'] = {
        'model': joblib.load('models/logistic_regression_model.pkl'),
        'scaler': joblib.load('models/lr_scaler.pkl'),
        'preprocessor': joblib.load('models/lr_preprocessor.pkl'),
        'encoder': joblib.load('models/lr_encoder.pkl'),
        'process_func': process_data_for_lr
    }

    # Load XGBoost
    models['xgb'] = {
        'model': joblib.load('models/xgboost_model.pkl'),
        'scaler': joblib.load('models/xgb_scaler.pkl'),
        'encoder': joblib.load('models/xgb_encoder.pkl'),
        'pca': joblib.load('models/xgb_pca.pkl'),
        'process_func': process_data_for_xgboost
    }

    return models


def process_data(df, models):
    """Process data for all models"""

    data = {}

    df_subset = df[['disposition', 'chiefcomplaint']]

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        embedder = EmbeddingExtractor()
        embeddings = embedder.get_or_create_embeddings(df_subset, column="chiefcomplaint")
        pca_obj = models['xgb']['pca']
        xgb_obj = process_data_for_xgboost(embeddings, df['disposition'], pca_obj=pca_obj, training=False)
        data['xgb_X'] = xgb_obj['X']
        data['xgb_y'] = xgb_obj['y']

    # Process data for Random Forest
    rf_obj = process_data_for_rf(df_subset)
    data['rf_X'] = rf_obj['X']
    data['rf_y'] = rf_obj['y']

    # Process data for LR
    preprocessor = models['lr']['preprocessor']
    lr_obj = process_data_for_lr(df, preprocessor=preprocessor, mode='inference')
    data['lr_X'] = lr_obj['X']
    data['lr_y'] = lr_obj['y']

    return data


def get_model_predictions(model_dict, X):
    """Get probability predictions from a model"""
    X_processed = model_dict['process_func'](X, model_dict['scaler'], model_dict['encoder'])
    return model_dict['model'].predict_proba(X_processed)[:, 1]


def create_meta_features(models, X):
    """Create meta-features by getting predictions from all base models"""
    meta_features = np.column_stack([
        get_model_predictions(models[name], X) for name in models.keys()
    ])
    return meta_features


def ensure_features(X, model, model_name):
    """
    Ensure that X has all the features expected by the model.
    
    Args:
        X (pd.DataFrame): Input features
        model: Trained model with feature_names_in_ attribute
        model_name (str): Name of the model for logging
        
    Returns:
        pd.DataFrame: DataFrame with all required features in the correct order
    """
    # Check if model has feature_names_in_ attribute (scikit-learn models)
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_

        # Find missing features
        missing_features = set(expected_features) - set(X.columns)
        if missing_features:
            print(f"Adding {len(missing_features)} missing features for {model_name} model")
            # Add missing features with zeros
            for feature in missing_features:
                X[feature] = 0

        # Ensure columns are in the same order as during training
        return X[expected_features]
    return X


def get_all_predictions(models, data):
    """
    Get predictions and probabilities from all models.

    Args:
        models: Dictionary containing all models (from load_base_models)
        data: Dictionary containing processed data (from process_data)

    Returns:
        Dictionary containing predictions and probabilities for all models
    """
    results = {}

    # Get predictions and probabilities for Logistic Regression
    lr_model = models['lr']['model']
    lr_X = data['lr_X']
    lr_X = ensure_features(lr_X, lr_model, 'Logistic Regression')
    results['lr_predictions'] = lr_model.predict(lr_X)
    results['lr_probabilities'] = lr_model.predict_proba(lr_X)[:, 1]

    # Get predictions and probabilities for XGBoost
    xgb_model = models['xgb']['model']
    xgb_X = data['xgb_X']
    results['xgb_predictions'] = xgb_model.predict(xgb_X)
    results['xgb_probabilities'] = xgb_model.predict_proba(xgb_X)[:, 1]

    # Get predictions and probabilities for Random Forest
    rf_model = models['rf']['model']
    rf_X = data['rf_X']
    results['rf_predictions'] = rf_model.predict(rf_X)
    results['rf_probabilities'] = rf_model.predict_proba(rf_X)[:, 1]

    return results


def train_ensemble(df):
    """Train the ensemble model"""
    # Load base models
    base_models = load_base_models()

    # Process data
    data = process_data(df, base_models)

    # Get predictions from base models
    results = get_all_predictions(base_models, data)

    # Create meta-features for training
    meta_features = np.column_stack([
        results['rf_probabilities'],
        results['lr_probabilities'],
        results['xgb_probabilities'],
        # Add squared terms to allow for non-linear relationships
        results['rf_probabilities']**2,
        results['lr_probabilities']**2,
        results['xgb_probabilities']**2,
        # Add interaction terms
        results['rf_probabilities'] * results['lr_probabilities'],
        results['rf_probabilities'] * results['xgb_probabilities'],
        results['lr_probabilities'] * results['xgb_probabilities']
    ])

    # Get target variable (should be the same across all models)
    y = data['lr_y']

    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        meta_features, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train meta-learner (Logistic Regression)
    meta_learner = LogisticRegression(random_state=42, max_iter=1000)
    meta_learner.fit(X_train, y_train)

    # Evaluate ensemble
    y_pred = meta_learner.predict(X_val)
    y_prob = meta_learner.predict_proba(X_val)[:, 1]

    # Print evaluation metrics
    print("\nEnsemble Model Evaluation:")
    metrics, _ = evaluate_model(y_val, y_pred, y_prob=y_prob, model_name="Ensemble")

    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation F1 Score: {metrics['f1']:.4f}")
    print(f"Validation AUC-ROC: {metrics['auc_roc']:.4f}")

    return meta_learner, base_models


def predict_ensemble(meta_learner, base_models, X):
    """Make predictions using the ensemble"""
    meta_features = create_meta_features(base_models, X)
    return meta_learner.predict(meta_features), meta_learner.predict_proba(meta_features)[:, 1]


def tune_lr(df, n_splits=5, verbose=True):
    """
    Tune a logistic regression ensemble model using k-fold cross-validation.

    Args:
        df (pd.DataFrame): Input DataFrame with features and target
        n_splits (int): Number of folds for cross-validation
        verbose (bool): Whether to print progress and results

    Returns:
        LogisticRegression: Best tuned logistic regression model
        dict: Cross-validation results
        dict: Base models used for ensemble
        dict: Ensemble evaluation metrics
        str: Ensemble evaluation markdown report
    """
    if verbose:
        print("Loading base models...")

    # Load base models
    base_models = load_base_models()

    if verbose:
        print("Processing data...")

    # Split data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['disposition'])
    
    if verbose:
        print(f"Training data: {train_df.shape[0]} samples")
        print(f"Validation data: {val_df.shape[0]} samples")
    
    # Process training data
    data = process_data(train_df, base_models)

    # Get target variable (1 = ADMIT, 0 = HOME)
    y = data['lr_y']  # All processing functions should encode the target the same way

    if verbose:
        print("Creating meta-features from base model predictions...")

    # Get predictions from base models
    results = get_all_predictions(base_models, data)

    # Create meta-features from base model predictions
    meta_features = np.column_stack([
        results['rf_probabilities'],
        results['lr_probabilities'],
        results['xgb_probabilities'],
        # Add squared terms to allow for non-linear relationships
        results['rf_probabilities']**2,
        results['lr_probabilities']**2,
        results['xgb_probabilities']**2,
        # Add interaction terms
        results['rf_probabilities'] * results['lr_probabilities'],
        results['rf_probabilities'] * results['xgb_probabilities'],
        results['lr_probabilities'] * results['xgb_probabilities']
    ])

    # Define parameter grid for logistic regression
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'lbfgs'],
        'class_weight': ['balanced', None]
    }

    # Create valid parameter combinations manually instead of using GridSearchCV
    param_grid_filtered = []

    for C in param_grid['C']:
        for penalty in param_grid['penalty']:
            for solver in param_grid['solver']:
                for class_weight in param_grid['class_weight']:
                    # Skip incompatible combinations
                    if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                        continue
                    if penalty is None and solver == 'liblinear':
                        continue

                    # For other penalties, l1_ratio is not needed
                    param_grid_filtered.append({
                        'C': C,
                        'penalty': penalty,
                        'solver': solver,
                        'class_weight': class_weight
                    })

    print(f"Starting {n_splits}-fold cross-validation with {len(param_grid_filtered)} parameter combinations...")

    # Initialize cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize variables to track best model
    best_score = 0
    best_model = None
    best_params = None
    cv_results = {
        'params': [],
        'mean_accuracy': [],
        'mean_f1': [],
        'mean_auc': [],
        'std_accuracy': [],
        'std_f1': [],
        'std_auc': []
    }

    # Perform grid search with cross-validation
    for params in param_grid_filtered:
        if verbose:
            print(f"Evaluating parameters: {params}")

        # Initialize metrics for this parameter set
        fold_accuracies = []
        fold_f1_scores = []
        fold_auc_scores = []

        # Perform k-fold cross-validation
        for train_idx, val_idx in kf.split(meta_features):
            # Split data
            X_train, X_val = meta_features[train_idx], meta_features[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Initialize and fit model with current parameters
            try:
                model = LogisticRegression(random_state=42, max_iter=1000, **params)
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_val)
                y_prob = model.predict_proba(X_val)[:, 1]

                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred)
                auc = roc_auc_score(y_val, y_prob)

                # Store metrics
                fold_accuracies.append(accuracy)
                fold_f1_scores.append(f1)
                fold_auc_scores.append(auc)
            except Exception as e:
                if verbose:
                    print(f"Error with parameters {params}: {str(e)}")
                continue

        # Calculate mean and std of metrics across folds
        if fold_accuracies:  # Only if we have valid results
            mean_accuracy = np.mean(fold_accuracies)
            mean_f1 = np.mean(fold_f1_scores)
            mean_auc = np.mean(fold_auc_scores)
            std_accuracy = np.std(fold_accuracies)
            std_f1 = np.std(fold_f1_scores)
            std_auc = np.std(fold_auc_scores)

            # Store results
            cv_results['params'].append(params)
            cv_results['mean_accuracy'].append(mean_accuracy)
            cv_results['mean_f1'].append(mean_f1)
            cv_results['mean_auc'].append(mean_auc)
            cv_results['std_accuracy'].append(std_accuracy)
            cv_results['std_f1'].append(std_f1)
            cv_results['std_auc'].append(std_auc)

            # Update best model if this one is better (using AUC as primary metric)
            if mean_auc > best_score:
                best_score = mean_auc
                best_params = params


            print(f"New best model found! AUC: {mean_auc:.4f}, Accuracy: {mean_accuracy:.4f}, F1: {mean_f1:.4f}")
            print(f"Parameters: {params}")

    if verbose:
        print("\nCross-validation complete!")
        print(f"Best parameters: {best_params}")
        print(f"Best AUC score: {best_score:.4f}")

    # Initialize variables for ensemble evaluation
    ensemble_metrics = None
    ensemble_markdown = None

    # Train final model on all data with best parameters
    if best_params:
        best_model = LogisticRegression(random_state=42, max_iter=1000, **best_params)
        best_model.fit(meta_features, y)

        if verbose:
            print("\nFinal model trained on all data with best parameters.")

            # Process validation data
            val_data = process_data(val_df, base_models)
            
            # Get predictions from base models on validation data
            val_results = get_all_predictions(base_models, val_data)
            
            # Create meta-features for validation data
            val_meta_features = np.column_stack([
                val_results['rf_probabilities'],
                val_results['lr_probabilities'],
                val_results['xgb_probabilities'],
                # Add squared terms to allow for non-linear relationships
                val_results['rf_probabilities']**2,
                val_results['lr_probabilities']**2,
                val_results['xgb_probabilities']**2,
                # Add interaction terms
                val_results['rf_probabilities'] * val_results['lr_probabilities'],
                val_results['rf_probabilities'] * val_results['xgb_probabilities'],
                val_results['lr_probabilities'] * val_results['xgb_probabilities']
            ])
            
            # Get validation target
            val_y = val_data['lr_y']
            
            # Get predictions on validation data for evaluation
            val_y_pred = best_model.predict(val_meta_features)
            val_y_prob = best_model.predict_proba(val_meta_features)[:, 1]

            # Evaluate final model on validation data
            ensemble_metrics, ensemble_markdown = evaluate_model(
                val_y, val_y_pred, y_prob=val_y_prob, 
                model_name="Tuned Ensemble",
                label_encoder=base_models['lr']['encoder']
            )

            print("\nValidation Data Evaluation:")
            print(f"Accuracy: {ensemble_metrics['accuracy']:.4f}")
            print(f"F1 Score: {ensemble_metrics['f1']:.4f}")
            print(f"AUC-ROC: {ensemble_metrics['auc_roc']:.4f}")

            # Plot confusion matrix for the ensemble model
            plot_ensemble_confusion_matrix(val_y, val_y_pred)

            # Add ensemble results to ROC curve plot
            ensemble_results = {
                'ensemble_y_true': val_y,
                'ensemble_predictions': val_y_pred,
                'ensemble_probabilities': val_y_prob
            }

            # Plot ROC curves including the ensemble
            plot_roc_curves(base_models, val_results, val_data, ensemble_results)
        
        # Create ensemble directory and save model and related files
        os.makedirs('ensemble', exist_ok=True)

        # Save the final ensemble model
        joblib.dump(best_model, 'ensemble/ensemble_model.pkl')

        # Save base models and their preprocessors
        for model_name, model_dict in base_models.items():
            for component_name, component in model_dict.items():
                if component_name != 'process_func':  # Skip the function reference
                    joblib.dump(component, f'ensemble/{model_name}_{component_name}.pkl')

        # Save CV results as JSON
        with open('ensemble/cv_results.json', 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            serializable_results = {
                'params': cv_results['params'],
                'mean_accuracy': [float(x) for x in cv_results['mean_accuracy']],
                'mean_f1': [float(x) for x in cv_results['mean_f1']],
                'mean_auc': [float(x) for x in cv_results['mean_auc']],
                'std_accuracy': [float(x) for x in cv_results['std_accuracy']],
                'std_f1': [float(x) for x in cv_results['std_f1']],
                'std_auc': [float(x) for x in cv_results['std_auc']]
            }
            json.dump(serializable_results, f, indent=4)

        # Save best configuration
        with open('ensemble/best_config.json', 'w') as f:
            json.dump(best_params, f, indent=4)

        # Save ensemble evaluation report
        if ensemble_markdown:
            with open('ensemble/tuned_ensemble_evaluation.md', 'w') as f:
                f.write(ensemble_markdown)

        if verbose:
            print("\nEnsemble model and related files saved to 'ensemble' directory")
    else:
        if verbose:
            print("No valid model found. Try different parameters or data.")

    return best_model, cv_results, base_models, ensemble_metrics, ensemble_markdown


def evaluate_all_models(models, results, data):
    """
    Evaluate all models and generate markdown reports.

    Args:
        models: Dictionary containing all models (from load_base_models)
        results: Dictionary containing predictions and probabilities (from get_all_predictions)
        data: Dictionary containing processed data (from process_data)

    Returns:
        dict: Dictionary containing evaluation metrics for all models
        dict: Dictionary containing markdown reports for all models
    """
    metrics = {}
    markdown_reports = {}

    # Evaluate XGBoost
    if 'xgb_predictions' in results and len(results['xgb_predictions']) > 0:
        xgb_metrics, xgb_markdown = evaluate_model(
            data['xgb_y'],
            results['xgb_predictions'],
            y_prob=results['xgb_probabilities'],
            model_name="XGBoost",
            label_encoder=models['xgb']['encoder']
        )
        metrics['xgb'] = xgb_metrics
        markdown_reports['xgb'] = xgb_markdown

    # Evaluate Random Forest
    if 'rf_predictions' in results and len(results['rf_predictions']) > 0:
        rf_metrics, rf_markdown = evaluate_model(
            data['rf_y'],
            results['rf_predictions'],
            y_prob=results['rf_probabilities'],
            model_name="Random Forest",
            label_encoder=models['rf']['encoder']
        )
        metrics['rf'] = rf_metrics
        markdown_reports['rf'] = rf_markdown

    # Evaluate Logistic Regression
    if 'lr_predictions' in results and len(results['lr_predictions']) > 0:
        lr_metrics, lr_markdown = evaluate_model(
            data['lr_y'],
            results['lr_predictions'],
            y_prob=results['lr_probabilities'],
            model_name="Logistic Regression",
            label_encoder=models['lr']['encoder']
        )
        metrics['lr'] = lr_metrics
        markdown_reports['lr'] = lr_markdown

    return metrics, markdown_reports


def display_model_evaluations(markdown_reports, metrics, ensemble_markdown=None):
    """
    Display markdown reports for all models and save them to files.

    Args:
        markdown_reports: Dictionary containing markdown reports (from evaluate_all_models)
        metrics: Dictionary containing evaluation metrics (from evaluate_all_models)
        ensemble_markdown: Markdown report for the ensemble model (from tune_lr)
    """
    # Create ensemble directory if it doesn't exist
    os.makedirs('ensemble', exist_ok=True)

    # Print and save markdown reports
    for model_name, markdown_report in markdown_reports.items():
        # Print to console
        print(f"\n{'-'*80}\n{model_name.upper()} EVALUATION\n{'-'*80}")
        print(markdown_report)

        # Save to file
        with open(f'ensemble/{model_name}_evaluation.md', 'w') as f:
            f.write(markdown_report)

    # Only create comparison table if we have metrics for multiple models
    if len(metrics) > 1:
        # Create a summary comparison table
        comparison_markdown = "# Model Comparison\n\n"
        comparison_markdown += "| Model | Accuracy | F1 Score | Precision | Recall | ROC-AUC |\n"
        comparison_markdown += "|-------|----------|----------|-----------|--------|--------|\n"

        for model_name, model_metrics in metrics.items():
            model_display_name = {
                'xgb': 'XGBoost',
                'rf': 'Random Forest',
                'lr': 'Logistic Regression'
            }.get(model_name, model_name)

            accuracy = model_metrics.get('accuracy', 'N/A')
            f1 = model_metrics.get('f1', 'N/A')
            precision = model_metrics.get('precision', 'N/A')
            recall = model_metrics.get('recall', 'N/A')
            auc_roc = model_metrics.get('auc_roc', 'N/A')

            if isinstance(accuracy, float):
                comparison_markdown += f"| {model_display_name} | {accuracy:.4f} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {auc_roc:.4f} |\n"
            else:
                comparison_markdown += f"| {model_display_name} | {accuracy} | {f1} | {precision} | {recall} | {auc_roc} |\n"

        # Add ensemble model to comparison table if available
        if ensemble_markdown:
            ensemble_metrics = {}
            for line in ensemble_markdown.split('\n'):
                if line.startswith('|'):
                    parts = line.strip().split('|')
                    if len(parts) > 1:
                        metric_name = parts[1].strip()
                        metric_value = parts[2].strip()
                        ensemble_metrics[metric_name] = float(metric_value) if metric_value.replace('.', '', 1).isdigit() else metric_value

            model_display_name = "Tuned Ensemble"
            accuracy = ensemble_metrics.get('Accuracy', 'N/A')
            f1 = ensemble_metrics.get('F1 Score', 'N/A')
            precision = ensemble_metrics.get('Precision', 'N/A')
            recall = ensemble_metrics.get('Recall', 'N/A')
            auc_roc = ensemble_metrics.get('ROC-AUC', 'N/A')

            if isinstance(accuracy, float):
                comparison_markdown += f"| {model_display_name} | {accuracy:.4f} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {auc_roc:.4f} |\n"
            else:
                comparison_markdown += f"| {model_display_name} | {accuracy} | {f1} | {precision} | {recall} | {auc_roc} |\n"

        # Print comparison table
        print(f"\n{'-'*80}\nMODEL COMPARISON\n{'-'*80}")
        print(comparison_markdown)

        # Save comparison table
        with open('ensemble/model_comparison.md', 'w') as f:
            f.write(comparison_markdown)


def plot_confusion_matrices(models, results, data):
    """
    Plot confusion matrices for all models.

    Args:
        models: Dictionary containing all models (from load_base_models)
        results: Dictionary containing predictions and probabilities (from get_all_predictions)
        data: Dictionary containing processed data (from process_data)
    """

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot confusion matrix for XGBoost
    if 'xgb_predictions' in results and len(results['xgb_predictions']) > 0:
        cm = confusion_matrix(data['xgb_y'], results['xgb_predictions'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[0], cmap='Blues', values_format='d')
        axes[0].set_title('XGBoost Confusion Matrix')

    # Plot confusion matrix for Random Forest
    if 'rf_predictions' in results and len(results['rf_predictions']) > 0:
        cm = confusion_matrix(data['rf_y'], results['rf_predictions'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[1], cmap='Blues', values_format='d')
        axes[1].set_title('Random Forest Confusion Matrix')

    # Plot confusion matrix for Logistic Regression
    if 'lr_predictions' in results and len(results['lr_predictions']) > 0:
        cm = confusion_matrix(data['lr_y'], results['lr_predictions'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[2], cmap='Blues', values_format='d')
        axes[2].set_title('Logistic Regression Confusion Matrix')

    plt.tight_layout()

    # Save the figure
    os.makedirs('ensemble', exist_ok=True)
    plt.savefig('ensemble/confusion_matrices.png')
    plt.close()


def plot_roc_curves(models, results, data, ensemble_results=None):
    """
    Plot ROC curves for all models including the ensemble model if provided.

    Args:
        models: Dictionary containing all models (from load_base_models)
        results: Dictionary containing predictions and probabilities (from get_all_predictions)
        data: Dictionary containing processed data (from process_data)
        ensemble_results: Optional dictionary containing ensemble predictions and probabilities
    """
    plt.figure(figsize=(10, 8))

    # Plot ROC curve for XGBoost
    if 'xgb_probabilities' in results and len(results['xgb_probabilities']) > 0:
        fpr, tpr, _ = roc_curve(data['xgb_y'], results['xgb_probabilities'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.4f})')

    # Plot ROC curve for Random Forest
    if 'rf_probabilities' in results and len(results['rf_probabilities']) > 0:
        fpr, tpr, _ = roc_curve(data['rf_y'], results['rf_probabilities'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.4f})')

    # Plot ROC curve for Logistic Regression
    if 'lr_probabilities' in results and len(results['lr_probabilities']) > 0:
        fpr, tpr, _ = roc_curve(data['lr_y'], results['lr_probabilities'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.4f})')

    # Plot ROC curve for Ensemble if provided
    if ensemble_results and 'ensemble_y_true' in ensemble_results and 'ensemble_probabilities' in ensemble_results:
        fpr, tpr, _ = roc_curve(ensemble_results['ensemble_y_true'], ensemble_results['ensemble_probabilities'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 'k-', linewidth=2, label=f'Ensemble (AUC = {roc_auc:.4f})')

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc='lower right')

    # Save the figure
    os.makedirs('ensemble', exist_ok=True)
    plt.savefig('ensemble/roc_curves.png')
    plt.close()


def plot_ensemble_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix for the ensemble model.

    Args:
        y_true: True labels
        y_pred: Predicted labels
    """   
    # Create figure
    plt.figure(figsize=(8, 6))

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Ensemble Model Confusion Matrix')

    # Save the figure
    os.makedirs('ensemble', exist_ok=True)
    plt.savefig('ensemble/ensemble_confusion_matrix.png')
    plt.close()


def main(tune_model=True, num_rows=None):
    """
    Main function to train and evaluate ensemble models.

    Args:
        tune_model (bool): Whether to tune the ensemble model
        num_rows (int, optional): Number of rows to randomly sample from the dataset
    """
    # Load data
    df = load_data('../data/train.csv', num_rows=num_rows)
    
    # Split data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['disposition'])
    print(f"Training data: {train_df.shape[0]} samples")
    print(f"Test data: {test_df.shape[0]} samples")
    
    # Load base models
    base_models = load_base_models()
    
    # Process training data
    train_data = process_data(train_df, base_models)
    
    # Get predictions from base models on training data
    train_results = get_all_predictions(base_models, train_data)
    
    # Evaluate all models on training data
    train_metrics, train_markdown_reports = evaluate_all_models(base_models, train_results, train_data)
    
    # Display evaluation reports for training data
    print("\n=== TRAINING DATA EVALUATION ===")
    display_model_evaluations(train_markdown_reports, train_metrics)
    
    # Plot confusion matrices and ROC curves for training data
    plot_confusion_matrices(base_models, train_results, train_data)
    plot_roc_curves(base_models, train_results, train_data)
    
    # Train ensemble
    meta_learner, base_models = train_ensemble(train_df)
    
    # Process test data
    test_data = process_data(test_df, base_models)
    
    # Get predictions from base models on test data
    test_results = get_all_predictions(base_models, test_data)
    
    # Evaluate all models on test data
    test_metrics, test_markdown_reports = evaluate_all_models(base_models, test_results, test_data)
    
    # Get ensemble predictions on test data
    meta_features = np.column_stack([
        test_results['rf_probabilities'],
        test_results['lr_probabilities'],
        test_results['xgb_probabilities'],
        # Add squared terms to allow for non-linear relationships
        test_results['rf_probabilities']**2,
        test_results['lr_probabilities']**2,
        test_results['xgb_probabilities']**2,
        # Add interaction terms
        test_results['rf_probabilities'] * test_results['lr_probabilities'],
        test_results['rf_probabilities'] * test_results['xgb_probabilities'],
        test_results['lr_probabilities'] * test_results['xgb_probabilities']
    ])
    
    ensemble_preds = meta_learner.predict(meta_features)
    ensemble_probs = meta_learner.predict_proba(meta_features)[:, 1]
    
    # Evaluate ensemble on test data
    ensemble_metrics, ensemble_markdown = evaluate_model(
        test_data['lr_y'],
        ensemble_preds,
        y_prob=ensemble_probs,
        model_name="Ensemble Model"
    )
    
    # Display evaluation reports for test data
    print("\n=== TEST DATA EVALUATION ===")
    display_model_evaluations(test_markdown_reports, test_metrics, ensemble_markdown)
    
    # Create ensemble results dictionary for ROC curve plotting
    ensemble_results = {
        'ensemble_y_true': test_data['lr_y'],
        'ensemble_predictions': ensemble_preds,
        'ensemble_probabilities': ensemble_probs
    }
    
    # Plot ROC curves for test data including ensemble
    plot_roc_curves(base_models, test_results, test_data, ensemble_results)
    
    # Save ensemble model
    os.makedirs('models', exist_ok=True)
    joblib.dump(meta_learner, 'models/ensemble_model.pkl')
    
    print("\nEnsemble model has been trained and saved to 'models/ensemble_model.pkl'")
    print("The ensemble combines predictions from Random Forest, Logistic Regression, and XGBoost models")
    
    # Tune logistic regression ensemble model if requested
    if tune_model:
        print("\n=== TUNING ENSEMBLE MODEL ===")
        best_model, cv_results, base_models, ensemble_metrics, ensemble_markdown = tune_lr(train_df)
        
        # Evaluate tuned model on test data
        tuned_preds = best_model.predict(meta_features)
        tuned_probs = best_model.predict_proba(meta_features)[:, 1]
        
        tuned_metrics, tuned_markdown = evaluate_model(
            test_data['lr_y'],
            tuned_preds,
            y_prob=tuned_probs,
            model_name="Tuned Ensemble"
        )
        
        print("\n=== TUNED MODEL TEST EVALUATION ===")
        display_model_evaluations({}, {}, tuned_markdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate ensemble models')
    parser.add_argument('-num_rows', type=int, help='Number of rows to randomly sample from the dataset')
    parser.add_argument('-no_tune', action='store_true', help='Skip model tuning')
    args = parser.parse_args()
    
    main(tune_model=not args.no_tune, num_rows=args.num_rows)