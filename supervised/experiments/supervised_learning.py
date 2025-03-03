#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hospital Admission Prediction - Ensemble Model Evaluation

This script demonstrates the performance of our ensemble model and its component models
on a validation dataset. The ensemble combines the strengths of three different 
machine learning approaches:

1. XGBoost - Utilizing embeddings from the chief complaint text
2. Random Forest - Using complexity scores and structured features
3. Logistic Regression - With carefully selected features and parameter tuning

The ensemble model leverages the predictions from these three models to make more
robust and accurate predictions about hospital admissions.

Related Scripts:
- Logistic Regression Experiments (logreg_experiments.py)
- Failure Analysis (failure_analysis.py)
- PCA Experiments (pca.py)

Project Overview:
This project aims to predict whether a patient visiting the emergency department
will be admitted to the hospital or sent home, based on their initial presentation data.
Accurate prediction can help with resource allocation and patient flow management.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, classification_report,
    roc_auc_score, precision_score, recall_score, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from utils.data_utils import load_data, process_data_for_rf, process_data_for_lr, process_data_for_xgboost
from utils.embeddings import EmbeddingExtractor
from utils.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve

# Set seaborn style for better visualizations
sns.set_theme(style="whitegrid")
sns.set_palette("Blues")
plt.rcParams['figure.figsize'] = [10, 6]

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)


def load_validation_data():
    """Load and display validation data"""
    print("Loading validation data...")
    validation_df = pd.read_csv('validation.csv')
    print(f"Loaded validation dataset with {validation_df.shape[0]} rows and {validation_df.shape[1]} columns")
    
    # Display a sample of the data
    print("\nSample of validation data:")
    print(validation_df.head())
    
    return validation_df


def load_base_models():
    """Load all base models and their associated components"""
    print("Loading base models...")
    
    models = {
        'xgb': {
            'model': joblib.load("models/xgboost_model.pkl"),
            'scaler': joblib.load('models/xgb_scaler.pkl'),
            'encoder': joblib.load('models/xgb_label_encoder.pkl'),
            'pca': joblib.load('models/xgb_pca.pkl')
        },
        'rf': {
            'model': joblib.load("models/random_forest_model.pkl"),
            'scaler': joblib.load('models/rf_scaler.pkl'),
            'encoder': joblib.load('models/rf_label_encoder.pkl')
        },
        'lr': {
            'model': joblib.load("models/logistic_regression_model.pkl"),
            'preprocessor': joblib.load('models/lr_preprocessor.pkl'),
            'scaler': joblib.load('models/lr_scaler.pkl'),
            'encoder': joblib.load('models/lr_encoder.pkl')
        }
    }
    
    # Add process functions
    models['xgb']['process_func'] = process_data_for_xgboost
    models['rf']['process_func'] = process_data_for_rf
    models['lr']['process_func'] = process_data_for_lr
    
    return models


def load_ensemble_model():
    """Load the tuned ensemble model"""
    print("Loading ensemble model...")
    
    ensemble_model = joblib.load("ensemble/ensemble_model.pkl")
    
    return ensemble_model


def process_data(df, models):
    """Process data for all models"""
    print("Processing data for all models...")

    df_subset = df[['disposition', 'chiefcomplaint']]

    # Process data for XGBoost
    embedder = EmbeddingExtractor()
    embeddings = embedder.get_or_create_embeddings(df_subset, column="chiefcomplaint")
    pca_obj = models['xgb']['pca']
    xgb_data = process_data_for_xgboost(embeddings, df['disposition'], pca_obj=pca_obj, training=False)

    # Process data for Random Forest
    rf_data = process_data_for_rf(df_subset)

    # Process data for Logistic Regression
    preprocessor = models['lr']['preprocessor']
    lr_data = process_data_for_lr(df, preprocessor=preprocessor, mode='inference')

    # Organize processed data
    data = {
        'xgb_X': xgb_data['X'],
        'xgb_y': xgb_data['y'],
        'rf_X': rf_data['X'],
        'rf_y': rf_data['y'],
        'lr_X': lr_data['X'],
        'lr_y': lr_data['y']
    }

    return data


def get_all_predictions(models, data):
    """Get predictions from all models"""
    print("Getting predictions from all models...")

    results = {}

    # XGBoost predictions
    results['xgb_predictions'] = models['xgb']['model'].predict(data['xgb_X'])
    results['xgb_probabilities'] = models['xgb']['model'].predict_proba(data['xgb_X'])[:, 1]
    
    # Random Forest predictions
    results['rf_predictions'] = models['rf']['model'].predict(data['rf_X'])
    results['rf_probabilities'] = models['rf']['model'].predict_proba(data['rf_X'])[:, 1]
    
    # Logistic Regression predictions
    results['lr_predictions'] = models['lr']['model'].predict(data['lr_X'])
    results['lr_probabilities'] = models['lr']['model'].predict_proba(data['lr_X'])[:, 1]
    
    return results


def evaluate_all_models(base_models, predictions, data, ensemble_preds, ensemble_probs):
    """Evaluate all models and create visualizations"""
    print("Evaluating all models...")
    
    metrics = {}
    markdown_reports = {}
    
    # Evaluate XGBoost
    xgb_metrics, xgb_markdown = evaluate_model(
        data['xgb_y'], 
        predictions['xgb_predictions'],
        predictions['xgb_probabilities'],
        model_name="XGBoost"
    )
    metrics['xgb'] = xgb_metrics
    markdown_reports['xgb'] = xgb_markdown
    
    # Evaluate Random Forest
    rf_metrics, rf_markdown = evaluate_model(
        data['rf_y'], 
        predictions['rf_predictions'],
        predictions['rf_probabilities'],
        model_name="Random Forest"
    )
    metrics['rf'] = rf_metrics
    markdown_reports['rf'] = rf_markdown
    
    # Evaluate Logistic Regression
    lr_metrics, lr_markdown = evaluate_model(
        data['lr_y'], 
        predictions['lr_predictions'],
        predictions['lr_probabilities'],
        model_name="Logistic Regression"
    )
    metrics['lr'] = lr_metrics
    markdown_reports['lr'] = lr_markdown
    
    # Evaluate Ensemble model
    ensemble_metrics, ensemble_markdown = evaluate_model(
        data['lr_y'],  # Using LR's y as the ground truth (same as in validation.py)
        ensemble_preds,
        ensemble_probs,
        model_name="Ensemble Model"
    )
    metrics['ensemble'] = ensemble_metrics
    markdown_reports['ensemble'] = ensemble_markdown
    
    return metrics, markdown_reports


def plot_confusion_matrices(base_models, predictions, data, ensemble_preds):
    """Plot confusion matrices for all models"""
    print("Plotting confusion matrices...")
    
    # Ensure ensemble directory exists
    os.makedirs('ensemble', exist_ok=True)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot confusion matrix for XGBoost
    cm_xgb = confusion_matrix(data['xgb_y'], predictions['xgb_predictions'])
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('XGBoost Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    # Plot confusion matrix for Random Forest
    cm_rf = confusion_matrix(data['rf_y'], predictions['rf_predictions'])
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('Random Forest Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    # Plot confusion matrix for Logistic Regression
    cm_lr = confusion_matrix(data['lr_y'], predictions['lr_predictions'])
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[2])
    axes[2].set_title('Logistic Regression Confusion Matrix')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')
    
    # Plot confusion matrix for Ensemble
    cm_ensemble = confusion_matrix(data['xgb_y'], ensemble_preds)
    sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Blues', ax=axes[3])
    axes[3].set_title('Ensemble Model Confusion Matrix')
    axes[3].set_xlabel('Predicted')
    axes[3].set_ylabel('True')
    
    plt.tight_layout()
    # Save to ensemble directory instead of showing
    plt.savefig('ensemble/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("Confusion matrices saved to 'ensemble/confusion_matrices.png'")
    
    return fig


def plot_roc_curves(base_models, predictions, data, ensemble_probs):
    """Plot ROC curves for all models"""
    print("Plotting ROC curves...")
    
    # Ensure ensemble directory exists
    os.makedirs('ensemble', exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for XGBoost
    fpr_xgb, tpr_xgb, _ = roc_curve(data['xgb_y'], predictions['xgb_probabilities'])
    roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.3f})')
    
    # Plot ROC curve for Random Forest
    fpr_rf, tpr_rf, _ = roc_curve(data['rf_y'], predictions['rf_probabilities'])
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
    
    # Plot ROC curve for Logistic Regression
    fpr_lr, tpr_lr, _ = roc_curve(data['lr_y'], predictions['lr_probabilities'])
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.3f})')
    
    # Plot ROC curve for Ensemble
    fpr_ensemble, tpr_ensemble, _ = roc_curve(data['xgb_y'], ensemble_probs)
    roc_auc_ensemble = auc(fpr_ensemble, tpr_ensemble)
    plt.plot(fpr_ensemble, tpr_ensemble, label=f'Ensemble Model (AUC = {roc_auc_ensemble:.3f})')
    
    # Plot the diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Save to ensemble directory instead of showing
    plt.savefig('ensemble/roc_curves.png', dpi=300, bbox_inches='tight')
    print("ROC curves saved to 'ensemble/roc_curves.png'")
    
    return plt.gcf()


def analyze_predictions(base_models, predictions, data, ensemble_preds):
    """Analyze predictions from all models"""
    print("Analyzing predictions...")
    
    # Create a DataFrame with all predictions
    results_df = pd.DataFrame({
        'True': data['xgb_y'],
        'XGBoost': predictions['xgb_predictions'],
        'RandomForest': predictions['rf_predictions'],
        'LogisticRegression': predictions['lr_predictions'],
        'Ensemble': ensemble_preds
    })
    
    # Add probability columns
    results_df['XGBoost_Prob'] = predictions['xgb_probabilities']
    results_df['RandomForest_Prob'] = predictions['rf_probabilities']
    results_df['LogisticRegression_Prob'] = predictions['lr_probabilities']
    
    # Find cases where ensemble is correct but individual models are wrong
    ensemble_correct = results_df[results_df['True'] == results_df['Ensemble']]
    ensemble_advantage = ensemble_correct[
        (ensemble_correct['XGBoost'] != ensemble_correct['True']) |
        (ensemble_correct['RandomForest'] != ensemble_correct['True']) |
        (ensemble_correct['LogisticRegression'] != ensemble_correct['True'])
    ]
    
    print(f"\nEnsemble Advantage: {len(ensemble_advantage)} cases where ensemble is correct but at least one individual model is wrong")
    
    # Sample of such cases
    if len(ensemble_advantage) > 0:
        print("\nSample cases where ensemble outperforms individual models:")
        print(ensemble_advantage.head(5))
    
    return results_df


def main():
    """Main function to run the evaluation pipeline"""
    # Load validation data
    validation_df = load_validation_data()
    
    # Load models
    base_models = load_base_models()
    ensemble_model = load_ensemble_model()
    print("All models loaded successfully!")
    
    # Process validation data
    processed_data = process_data(validation_df, base_models)
    print("Data processing complete!")
    
    # Get predictions from base models
    predictions = get_all_predictions(base_models, processed_data)
    
    # Create meta-features for ensemble model
    meta_features = np.column_stack([
        predictions['rf_probabilities'],
        predictions['lr_probabilities'],
        predictions['xgb_probabilities'],
        predictions['rf_predictions'],
        predictions['lr_predictions'],
        predictions['xgb_predictions'],
        predictions['rf_probabilities'] * predictions['lr_probabilities'],
        predictions['rf_probabilities'] * predictions['xgb_probabilities'],
        predictions['lr_probabilities'] * predictions['xgb_probabilities']
    ])
    
    # Get ensemble predictions
    ensemble_predictions = ensemble_model.predict(meta_features)
    ensemble_probabilities = ensemble_model.predict_proba(meta_features)[:, 1]
    print("All predictions complete!")
    
    # Evaluate all models
    metrics, markdown_reports = evaluate_all_models(
        base_models, 
        predictions, 
        processed_data, 
        ensemble_predictions, 
        ensemble_probabilities
    )
    
    # Display evaluation reports
    for model_name, report in markdown_reports.items():
        print(f"\n{report}")
    
    # Plot confusion matrices
    plot_confusion_matrices(base_models, predictions, processed_data, ensemble_predictions)
    
    # Plot ROC curves
    plot_roc_curves(base_models, predictions, processed_data, ensemble_probabilities)
    
    # Analyze predictions
    results_df = analyze_predictions(base_models, predictions, processed_data, ensemble_predictions)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
