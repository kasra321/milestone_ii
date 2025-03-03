#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Validation Script

This script loads and evaluates machine learning models for prediction tasks.
It was converted from the validation.ipynb notebook.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, roc_auc_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from sklearn.metrics import ConfusionMatrixDisplay

from transformers import AutoTokenizer, AutoModel
import torch

from utils.data_utils import load_data, process_data_for_rf, process_data_for_lr, process_data_for_xgboost
from utils.embeddings import EmbeddingExtractor
from utils.evaluation import evaluate_model

def main():
    """Main function to run the validation script"""
    
    # Create validation directory if it doesn't exist
    validation_dir = "validation"
    os.makedirs(validation_dir, exist_ok=True)
    
    print("Loading models...")
    # Load the saved model and make predictions
    xgb_model = joblib.load("models/xgboost_model.pkl")
    xgb_scaler = joblib.load('models/xgb_scaler.pkl')
    xgb_label_encoder = joblib.load('models/xgb_label_encoder.pkl')
    xgb_pca = joblib.load('models/xgb_pca.pkl')

    # Load random forest
    rf_model = joblib.load("models/random_forest_model.pkl")
    rf_scaler = joblib.load('models/rf_scaler.pkl')
    rf_label_encoder = joblib.load('models/rf_label_encoder.pkl')

    # Load logistic regression
    lr_model = joblib.load("models/logistic_regression_model.pkl")
    lr_encoder = joblib.load("models/lr_encoder.pkl")

    print("Loading and processing data...")
    # Load validation data
    df = pd.read_csv('validation.csv')

    # Process data for XGBoost
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        embedder = EmbeddingExtractor()
        embeddings = embedder.get_or_create_embeddings(df)
        xgb_obj = process_data_for_xgboost(embeddings, df['disposition'])
        xgb_X = xgb_obj['X']
        xgb_y = xgb_obj['y']

    # Process data for Random Forest
    rf_obj = process_data_for_rf(df)
    rf_X = rf_obj['X']
    rf_y = rf_obj['y'] 

    # Process data for LR
    lr_obj = process_data_for_lr(df)
    lr_X = lr_obj['X']
    lr_y = lr_obj['y']

    print("Making predictions...")
    # Make predictions
    lr_y_predictions = lr_model.predict(lr_X)
    lr_y_probabilities = lr_model.predict_proba(lr_X)[:, 1]

    xgb_y_predictions = xgb_model.predict(xgb_X)
    xgb_y_probabilities = xgb_model.predict_proba(xgb_X)[:,1]

    rf_y_predictions = rf_model.predict(rf_X)
    rf_y_probabilities = rf_model.predict_proba(rf_X)[:, 1]

    print("Evaluating models...")
    # For XGBoost
    xgb_metrics, xgb_markdown = evaluate_model(
        xgb_y, 
        xgb_y_predictions,
        xgb_y_probabilities,
        model_name="XGBoost",
        label_encoder=xgb_label_encoder
    )

    # For Random Forest
    rf_metrics, rf_markdown = evaluate_model(
        rf_y,
        rf_y_predictions,
        y_prob=rf_y_probabilities,
        model_name="Random Forest"
    )

    # For Logistic Regression
    lr_metrics, lr_markdown = evaluate_model(
        lr_y,
        lr_y_predictions,
        y_prob=lr_y_probabilities,
        model_name="Logistic Regression"
    )

    # Print evaluation results
    print("\n" + xgb_markdown)
    print("\n" + rf_markdown)
    print("\n" + lr_markdown)

    # Create a comparison table
    markdown_text = f"""
# Model Comparison Summary

| Model | AUC-ROC | Accuracy |
|-------|---------|----------|
| Logistic Regression | {lr_metrics['auc_roc']:.4f} | {lr_metrics['accuracy']:.4f} |
| XGBoost | {xgb_metrics['auc_roc']:.4f} | {xgb_metrics['accuracy']:.4f} |
| Random Forest | {rf_metrics['auc_roc']:.4f} | {rf_metrics['accuracy']:.4f} |
"""
    print("\n" + markdown_text)

    # Save markdown reports to validation directory
    with open(os.path.join(validation_dir, "xgb_evaluation.md"), "w") as f:
        f.write(xgb_markdown)
    
    with open(os.path.join(validation_dir, "rf_evaluation.md"), "w") as f:
        f.write(rf_markdown)
    
    with open(os.path.join(validation_dir, "lr_evaluation.md"), "w") as f:
        f.write(lr_markdown)
    
    with open(os.path.join(validation_dir, "model_comparison.md"), "w") as f:
        f.write(markdown_text)

    # Load and evaluate the ensemble model
    print("Loading and evaluating ensemble model...")
    ensemble_model = joblib.load("ensemble/ensemble_model.pkl")
    print("Ensemble model loaded successfully")

    # Create meta-features from base model predictions
    meta_features = np.column_stack([
        rf_y_probabilities,
        lr_y_probabilities,
        xgb_y_probabilities,
        # Add squared terms to allow for non-linear relationships
        rf_y_probabilities**2,
        lr_y_probabilities**2,
        xgb_y_probabilities**2,
        # Add interaction terms
        rf_y_probabilities * lr_y_probabilities,
        rf_y_probabilities * xgb_y_probabilities,
        lr_y_probabilities * xgb_y_probabilities
    ])

    print(f"Meta-features shape: {meta_features.shape}")

    # Get predictions from the ensemble model
    ensemble_predictions = ensemble_model.predict(meta_features)
    ensemble_probabilities = ensemble_model.predict_proba(meta_features)[:, 1]

    # Use any of the label encoders (they should all encode the target the same way)
    # We'll use the lr_y as the true values
    ensemble_y_true = lr_y

    # Evaluate the ensemble model
    ensemble_metrics, ensemble_markdown = evaluate_model(
        ensemble_y_true,
        ensemble_predictions,
        y_prob=ensemble_probabilities,
        model_name="Tuned Ensemble"
    )

    # Display the evaluation results
    print("\n" + ensemble_markdown)

    # Save ensemble evaluation to validation directory
    with open(os.path.join(validation_dir, "ensemble_evaluation.md"), "w") as f:
        f.write(ensemble_markdown)

    # Create an updated comparison including the ensemble model
    updated_comparison = f"""
# Updated Model Comparison Summary

| Model | AUC-ROC | Accuracy | F1 Score | Precision | Recall |
|-------|---------|----------|----------|-----------|--------|
| Logistic Regression | {lr_metrics['auc_roc']:.4f} | {lr_metrics['accuracy']:.4f} | {lr_metrics['f1']:.4f} | {lr_metrics['precision']:.4f} | {lr_metrics['recall']:.4f} |
| XGBoost | {xgb_metrics['auc_roc']:.4f} | {xgb_metrics['accuracy']:.4f} | {xgb_metrics['f1']:.4f} | {xgb_metrics['precision']:.4f} | {xgb_metrics['recall']:.4f} |
| Random Forest | {rf_metrics['auc_roc']:.4f} | {rf_metrics['accuracy']:.4f} | {rf_metrics['f1']:.4f} | {rf_metrics['precision']:.4f} | {rf_metrics['recall']:.4f} |
| **Tuned Ensemble** | {ensemble_metrics['auc_roc']:.4f} | {ensemble_metrics['accuracy']:.4f} | {ensemble_metrics['f1']:.4f} | {ensemble_metrics['precision']:.4f} | {ensemble_metrics['recall']:.4f} |
"""
    print("\n" + updated_comparison)

    # Save updated comparison to validation directory
    with open(os.path.join(validation_dir, "updated_model_comparison.md"), "w") as f:
        f.write(updated_comparison)

    # Plot ROC curves for all models
    plt.figure(figsize=(10, 8))

    # Plot ROC curve for XGBoost
    fpr_xgb, tpr_xgb, _ = roc_curve(xgb_y, xgb_y_probabilities)
    roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.4f})')

    # Plot ROC curve for Random Forest
    fpr_rf, tpr_rf, _ = roc_curve(rf_y, rf_y_probabilities)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.4f})')

    # Plot ROC curve for Logistic Regression
    fpr_lr, tpr_lr, _ = roc_curve(lr_y, lr_y_probabilities)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.4f})')

    # Plot ROC curve for Ensemble
    fpr_ensemble, tpr_ensemble, _ = roc_curve(ensemble_y_true, ensemble_probabilities)
    roc_auc_ensemble = auc(fpr_ensemble, tpr_ensemble)
    plt.plot(fpr_ensemble, tpr_ensemble, 'k-', linewidth=2, label=f'Ensemble (AUC = {roc_auc_ensemble:.4f})')

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Save the plot to validation directory instead of showing it
    roc_plot_path = os.path.join(validation_dir, 'roc_curves_comparison.png')
    plt.savefig(roc_plot_path)
    print(f"ROC curves saved to '{roc_plot_path}'")
    
    # Create confusion matrices for each model
    for model_name, y_true, y_pred in [
        ('xgboost', xgb_y, xgb_y_predictions),
        ('random_forest', rf_y, rf_y_predictions),
        ('logistic_regression', lr_y, lr_y_predictions),
        ('ensemble', ensemble_y_true, ensemble_predictions)
    ]:
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}')
        plt.savefig(os.path.join(validation_dir, f'{model_name}_confusion_matrix.png'))
        print(f"Confusion matrix for {model_name} saved to validation directory")

if __name__ == "__main__":
    main()
