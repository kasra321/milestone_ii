from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
)
import warnings
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate_model(y_true, y_pred, y_prob=None, model_name="Model", label_encoder=None, model=None, pos_label=1):
    """
    Calculate and print evaluation metrics for a classification model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities for ROC-AUC (optional)
        model_name: Name of the model for display
        label_encoder: Optional label encoder to convert predictions back to original labels
        model: Optional model object for device handling with XGBoost
        pos_label: The label of the positive class (default: 1)
        
    Returns:
        dict: Dictionary containing all computed metrics
        str: Markdown string containing evaluation results
    """
    # Handle XGBoost device warnings
    if isinstance(model, xgb.XGBClassifier):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            # If model is on GPU but no probabilities provided, get them
            if y_prob is None and hasattr(model, 'get_booster'):
                if model.get_booster().device == 'cuda:0':
                    # Move model to CPU for prediction
                    model.set_params(device='cpu')
                    y_prob = model.predict_proba(y_true)[:, 1]
    
    # Calculate binary metrics for the positive class
    binary_classification = len(np.unique(y_true)) == 2
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='weighted'),
    }
    
    # For binary classification, calculate recall specifically for the positive class
    if binary_classification:
        metrics['recall'] = recall_score(y_true, y_pred, pos_label=pos_label)
        # Also add binary precision and F1 for the positive class
        metrics['binary_precision'] = precision_score(y_true, y_pred, pos_label=pos_label)
        metrics['binary_f1'] = f1_score(y_true, y_pred, pos_label=pos_label)
    else:
        # For multi-class, use weighted average
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    
    # Add ROC-AUC if probabilities are provided
    if y_prob is not None:
        metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob))
    
    # Get classification report
    report = classification_report(y_true, y_pred)
    
    # Create markdown string with improved table formatting
    markdown_string = f"# {model_name} Evaluation Results\n\n"

    # Metrics Table
    markdown_string += "## Metrics\n\n"
    markdown_string += "| Metric                    | Score    |\n"
    markdown_string += "|---------------------------|----------|\n"
    markdown_string += f"| Validation Accuracy       | {metrics['accuracy']:.4f} |\n"
    
    if binary_classification:
        markdown_string += f"| Validation F1 Score (weighted) | {metrics['f1']:.4f} |\n"
        markdown_string += f"| Validation Precision (weighted) | {metrics['precision']:.4f} |\n"
        markdown_string += f"| Validation Recall (class {pos_label}) | {metrics['recall']:.4f} |\n"
        markdown_string += f"| Binary Precision (class {pos_label}) | {metrics['binary_precision']:.4f} |\n"
        markdown_string += f"| Binary F1 Score (class {pos_label}) | {metrics['binary_f1']:.4f} |\n"
    else:
        markdown_string += f"| Validation F1 Score       | {metrics['f1']:.4f} |\n"
        markdown_string += f"| Validation Precision      | {metrics['precision']:.4f} |\n"
        markdown_string += f"| Validation Recall         | {metrics['recall']:.4f} |\n"
        
    if 'auc_roc' in metrics:
        markdown_string += f"| ROC-AUC Score             | {metrics['auc_roc']:.4f} |\n"

    # Detailed Classification Report in a code block
    markdown_string += "\n## Detailed Classification Report\n\n"
    markdown_string += "```\n"
    markdown_string += report
    markdown_string += "\n```\n"

    # Sample predictions with original labels
    if label_encoder is not None:
        predictions_labels = label_encoder.inverse_transform(y_pred)
        markdown_string += "\n## Sample of Predictions in Original Labels\n\n"
        markdown_string += f"`{predictions_labels[:5]} ...`\n"

    
    return metrics, markdown_string

def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot and optionally save a confusion matrix for a classification model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for display
        save_path: Path to save the plot (optional)
        
    Returns:
        matplotlib.figure.Figure: The figure containing the confusion matrix
    """
    from sklearn.metrics import ConfusionMatrixDisplay
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'{model_name} Confusion Matrix')
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    return fig

def plot_roc_curve(y_true, y_prob, model_name="Model", save_path=None):
    """
    Plot and optionally save a ROC curve for a classification model.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        model_name: Name of the model for display
        save_path: Path to save the plot (optional)
        
    Returns:
        matplotlib.figure.Figure: The figure containing the ROC curve
    """
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{model_name} ROC Curve')
    ax.legend(loc='lower right')
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    return fig

def plot_feature_importance(model, feature_names, model_name="Model", top_n=20, save_path=None):
    """
    Plot and optionally save feature importance for a model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: Names of the features
        model_name: Name of the model for display
        top_n: Number of top features to display
        save_path: Path to save the plot (optional)
        
    Returns:
        matplotlib.figure.Figure: The figure containing the feature importance plot
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {model_name} does not have feature_importances_ attribute")
        return None
    
    # Get feature importances and sort them
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Feature Importance for {model_name}')
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    return fig
