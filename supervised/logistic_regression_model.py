import os
import json
import logging
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from IPython.display import display, Markdown

from utils.data_utils import load_data
from utils.evaluation import evaluate_model

# Set up logger
logger = logging.getLogger(__name__)

class LogisticRegressionModel:
    """
    A class that encapsulates functionality for training, evaluating, and using 
    logistic regression models with preprocessing capabilities.
    """

    def __init__(self, 
                 numeric_features=None, 
                 categorical_features=None,
                 target_column='disposition',
                 random_state=42):
        """
        Initialize the LogisticRegressionModel.
        
        Args:
            numeric_features: List of numeric feature column names
            categorical_features: List of categorical feature column names
            target_column: Name of the target column
            random_state: Random state for reproducibility
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_column = target_column
        self.random_state = random_state

        # Model and preprocessing objects (to be set during training)
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.best_params = None
        self.metrics = None

        # Training data (optional to keep)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
  
    def _resample(self, X, y):
        """
        Undersample to match classes 1:1
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            X_resampled, y_resampled: Resampled data
        """
        sampler = RandomUnderSampler(random_state=self.random_state)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        logger.info('Class distribution before sampling: %s', np.bincount(y))
        logger.info('Class distribution after sampling: %s', np.bincount(y_resampled))
        return X_resampled, y_resampled

    def _tune_lr(self, X_train, y_train, cv_folds=5, param_grid=None):
        """
        Train a Logistic Regression model using k-fold cross-validation with hyperparameter tuning.

        Parameters:
        - X_train: Training features (DataFrame or array)
        - y_train: Training labels (Series or array)
        - cv_folds: Number of folds for cross-validation
        - param_grid: Dictionary of hyperparameters to search over

        Returns:
        - best_model: The best Logistic Regression model found via GridSearchCV
        - grid_search: The fitted GridSearchCV object (for further inspection if needed)
        - best_params: Dictionary of best parameters found
        """
        # Set default hyperparameter grid if not provided
        if param_grid is None:
            # Define separate parameter grids for different solvers to ensure compatibility
            param_grid = [
                # liblinear solver - works with l1 and l2 penalties
                {
                    'solver': ['liblinear'],
                    'penalty': ['l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10],
                    'max_iter': [100, 500, 1000]
                },
                # lbfgs solver - only works with l2 penalty
                {
                    'solver': ['lbfgs'],
                    'penalty': ['l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10],
                    'max_iter': [100, 500]
                }
            ]

        # Initialize the logistic regression classifier
        lr = LogisticRegression(random_state=self.random_state)

        # Setup k-fold cross-validation
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        # Initialize GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=lr,
            param_grid=param_grid,
            cv=kfold,
            scoring='accuracy',
            n_jobs=-1,
            refit=True,  # refit on the entire training set with the best params
            verbose=2
        )

        # Fit grid search
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print("Best parameters found:", best_params)
        print("Best cross-validation accuracy: {:.4f}".format(grid_search.best_score_))

        best_model = grid_search.best_estimator_
        return best_model, grid_search, best_params

    def _process_features(self, df, preprocessor=None):
        """
        Process features for logistic regression model.
        """
        numeric_features = self.numeric_features
        categorical_features = self.categorical_features
        result = {}

        # Get expected features and their order from model
        if hasattr(self.model, 'feature_names_in_'):
            expected_features = list(self.model.feature_names_in_)
        else:
            expected_features = self.feature_names or []
      
        # Initialize feature DataFrames
        numeric_df = pd.DataFrame(index=df.index)
        categorical_df = pd.DataFrame(index=df.index)

        # Process numeric features
        if numeric_features:
            available_numeric = [f for f in numeric_features if f in df.columns]
            if expected_features:
                available_numeric = [f for f in available_numeric if f in expected_features]
  
            if available_numeric:
                numeric_data = df[available_numeric].copy()
                for col in numeric_data.columns:
                    numeric_data[col] = numeric_data[col].fillna(numeric_data[col].median())

                if preprocessor and 'scaler' in preprocessor:
                    scaler = preprocessor['scaler']
                    numeric_scaled = scaler.transform(numeric_data)
                else:
                    scaler = StandardScaler()
                    numeric_scaled = scaler.fit_transform(numeric_data)
                    result['scaler'] = scaler

                numeric_df = pd.DataFrame(numeric_scaled, columns=available_numeric, index=df.index)

        # Process categorical features
        if categorical_features:
            available_categorical = [f for f in categorical_features if f in df.columns]

            if available_categorical:
                categorical_data = df[available_categorical].copy()
                for col in categorical_data.columns:
                    categorical_data[col] = categorical_data[col].fillna(categorical_data[col].mode()[0])

                if preprocessor and 'encoder' in preprocessor:
                    encoder = preprocessor['encoder']
                    try:
                        categorical_encoded = encoder.transform(categorical_data)
                        cat_feature_names = encoder.get_feature_names_out(available_categorical)
                    except (ValueError, KeyError, AttributeError) as e:
                        print(f"Warning: Using new encoder due to: {e}")
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        categorical_encoded = encoder.fit_transform(categorical_data)
                        result['encoder'] = encoder
                        cat_feature_names = encoder.get_feature_names_out(available_categorical)
                else:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    categorical_encoded = encoder.fit_transform(categorical_data)
                    result['encoder'] = encoder
                    cat_feature_names = encoder.get_feature_names_out(available_categorical)

                # Filter to expected features if needed
                if expected_features:
                    valid_indices = [i for i, name in enumerate(cat_feature_names) 
                                   if name in expected_features]
                    if valid_indices:
                        categorical_encoded = categorical_encoded[:, valid_indices]
                        cat_feature_names = [cat_feature_names[i] for i in valid_indices]

                categorical_df = pd.DataFrame(categorical_encoded, 
                                           columns=cat_feature_names, 
                                           index=df.index)

        # Combine features
        X = pd.concat([numeric_df, categorical_df], axis=1)

        # Handle expected features
        if expected_features:
            # Add missing features as zeros
            missing_features = set(expected_features) - set(X.columns)
            for feature in missing_features:
                X[feature] = 0
                
            # Ensure exact column order from training
            X = X[expected_features]
            
            print(f"Final feature count: {len(X.columns)} (matching training features)")
        
        result.update({
            'X': X,
            'feature_names': list(X.columns),
            'all_feature_names': expected_features if expected_features else list(X.columns)
        })
        
        return result
    
    def train(self, 
             data_file=None,
             df=None,
             num_rows=None,
             resample_flag=False, 
             tune_model=True, 
             default_model_params=None,
             test_size=0.1):
        """
        Train a logistic regression model with specified features
        
        Args:
            data_file: Path to the data file (optional if df is provided)
            df: DataFrame to train on (optional if data_file is provided)
            num_rows: Number of rows to load from data_file (None for all rows)
            resample_flag: Whether to resample the data to handle class imbalance
            tune_model: Whether to perform hyperparameter tuning
            default_model_params: Dictionary of default parameters for LogisticRegression when tune_model is False
            test_size: Proportion of the dataset to include in the test split
            
        Returns:
            self: Returns self for method chaining
        """
        # Load data if DataFrame not provided
        if df is None:
            if data_file is None:
                raise ValueError("Either data_file or df must be provided")
            print(f"Loading data from {data_file}" + (f" (first {num_rows} rows)" if num_rows else ""))
            df = load_data(data_file, num_rows=num_rows)
        
        # Verify features exist in dataframe
        all_features = []
        if self.numeric_features:
            missing_numeric = [f for f in self.numeric_features if f not in df.columns]
            if missing_numeric:
                raise ValueError(f"Numeric features not found in dataframe: {missing_numeric}")
            all_features.extend(self.numeric_features)
        
        if self.categorical_features:
            missing_categorical = [f for f in self.categorical_features if f not in df.columns]
            if missing_categorical:
                raise ValueError(f"Categorical features not found in dataframe: {missing_categorical}")
            all_features.extend(self.categorical_features)
        
        if not all_features:
            raise ValueError("No features specified. Please provide numeric_features and/or categorical_features.")
        
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataframe.")
        
        print(f"Processing data with {len(self.numeric_features or [])} numeric and {len(self.categorical_features or [])} categorical features")
        
        # Process features
        feature_result = self._process_features(df)
        X = feature_result['X']
        self.feature_names = feature_result['feature_names']
        
        # Create preprocessor dictionary
        self.preprocessor = {
            'scaler': feature_result.get('scaler'),
            'encoder': feature_result.get('encoder'),
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'all_feature_names': feature_result.get('all_feature_names')
        }
        
        # Encode target variable
        encoder = LabelEncoder()
        y = encoder.fit_transform(df[self.target_column])
        self.preprocessor['target_encoder'] = encoder
        
        print(f"Final feature set shape: {X.shape}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Resample if specified
        if resample_flag:
            print("Resampling training data to handle class imbalance...")
            self.X_train, self.y_train = self._resample(self.X_train, self.y_train)
            print(f"Resampled data shape: {self.X_train.shape}")
            self.best_params = {'class_weight': 'balanced'}
        else:
            self.best_params = {}
        
        # Tune model if specified
        if tune_model:
            print("Tuning logistic regression hyperparameters...")
            self.model, grid_search, self.best_params = self._tune_lr(self.X_train, self.y_train, cv_folds=5)
        else:
            print("Training logistic regression with default parameters...")
            # Default model parameters if not provided
            if default_model_params is None:
                default_model_params = {
                    'C': 1.0,
                    'penalty': 'l2',
                    'solver': 'lbfgs',
                    'max_iter': 1000,
                    'class_weight': 'balanced' if resample_flag else None,
                    'random_state': self.random_state
                }
            else:
                # Ensure random_state is set
                if 'random_state' not in default_model_params:
                    default_model_params['random_state'] = self.random_state
                    
            print(f"Using model parameters: {default_model_params}")
            self.model = LogisticRegression(**default_model_params)
            self.model.fit(self.X_train, self.y_train)
            self.best_params = default_model_params
        
        # Make predictions on test set
        y_predictions = self.model.predict(self.X_test)
        y_probabilities = self.model.predict_proba(self.X_test)[:, 1]
        
        # Evaluate model
        print("Evaluating model performance...")
        self.metrics, markdown = evaluate_model(
            self.y_test, 
            y_predictions, 
            y_probabilities, 
            model_name="Logistic Regression",
            label_encoder=encoder
        )
        
        # Display results if in a notebook environment
        try:
            display(Markdown(markdown))
        except:
            print(markdown)
        
        return self
    
    def predict(self, 
               data_file=None,
               df=None,
               num_rows=None,
               return_probabilities=False):
        """
        Make predictions using the trained logistic regression model
        
        Args:
            data_file: Path to the data file (optional if df is provided)
            df: DataFrame to make predictions on (optional if data_file is provided)
            num_rows: Number of rows to load from data_file (None for all rows)
            return_probabilities: Whether to return probabilities instead of class predictions
            
        Returns:
            dict: Dictionary containing predictions and processed features
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        # Load data if DataFrame not provided
        if df is None:
            if data_file is None:
                raise ValueError("Either data_file or df must be provided")
            print(f"Loading data from {data_file}" + (f" (first {num_rows} rows)" if num_rows else ""))
            df = load_data(data_file, num_rows=num_rows)
        
        print(f"Processing data with {len(self.numeric_features or [])} numeric and {len(self.categorical_features or [])} categorical features")
        
        # Process features using the preprocessor
        feature_result = self._process_features(df, self.preprocessor)
        X = feature_result['X']
        
        print(f"Feature set shape: {X.shape}")
        
        # Make predictions
        if return_probabilities:
            predictions = self.model.predict_proba(X)[:, 1]
            print("Generated probability predictions")
        else:
            predictions = self.model.predict(X)
            print("Generated class predictions")
        
        # Convert predictions to original labels if encoder is available
        if 'target_encoder' in self.preprocessor and not return_probabilities:
            predictions = self.preprocessor['target_encoder'].inverse_transform(predictions)
            print("Converted predictions to original labels")
        
        # Return results
        return {
            'predictions': predictions,
            'X': X,
            'feature_names': feature_result['feature_names'],
            'df': df
        }
    
    def test(self, 
            data_file=None,
            df=None,
            num_rows=None):
        """
        Test the trained logistic regression model and evaluate its performance against true labels
        
        Args:
            data_file: Path to the data file (optional if df is provided)
            df: DataFrame to test on (optional if data_file is provided)
            num_rows: Number of rows to load from data_file (None for all rows)
            
        Returns:
            dict: Dictionary containing predictions, metrics, and processed features
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        # Load data if DataFrame not provided
        if df is None:
            if data_file is None:
                raise ValueError("Either data_file or df must be provided")
            print(f"Loading data from {data_file}" + (f" (first {num_rows} rows)" if num_rows else ""))
            df = load_data(data_file, num_rows=num_rows)
        
        # Check if target column exists
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataframe.")
        
        # Get expected features from model
        if hasattr(self.model, 'feature_names_in_'):
            expected_features = list(self.model.feature_names_in_)
            print(f"Using {len(expected_features)} features from trained model")
        else:
            expected_features = self.feature_names
            print(f"Using {len(expected_features or [])} features from stored feature list")
            
        print(f"Processing data with {len(self.numeric_features or [])} numeric and {len(self.categorical_features or [])} categorical features")
        
        # Process features using the preprocessor
        feature_result = self._process_features(df, self.preprocessor)
        X = feature_result['X']
        
        # Verify we have the correct features in the correct order
        if not all(X.columns == expected_features):
            raise ValueError("Feature mismatch after processing. This should not happen - please report this bug.")
        
        print(f"Feature set shape: {X.shape}")
        
        # Encode target variable
        if 'target_encoder' in self.preprocessor:
            encoder = self.preprocessor['target_encoder']
            y_true = encoder.transform(df[self.target_column])
        else:
            print("Warning: No target encoder found in preprocessor. Creating a new one.")
            encoder = LabelEncoder()
            y_true = encoder.fit_transform(df[self.target_column])
        
        # Make predictions
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        
        print("Generated predictions and probabilities")
        
        # Evaluate model
        print("Evaluating model performance...")
        self.metrics, markdown = evaluate_model(
            y_true, 
            y_pred, 
            y_prob, 
            model_name="Logistic Regression Test",
            label_encoder=encoder
        )
        
        # Display results if in a notebook environment
        try:
            display(Markdown(markdown))
        except:
            print(markdown)
        
        # Convert predictions to original labels
        predictions = encoder.inverse_transform(y_pred)
        
        # Return results
        return {
            'predictions': predictions,  # Original string labels
            'y_pred': y_pred,  # Encoded predictions (0/1)
            'probabilities': y_prob,
            'metrics': self.metrics,
            'markdown': markdown,
            'X': X,
            'y_true': y_true,
            'feature_names': list(X.columns),
            'df': df
        }
    
    def save(self, save_dir, model_name='lr_model'):
        """
        Save the logistic regression model and its preprocessor to disk
        
        Args:
            save_dir: Directory to save the model files
            model_name: Base name for the saved model files
            
        Returns:
            dict: Dictionary containing paths to saved files
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Paths for saved files
        model_path = os.path.join(save_dir, f"{model_name}_{timestamp}.pkl")
        preprocessor_path = os.path.join(save_dir, f"{model_name}_preprocessor_{timestamp}.pkl")
        metadata_path = os.path.join(save_dir, f"{model_name}_metadata_{timestamp}.json")
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {model_path}")
        
        # Save preprocessor and feature names
        preprocessor_data = {
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features
        }
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        print(f"Preprocessor and features saved to {preprocessor_path}")
        
        # Save metadata
        metadata = {
            'model_params': self.model.get_params(),
            'metrics': self.metrics,
            'feature_names': self.feature_names,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'timestamp': timestamp
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")
        
        return {
            'model_path': model_path,
            'preprocessor_path': preprocessor_path,
            'metadata_path': metadata_path,
            'timestamp': timestamp
        }
    
    @classmethod
    def load(cls, model_path, preprocessor_path):
        """
        Load a saved logistic regression model and its preprocessor from disk
        
        Args:
            model_path: Path to the saved model file
            preprocessor_path: Path to the saved preprocessor file
            
        Returns:
            LogisticRegressionModel: Initialized model instance
        """
        print(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        print(f"Loading preprocessor from {preprocessor_path}")
        with open(preprocessor_path, 'rb') as f:
            preprocessor_data = pickle.load(f)
            
        # Create new instance
        instance = cls()
        instance.model = model
        instance.preprocessor = preprocessor_data  # Old format stored preprocessor directly
        
        # Try to load metadata to get feature information
        try:
            metadata_path = preprocessor_path.replace('_preprocessor_', '_metadata_').replace('.pkl', '.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    instance.feature_names = metadata.get('feature_names')
                    instance.numeric_features = metadata.get('numeric_features')
                    instance.categorical_features = metadata.get('categorical_features')
        except Exception as e:
            print(f"Note: Could not load metadata file: {e}")
            
        return instance
    
    def plot_confusion_matrix(self, y_true=None, y_pred=None, figsize=(10, 8), cmap='Blues', normalize=False, title=None):
        """
        Plot confusion matrix for the model predictions.
        
        Args:
            y_true: True labels (if None, uses self.y_test)
            y_pred: Predicted labels (if None, generates predictions on self.X_test)
            figsize: Figure size as (width, height) tuple
            cmap: Colormap for the heatmap
            normalize: Whether to normalize the confusion matrix
            title: Title for the plot
            
        Returns:
            fig: The matplotlib figure
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        # Use test data if not provided
        if y_true is None:
            if self.y_test is None:
                raise ValueError("No test data available. Provide y_true or train the model first.")
            y_true = self.y_test
            
        if y_pred is None:
            if self.X_test is None:
                raise ValueError("No test data available. Provide y_pred or train the model first.")
            y_pred = self.model.predict(self.X_test)
        
        # Get class names if available
        class_names = ['HOME', 'ADMITTED']  # These are the known classes
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
                    xticklabels=class_names, 
                    yticklabels=class_names,
                    cbar=True, ax=ax)
        
        # Set labels
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # Set title
        if title is None:
            title = 'Confusion Matrix'
            if normalize:
                title += ' (Normalized)'
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, y_true=None, y_score=None, figsize=(10, 8), title=None):
        """
        Plot ROC curve for the model.
        
        Args:
            y_true: True labels (if None, uses self.y_test)
            y_score: Predicted probabilities (if None, generates predictions on self.X_test)
            figsize: Figure size as (width, height) tuple
            title: Title for the plot
            
        Returns:
            fig: The matplotlib figure
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        # Use test data if not provided
        if y_true is None:
            if self.y_test is None:
                raise ValueError("No test data available. Provide y_true or train the model first.")
            y_true = self.y_test
            
        if y_score is None:
            if self.X_test is None:
                raise ValueError("No test data available. Provide y_score or train the model first.")
            y_score = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Set labels and title
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        
        if title is None:
            title = 'Receiver Operating Characteristic (ROC) Curve'
        ax.set_title(title)
        
        ax.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curve(self, y_true=None, y_score=None, figsize=(10, 8), title=None):
        """
        Plot Precision-Recall curve for the model.
        
        Args:
            y_true: True labels (if None, uses self.y_test)
            y_score: Predicted probabilities (if None, generates predictions on self.X_test)
            figsize: Figure size as (width, height) tuple
            title: Title for the plot
            
        Returns:
            fig: The matplotlib figure
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        # Use test data if not provided
        if y_true is None:
            if self.y_test is None:
                raise ValueError("No test data available. Provide y_true or train the model first.")
            y_true = self.y_test
            
        if y_score is None:
            if self.X_test is None:
                raise ValueError("No test data available. Provide y_score or train the model first.")
            y_score = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot precision-recall curve
        ax.plot(recall, precision, lw=2, label=f'PR curve (area = {pr_auc:.3f})')
        
        # Set labels and title
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        
        if title is None:
            title = 'Precision-Recall Curve'
        ax.set_title(title)
        
        ax.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, top_n=20, figsize=(12, 10), title=None, X=None):
        """
        Plot feature importance for the logistic regression model.
        
        Args:
            top_n: Number of top features to display
            figsize: Figure size as (width, height) tuple
            title: Title for the plot
            X: Optional DataFrame containing feature names to use
            
        Returns:
            fig: The matplotlib figure
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model does not have feature coefficients.")
        
        # Get feature names from X if provided, otherwise use stored names
        if X is not None:
            feature_names = list(X.columns)
        else:
            feature_names = self.feature_names
            if feature_names is None or len(feature_names) == 0:
                feature_names = [f"Feature {i}" for i in range(self.model.coef_.shape[1])]
        
        # For binary classification, coefficients are in shape (1, n_features)
        coefs = self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_
        
        # Create DataFrame with feature names and coefficients
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(coefs)
        })
        
        # Sort by absolute importance and take top N
        importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot horizontal bar chart
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        
        # Set labels and title
        ax.set_xlabel('Absolute Coefficient Value')
        ax.set_ylabel('Feature')
        
        if title is None:
            title = f'Top {top_n} Feature Importance'
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def visualize(self, y_true=None, y_pred=None, y_score=None, figsize=(20, 15), X=None):
        """
        Create a comprehensive visualization dashboard for the model.
        
        Args:
            y_true: True labels (if None, uses self.y_test)
            y_pred: Predicted labels (if None, generates predictions on self.X_test)
            y_score: Predicted probabilities (if None, generates predictions on self.X_test)
            figsize: Figure size as (width, height) tuple
            X: Optional DataFrame containing feature names to use for feature importance plot
            
        Returns:
            fig: The matplotlib figure
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        # Use test data if not provided
        if y_true is None:
            if self.y_test is None:
                raise ValueError("No test data available. Provide y_true or train the model first.")
            y_true = self.y_test
            
        if y_pred is None:
            if self.X_test is None:
                raise ValueError("No test data available. Provide y_pred or train the model first.")
            y_pred = self.model.predict(self.X_test)
            
        if y_score is None:
            if self.X_test is None:
                raise ValueError("No test data available. Provide y_score or train the model first.")
            y_score = self.model.predict_proba(self.X_test)[:, 1]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        class_names = ['HOME', 'ADMITTED']  # Use known class names
            
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names,
                    cbar=True, ax=axes[0, 0])
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        axes[0, 0].set_title('Confusion Matrix')
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend(loc="lower right")
        
        # Plot precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        axes[1, 0].plot(recall, precision, color='darkorange', lw=2,
                       label=f'PR curve (AUC = {pr_auc:.2f})')
        axes[1, 0].set_xlim([0.0, 1.0])
        axes[1, 0].set_ylim([0.0, 1.05])
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].legend(loc="lower right")
        
        # Plot feature importance
        if hasattr(self.model, 'coef_'):
            # Get feature names from X if provided, otherwise use model's feature names
            if X is not None:
                feature_names = list(X.columns)
            elif hasattr(self.model, 'feature_names_in_'):
                feature_names = list(self.model.feature_names_in_)
            else:
                feature_names = self.feature_names
                if feature_names is None or len(feature_names) == 0:
                    feature_names = [f"Feature {i}" for i in range(self.model.coef_.shape[1])]
            
            # For binary classification, coefficients are in shape (1, n_features)
            coefs = self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_
            
            # Create DataFrame with feature names and coefficients
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.abs(coefs)
            })
            
            # Sort by absolute importance and take top 10
            importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
            
            # Plot horizontal bar chart
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=axes[1, 1])
            axes[1, 1].set_xlabel('Absolute Coefficient Value')
            axes[1, 1].set_ylabel('Feature')
            axes[1, 1].set_title('Top 10 Feature Importance')
        
        plt.tight_layout()
        return fig
    
    def get_feature_importance(self, X=None):
        """
        Get feature importance based on logistic regression coefficients.
        
        Args:
            X: Optional DataFrame containing feature names to use
            
        Returns:
            pd.DataFrame: DataFrame with feature names and their importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model does not have feature coefficients.")
        
        # Get feature names from X if provided, otherwise use stored names
        if X is not None:
            feature_names = list(X.columns)
        else:
            feature_names = self.feature_names
            if feature_names is None or len(feature_names) == 0:
                feature_names = [f"Feature {i}" for i in range(self.model.coef_.shape[1])]
        
        # For binary classification, coefficients are in shape (1, n_features)
        coefs = self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_
        
        # Create DataFrame with feature names and importance scores (absolute coefficients)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(coefs),
            'coefficient': coefs  # Keep the original coefficient value for reference
        })
        
        # Sort by absolute importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df

    def perform_ablation_study(self, X=None, y=None, feature_groups=None):
        """
        Perform ablation study by removing feature groups and measuring impact.
        
        Args:
            X (pd.DataFrame, optional): Feature matrix. If None, uses self.X_test
            y (array-like, optional): Target variable. If None, uses self.y_test
            feature_groups (dict, optional): Dictionary mapping group names to lists of features.
                If None, will use numeric and categorical features as groups.
                
        Returns:
            pd.DataFrame: DataFrame with ablation results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Use test data if not provided
        if X is None:
            if self.X_test is None:
                raise ValueError("No test data available. Provide X or train the model first.")
            X = self.X_test
            
        if y is None:
            if self.y_test is None:
                raise ValueError("No test data available. Provide y or train the model first.")
            y = self.y_test
        
        # If feature groups not provided, use numeric and categorical features
        if feature_groups is None:
            feature_groups = {}
            if self.numeric_features:
                feature_groups['numeric'] = self.numeric_features
            if self.categorical_features:
                feature_groups['categorical'] = self.categorical_features
        
        results = []
        
        # Baseline performance with all features
        baseline_metrics = {
            'accuracy': accuracy_score(y, self.model.predict(X)),
            'precision': precision_score(y, self.model.predict(X), average='binary'),
            'recall': recall_score(y, self.model.predict(X), average='binary'),
            'f1': f1_score(y, self.model.predict(X), average='binary'),
            'roc_auc': roc_auc_score(y, self.model.predict_proba(X)[:, 1])
        }
        
        results.append({
            'features_removed': 'None (Baseline)',
            **baseline_metrics,
            **{f'{metric}_change': 0.0 for metric in baseline_metrics}  # Zero change for baseline
        })
        
        # Test removing each feature group
        for group_name, features in feature_groups.items():
            print(f"Testing ablation of feature group: {group_name}")
            
            # Find columns to remove based on feature group
            if group_name == 'categorical':
                cols_to_remove = [col for col in X.columns 
                                if any(col.startswith(f"{feat}_") 
                                    for feat in features)]
            else:
                cols_to_remove = [col for col in X.columns 
                                if col in features]
            
            if not cols_to_remove:
                print(f"Warning: No columns found to remove for group: {group_name}")
                continue
            
            # Create copy of data without the feature group
            X_subset = X.drop(columns=cols_to_remove)
            
            # Create and train a new model with the same parameters
            model_params = self.model.get_params()
            ablation_model = LogisticRegression(**model_params)
            ablation_model.fit(X_subset, y)
            
            # Evaluate model
            metrics = {
                'accuracy': accuracy_score(y, ablation_model.predict(X_subset)),
                'precision': precision_score(y, ablation_model.predict(X_subset), average='binary'),
                'recall': recall_score(y, ablation_model.predict(X_subset), average='binary'),
                'f1': f1_score(y, ablation_model.predict(X_subset), average='binary'),
                'roc_auc': roc_auc_score(y, ablation_model.predict_proba(X_subset)[:, 1])
            }
            
            # Calculate performance change
            perf_change = {
                f'{metric}_change': metrics[metric] - baseline_metrics[metric]
                for metric in metrics
            }
            
            results.append({
                'features_removed': group_name,
                **metrics,
                **perf_change
            })
            
            print(f"Performance change after removing {group_name}: "
                f"accuracy_change={perf_change['accuracy_change']:.4f}, "
                f"f1_change={perf_change['f1_change']:.4f}")
        
        return pd.DataFrame(results)

    def perform_individual_feature_ablation(self, X=None, y=None, top_n=None):
        """
        Perform ablation study by removing each feature individually.
        
        Args:
            X (pd.DataFrame, optional): Feature matrix. If None, uses self.X_test
            y (array-like, optional): Target variable. If None, uses self.y_test
            top_n (int, optional): Number of top features to test. If None, tests all features.
                
        Returns:
            pd.DataFrame: DataFrame with ablation results for each feature
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Use test data if not provided
        if X is None:
            if self.X_test is None:
                raise ValueError("No test data available. Provide X or train the model first.")
            X = self.X_test
            
        if y is None:
            if self.y_test is None:
                raise ValueError("No test data available. Provide y or train the model first.")
            y = self.y_test
        
        results = []
        
        # Baseline performance with all features
        baseline_metrics = {
            'accuracy': accuracy_score(y, self.model.predict(X)),
            'precision': precision_score(y, self.model.predict(X), average='binary'),
            'recall': recall_score(y, self.model.predict(X), average='binary'),
            'f1': f1_score(y, self.model.predict(X), average='binary'),
            'roc_auc': roc_auc_score(y, self.model.predict_proba(X)[:, 1])
        }
        
        results.append({
            'feature_removed': 'None (Baseline)',
            'feature_group': 'baseline',
            **baseline_metrics,
            **{f'{metric}_change': 0.0 for metric in baseline_metrics}  # Zero change for baseline
        })
        
        # Get feature importance for prioritizing features to test
        feature_importance = self.get_feature_importance()
        feature_list = list(feature_importance['feature'])
        
        # Limit to top_n if specified
        if top_n is not None and top_n < len(feature_list):
            feature_list = feature_list[:top_n]
        
        # Test removing each feature individually
        for feature in feature_list:
            print(f"Testing ablation of feature: {feature}")
            
            # Determine feature group (numeric or categorical)
            feature_group = 'unknown'
            if self.numeric_features and any(feature == nf for nf in self.numeric_features):
                feature_group = 'numeric'
            elif self.categorical_features and any(feature.startswith(f"{cf}_") for cf in self.categorical_features):
                feature_group = 'categorical'
                orig_feature = next((cf for cf in self.categorical_features if feature.startswith(f"{cf}_")), feature)
            else:
                orig_feature = feature
            
            # Create copy of data without the feature
            X_subset = X.drop(columns=[feature])
            
            # Create and train a new model with the same parameters
            model_params = self.model.get_params()
            ablation_model = LogisticRegression(**model_params)
            ablation_model.fit(X_subset, y)
            
            # Evaluate model
            metrics = {
                'accuracy': accuracy_score(y, ablation_model.predict(X_subset)),
                'precision': precision_score(y, ablation_model.predict(X_subset), average='binary'),
                'recall': recall_score(y, ablation_model.predict(X_subset), average='binary'),
                'f1': f1_score(y, ablation_model.predict(X_subset), average='binary'),
                'roc_auc': roc_auc_score(y, ablation_model.predict_proba(X_subset)[:, 1])
            }
            
            # Calculate performance change
            perf_change = {
                f'{metric}_change': metrics[metric] - baseline_metrics[metric]
                for metric in metrics
            }
            
            results.append({
                'feature_removed': orig_feature if feature_group == 'categorical' else feature,
                'feature_group': feature_group,
                **metrics,
                **perf_change
            })
        
        # Sort results by performance impact (roc_auc change)
        results_df = pd.DataFrame(results)
        if len(results_df) > 1:  # Only sort if we have results beyond baseline
            baseline_row = results_df.iloc[0:1]  # Keep baseline at top
            remaining_rows = results_df.iloc[1:].sort_values('roc_auc_change')  # Sort rest by impact
            results_df = pd.concat([baseline_row, remaining_rows])
        
        return results_df

    def compare_sampling_strategies(self, X=None, y=None, cv=5):
        """
        Compare different sampling strategies (None, SMOTE, random over/undersampling) 
        using cross-validation.
        
        Args:
            X (pd.DataFrame, optional): Feature matrix. If None, uses self.X_train
            y (array-like, optional): Target variable. If None, uses self.y_train
            cv (int): Number of cross-validation folds
            
        Returns:
            tuple: (results_dict, best_results_dict) with metrics for each strategy
        """
        from collections import defaultdict
        from sklearn.model_selection import cross_validate
        from imblearn.pipeline import Pipeline
        from imblearn.over_sampling import SMOTE, RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Use training data if not provided
        if X is None:
            if self.X_train is None:
                raise ValueError("No training data available. Provide X or train the model first.")
            X = self.X_train
            
        if y is None:
            if self.y_train is None:
                raise ValueError("No training data available. Provide y or train the model first.")
            y = self.y_train
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        # Define sampling strategies
        sampling_strategies = {
            'none': None,
            'smote': SMOTE(random_state=self.random_state),
            'over': RandomOverSampler(random_state=self.random_state),
            'under': RandomUnderSampler(random_state=self.random_state)
        }
        
        results = defaultdict(dict)
        
        # Get model parameters
        model_params = self.model.get_params()
        
        # Evaluate each strategy
        for strategy_name, sampler in sampling_strategies.items():
            print(f"\nEvaluating {strategy_name} sampling...")
            
            if sampler is None:
                # No sampling
                model = LogisticRegression(**model_params)
                cv_results = cross_validate(
                    model, X, y,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=-1,
                    return_train_score=True
                )
            else:
                # With sampling
                pipeline = Pipeline([
                    ('sampler', sampler),
                    ('classifier', LogisticRegression(**model_params))
                ])
                cv_results = cross_validate(
                    pipeline, X, y,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=-1,
                    return_train_score=True
                )
            
            # Store results
            for metric in scoring.keys():
                test_scores = cv_results[f'test_{metric}']
                train_scores = cv_results[f'train_{metric}']
                
                results[strategy_name][metric] = {
                    'test_mean': np.mean(test_scores),
                    'test_std': np.std(test_scores),
                    'train_mean': np.mean(train_scores),
                    'train_std': np.std(train_scores)
                }
                
            # Print results
            print(f"\n{strategy_name.capitalize()} Sampling Results:")
            for metric, values in results[strategy_name].items():
                print(f"{metric}: {values['test_mean']:.3f} (+/- {values['test_std']:.3f})")
        
        # Find best strategy based on F1 score
        best_strategy = max(results.keys(), 
                        key=lambda k: results[k]['f1']['test_mean'])
        
        best_results = {
            'strategy': best_strategy,
            'metrics': results[best_strategy]
        }
        
        print(f"\nBest strategy: {best_strategy}")
        print("Best strategy metrics:")
        for metric, values in best_results['metrics'].items():
            print(f"{metric}: {values['test_mean']:.3f} (+/- {values['test_std']:.3f})")
        
        return results, best_results

    def perform_sensitivity_analysis(self, X=None, y=None, param_ranges=None):
        """
        Perform sensitivity analysis by testing different hyperparameter values.
        
        Args:
            X (pd.DataFrame, optional): Feature matrix. If None, uses self.X_train
            y (array-like, optional): Target variable. If None, uses self.y_train
            param_ranges (dict, optional): Dictionary mapping parameter names to lists of values.
                If None, uses default ranges.
                
        Returns:
            pd.DataFrame: DataFrame with sensitivity analysis results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Use training data if not provided
        if X is None:
            if self.X_train is None:
                raise ValueError("No training data available. Provide X or train the model first.")
            X = self.X_train
            
        if y is None:
            if self.y_train is None:
                raise ValueError("No training data available. Provide y or train the model first.")
            y = self.y_train
        
        # Default parameter ranges if not provided
        if param_ranges is None:
            param_ranges = {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'lbfgs', 'saga'],
                'class_weight': [None, 'balanced']
            }
        
        results = []
        
        # Get current model parameters
        base_params = self.model.get_params()
        
        # Get baseline performance with current model
        baseline_metrics = {
            'accuracy': accuracy_score(y, self.model.predict(X)),
            'precision': precision_score(y, self.model.predict(X), average='binary'),
            'recall': recall_score(y, self.model.predict(X), average='binary'),
            'f1': f1_score(y, self.model.predict(X), average='binary'),
            'roc_auc': roc_auc_score(y, self.model.predict_proba(X)[:, 1])
        }
        
        results.append({
            'parameter': 'baseline',
            'value': 'current',
            **baseline_metrics,
            **{f'{metric}_change': 0.0 for metric in baseline_metrics}  # Zero change for baseline
        })
        
        # Test each parameter
        for param_name, param_values in param_ranges.items():
            print(f"Testing parameter: {param_name} with values: {param_values}")
            
            for value in param_values:
                # Skip if this value is the same as current param value
                if base_params.get(param_name) == value:
                    continue
                    
                # Create new parameters dictionary for this test
                test_params = base_params.copy()
                test_params[param_name] = value
                
                # Skip invalid combinations
                if param_name == 'penalty' and value == 'l1' and test_params.get('solver') not in ['liblinear', 'saga']:
                    print(f"Skipping invalid combination: {param_name}={value} with solver={test_params.get('solver')}")
                    continue
                
                # Create and train a new model with test parameters
                try:
                    sensitivity_model = LogisticRegression(**test_params)
                    sensitivity_model.fit(X, y)
                    
                    # Evaluate model
                    metrics = {
                        'accuracy': accuracy_score(y, sensitivity_model.predict(X)),
                        'precision': precision_score(y, sensitivity_model.predict(X), average='binary'),
                        'recall': recall_score(y, sensitivity_model.predict(X), average='binary'),
                        'f1': f1_score(y, sensitivity_model.predict(X), average='binary'),
                        'roc_auc': roc_auc_score(y, sensitivity_model.predict_proba(X)[:, 1])
                    }
                    
                    # Calculate performance change
                    perf_change = {
                        f'{metric}_change': metrics[metric] - baseline_metrics[metric]
                        for metric in metrics
                    }
                    
                    results.append({
                        'parameter': param_name,
                        'value': str(value),  # Convert to string for consistent representation
                        **metrics,
                        **perf_change
                    })
                    
                except Exception as e:
                    print(f"Error with {param_name}={value}: {str(e)}")
        
        # Sort results by ROC-AUC score
        results_df = pd.DataFrame(results)
        if len(results_df) > 1:  # Only sort if we have results beyond baseline
            baseline_row = results_df.iloc[0:1]  # Keep baseline at top
            remaining_rows = results_df.iloc[1:].sort_values('roc_auc', ascending=False)  # Sort rest by performance
            results_df = pd.concat([baseline_row, remaining_rows])
        
        return results_df

    def retrain_with_recommendations(self, recommendations, X=None, y=None, test_size=0.1):
        """
        Retrain the model using recommendations from analysis.
        
        Args:
            recommendations (dict): Dictionary of recommendations with the following possible keys:
                - 'model_params': Dict of model parameters to use
                - 'sampling_strategy': Sampling strategy to use (none, smote, over, under)
                - 'features_to_keep': List of features to keep (if None, keeps all features)
            X (pd.DataFrame, optional): Feature matrix. If None, uses current data
            y (array-like, optional): Target variable. If None, uses current data
            test_size (float): Proportion of the dataset to include in the test split
            
        Returns:
            self: Returns self for method chaining
        """
        print("Retraining model with recommendations...")
        
        # Get current data if not provided
        X_current = X or pd.concat([self.X_train, self.X_test]) if self.X_train is not None else None
        y_current = y or np.concatenate([self.y_train, self.y_test]) if self.y_train is not None else None
        
        if X_current is None or y_current is None:
            raise ValueError("No data available. Provide X and y or train the model first.")
        
        # Apply feature selection if specified
        features_to_keep = recommendations.get('features_to_keep')
        if features_to_keep is not None:
            X_current = X_current[features_to_keep]
            print(f"Selected {len(features_to_keep)} features.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_current, y_current, test_size=test_size, random_state=self.random_state
        )
        
        # Apply sampling if specified
        sampling_strategy = recommendations.get('sampling_strategy')
        if sampling_strategy and sampling_strategy != 'none':
            print(f"Applying {sampling_strategy} sampling...")
            
            if sampling_strategy == 'smote':
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(random_state=self.random_state)
            elif sampling_strategy == 'over':
                from imblearn.over_sampling import RandomOverSampler
                sampler = RandomOverSampler(random_state=self.random_state)
            elif sampling_strategy == 'under':
                from imblearn.under_sampling import RandomUnderSampler
                sampler = RandomUnderSampler(random_state=self.random_state)
            
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            print(f"Resampled training data shape: {X_train.shape}")
        
        # Get model parameters
        model_params = recommendations.get('model_params', self.model.get_params())
        print(f"Using model parameters: {model_params}")
        
        # Create and train new model
        self.model = LogisticRegression(**model_params)
        self.model.fit(X_train, y_train)
        
        # Update instance variables
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_params = model_params
        
        # Make predictions on test set
        y_predictions = self.model.predict(X_test)
        y_probabilities = self.model.predict_proba(X_test)[:, 1]
        
        # Evaluate model
        print("Evaluating retrained model performance...")
        target_encoder = self.preprocessor.get('target_encoder') if hasattr(self, 'preprocessor') else None
        
        self.metrics, markdown = evaluate_model(
            y_test, 
            y_predictions, 
            y_probabilities, 
            model_name="Retrained Logistic Regression",
            label_encoder=target_encoder
        )
        
        # Display results if in a notebook environment
        try:
            display(Markdown(markdown))
        except:
            print(markdown)
        
        return self
    
    def analyze_errors(self, df, y_true=None, y_pred=None, y_prob=None, processed_features=None):
        """
        Analyze misclassified cases in detail, preserving all data fields for analysis
        
        Args:
            df: DataFrame containing all features and data
            y_true: True labels (if None, uses self.y_test)
            y_pred: Predicted labels (if None, generates predictions)
            y_prob: Prediction probabilities (if None, generates probabilities)
            processed_features: Optional DataFrame of processed features used by the model
            
        Returns:
            Dictionary containing:
                - false_positives: Cases predicted as ADMITTED but actually went HOME
                - false_negatives: Cases predicted as HOME but actually ADMITTED
                - high_confidence_errors: Cases with prediction probability > 0.8 that were wrong
                - summary: Dictionary of error counts and rates
        """
        if y_true is None:
            if self.y_test is None:
                raise ValueError("No test data available. Provide y_true or test the model first.")
            y_true = self.y_test
            
        if y_pred is None or y_prob is None:
            if self.X_test is None:
                raise ValueError("No test data available. Provide predictions or test the model first.")
            y_pred = self.model.predict(self.X_test)
            y_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        # Create DataFrame with predictions
        analysis_df = df.copy()  # Keep all original columns
        analysis_df['y_true'] = y_true  # Use y_true instead of true_label
        analysis_df['y_pred'] = y_pred  # Use y_pred instead of predicted_label
        analysis_df['prediction_probability'] = y_prob
        analysis_df['is_error'] = y_true != y_pred
        
        # Add processed features if provided
        if processed_features is not None:
            for col in processed_features.columns:
                analysis_df[f'processed_{col}'] = processed_features[col]
        
        # Separate false positives and negatives
        false_positives = analysis_df[(analysis_df['y_true'] == 0) & (analysis_df['y_pred'] == 1)]
        false_negatives = analysis_df[(analysis_df['y_true'] == 1) & (analysis_df['y_pred'] == 0)]
        
        # Analyze high confidence errors
        high_conf_errors = analysis_df[
            (analysis_df['is_error']) & 
            (analysis_df['prediction_probability'] > 0.8)
        ]
        
        # Calculate error statistics using numpy arrays
        total_errors = len(false_positives) + len(false_negatives)
        error_rate = total_errors / len(df) * 100
        
        # Calculate rates using numpy arrays
        n_negatives = (y_true == 0).sum()
        n_positives = (y_true == 1).sum()
        fp_rate = len(false_positives) / n_negatives * 100 if n_negatives > 0 else 0
        fn_rate = len(false_negatives) / n_positives * 100 if n_positives > 0 else 0
        
        summary = {
            'total_cases': len(df),
            'total_errors': total_errors,
            'error_rate': error_rate,
            'false_positives': len(false_positives),
            'false_positive_rate': fp_rate,
            'false_negatives': len(false_negatives),
            'false_negative_rate': fn_rate,
            'high_confidence_errors': len(high_conf_errors)
        }
        
        # Print summary
        print("\nError Analysis Summary:")
        print(f"Total cases analyzed: {summary['total_cases']}")
        print(f"Total errors: {summary['total_errors']} ({summary['error_rate']:.1f}%)")
        print(f"False positives: {summary['false_positives']} ({summary['false_positive_rate']:.1f}% of negative cases)")
        print(f"False negatives: {summary['false_negatives']} ({summary['false_negative_rate']:.1f}% of positive cases)")
        print(f"High confidence errors (prob > 0.8): {summary['high_confidence_errors']}")
        
        return {
            'false_positives': false_positives.sort_values('prediction_probability', ascending=False),
            'false_negatives': false_negatives.sort_values('prediction_probability', ascending=False),
            'high_confidence_errors': high_conf_errors.sort_values('prediction_probability', ascending=False),
            'summary': summary
        }