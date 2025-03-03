import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from IPython.display import display, Markdown

from utils.data_utils import load_data
from utils.evaluation import evaluate_model

# Set up logger
logger = logging.getLogger(__name__)

def resample(X, y):
    """
    Undersample to match classes 1:1
    
    Args:
        X: Feature matrix
        y: Target variable
        
    Returns:
        X_resampled, y_resampled: Resampled data
    """
    sampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    logger.info('Class distribution before sampling: %s', np.bincount(y))
    logger.info('Class distribution after sampling: %s', np.bincount(y_resampled))
    return X_resampled, y_resampled

def tune_lr(X_train, y_train, cv_folds=5, param_grid=None):
    """
    Train a Logistic Regression model using k-fold cross-validation with hyperparameter tuning.

    Parameters:
    - X_train: Training features (DataFrame or array)
    - y_train: Training labels (Series or array)
    - cv_folds: Number of folds for cross-validation.
    - param_grid: Dictionary of hyperparameters to search over.

    Returns:
    - best_model: The best Logistic Regression model found via GridSearchCV.
    - grid_search: The fitted GridSearchCV object (for further inspection if needed).
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
    lr = LogisticRegression(random_state=42)

    # Setup k-fold cross-validation
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

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

def process_features(df, numeric_features=None, categorical_features=None, preprocessor=None):
    """
    Process features for logistic regression model.
    
    Args:
        df: Input DataFrame
        numeric_features: List of numeric feature column names
        categorical_features: List of categorical feature column names
        preprocessor: Dictionary containing preprocessing objects (for inference)
        
    Returns:
        dict: Dictionary containing processed features and preprocessing objects
    """
    result = {}
    
    # Initialize DataFrames with the same index as input
    numeric_df = pd.DataFrame(index=df.index)
    categorical_df = pd.DataFrame(index=df.index)
    
    # Process numeric features
    if numeric_features:
        # Determine which numeric features are available in the dataframe
        available_numeric = [f for f in numeric_features if f in df.columns]
        missing_numeric = [f for f in numeric_features if f not in df.columns]
        
        if missing_numeric:
            print(f"Warning: The following numeric features are not in the dataframe: {missing_numeric}")
        
        # Scale available numeric features
        if available_numeric:
            # Extract numeric features
            numeric_data = df[available_numeric].copy()
            
            # Handle missing values in numeric features
            for col in numeric_data.columns:
                if numeric_data[col].isna().any():
                    print(f"Imputing missing values for {col} with median")
                    numeric_data[col] = numeric_data[col].fillna(numeric_data[col].median())
            
            # Scale numeric features
            if preprocessor and 'scaler' in preprocessor:
                # Use existing scaler for inference
                scaler = preprocessor['scaler']
                numeric_scaled = scaler.transform(numeric_data)
            else:
                # Create new scaler for training
                scaler = StandardScaler()
                numeric_scaled = scaler.fit_transform(numeric_data)
                result['scaler'] = scaler
            
            # Create DataFrame with scaled values
            numeric_df = pd.DataFrame(numeric_scaled, columns=available_numeric, index=df.index)
        
        # Add missing numeric features with zeros
        for col in missing_numeric:
            print(f"Adding missing numeric feature column: {col}")
            numeric_df[col] = 0.0
    
    # Process categorical features
    encoders = {}
    encoded_feature_names = []
    
    if categorical_features:
        # Determine which categorical features are available in the dataframe
        available_categorical = [f for f in categorical_features if f in df.columns]
        missing_categorical = [f for f in categorical_features if f not in df.columns]
        
        if missing_categorical:
            print(f"Warning: The following categorical features are not in the dataframe: {missing_categorical}")
        
        # Process available categorical features
        for feature in available_categorical:
            # Fill missing values with most frequent value
            if df[feature].isna().any():
                most_frequent = df[feature].mode()[0]
                print(f"Imputing missing values for {feature} with mode: {most_frequent}")
                df[feature] = df[feature].fillna(most_frequent)
            
            # One-hot encode the feature
            if preprocessor and 'encoders' in preprocessor and feature in preprocessor['encoders']:
                # Use existing encoder for inference
                encoder = preprocessor['encoders'][feature]
                feature_array = df[feature].values.reshape(-1, 1)
                encoded = encoder.transform(feature_array)
                
                # Get categories from encoder
                categories = encoder.categories_[0]
            else:
                # Create new encoder for training
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                feature_array = df[feature].values.reshape(-1, 1)
                encoded = encoder.fit_transform(feature_array)
                
                # Get categories from encoder
                categories = encoder.categories_[0]
                encoders[feature] = encoder
            
            # Create column names for encoded features
            feature_names = [f"{feature}_{cat}" for cat in categories]
            
            # Create DataFrame with encoded values
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
            
            # Add to the categorical dataframe
            categorical_df = pd.concat([categorical_df, encoded_df], axis=1)
            encoded_feature_names.extend(feature_names)
        
        # For missing categorical features in inference mode, use dummy encoders
        if preprocessor and 'encoders' in preprocessor:
            for feature in missing_categorical:
                if feature in preprocessor['encoders']:
                    encoder = preprocessor['encoders'][feature]
                    # Get expected column names from encoder
                    categories = encoder.categories_[0]
                    feature_names = [f"{feature}_{cat}" for cat in categories]
                    
                    # Add columns with zeros for each category
                    for col in feature_names:
                        categorical_df[col] = 0.0
                    
                    encoded_feature_names.extend(feature_names)
    
    # Store encoders if in training mode
    if encoders:
        result['encoders'] = encoders
    
    # Combine numeric and categorical features
    X = pd.concat([numeric_df, categorical_df], axis=1)
    
    # Ensure consistent column order if preprocessor is provided
    if preprocessor and 'all_feature_names' in preprocessor:
        expected_columns = preprocessor['all_feature_names']
        # Add missing columns with zeros
        for col in expected_columns:
            if col not in X.columns:
                X[col] = 0.0
        # Reorder columns to match expected order
        X = X[expected_columns]
        feature_names = expected_columns
    else:
        # In training mode, just use the current feature names
        feature_names = list(X.columns)
        result['all_feature_names'] = feature_names
    
    # Store feature names
    result['X'] = X
    result['feature_names'] = feature_names
    
    return result

def train_lr_model(
    data_file='train.csv', 
    num_rows=None, 
    numeric_features=None, 
    categorical_features=None,
    target_column='disposition',
    resample_flag=False, 
    tune_model=True, 
    default_model_params=None,
    test_size=0.1, 
    random_state=42
):
    """
    Train a logistic regression model with specified features
    
    Args:
        data_file: Path to the data file
        num_rows: Number of rows to load (None for all rows)
        numeric_features: List of numeric feature column names to use
        categorical_features: List of categorical feature column names to use
        target_column: Name of the target column
        resample_flag: Whether to resample the data to handle class imbalance
        tune_model: Whether to perform hyperparameter tuning
        default_model_params: Dictionary of default parameters for LogisticRegression when tune_model is False
        test_size: Proportion of the dataset to include in the test split
        random_state: Random state for reproducibility
        
    Returns:
        dict: Dictionary containing model, metrics, and preprocessing objects
    """
    print(f"Loading data from {data_file}" + (f" (first {num_rows} rows)" if num_rows else ""))
    
    # Load data
    df = load_data(data_file, num_rows=num_rows)
    
    # Verify features exist in dataframe
    all_features = []
    if numeric_features:
        missing_numeric = [f for f in numeric_features if f not in df.columns]
        if missing_numeric:
            raise ValueError(f"Numeric features not found in dataframe: {missing_numeric}")
        all_features.extend(numeric_features)
    
    if categorical_features:
        missing_categorical = [f for f in categorical_features if f not in df.columns]
        if missing_categorical:
            raise ValueError(f"Categorical features not found in dataframe: {missing_categorical}")
        all_features.extend(categorical_features)
    
    if not all_features:
        raise ValueError("No features specified. Please provide numeric_features and/or categorical_features.")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")
    
    print(f"Processing data with {len(numeric_features or [])} numeric and {len(categorical_features or [])} categorical features")
    
    # Process features
    feature_result = process_features(df, numeric_features, categorical_features)
    X = feature_result['X']
    feature_names = feature_result['feature_names']
    
    # Create preprocessor dictionary
    preprocessor = {
        'scaler': feature_result.get('scaler'),
        'encoders': feature_result.get('encoders'),
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'all_feature_names': feature_result.get('all_feature_names')
    }
    
    # Encode target variable
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[target_column])
    preprocessor['target_encoder'] = encoder
    
    print(f"Final feature set shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Resample if specified
    if resample_flag:
        print("Resampling training data to handle class imbalance...")
        X_train, y_train = resample(X_train, y_train)
        print(f"Resampled data shape: {X_train.shape}")
        best_params = {'class_weight': 'balanced'}
    else:
        best_params = {}
    
    # Tune model if specified
    if tune_model:
        print("Tuning logistic regression hyperparameters...")
        best_model, grid_search, best_params = tune_lr(X_train, y_train, cv_folds=5)
        lr_model = best_model
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
                'random_state': random_state
            }
        else:
            # Ensure random_state is set
            if 'random_state' not in default_model_params:
                default_model_params['random_state'] = random_state
                
        print(f"Using model parameters: {default_model_params}")
        lr_model = LogisticRegression(**default_model_params)
        lr_model.fit(X_train, y_train)
        best_params = default_model_params
    
    # Make predictions
    y_predictions = lr_model.predict(X_test)
    y_probabilities = lr_model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    print("Evaluating model performance...")
    metrics, markdown = evaluate_model(
        y_test, 
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
    
    # Return results
    return {
        'model': lr_model,
        'metrics': metrics,
        'markdown': markdown,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'preprocessor': preprocessor,
        'best_params': best_params
    }

def predict_with_lr(
    model,
    preprocessor,
    data_file=None,
    df=None,
    num_rows=None,
    return_probabilities=False
):
    """
    Make predictions using a trained logistic regression model
    
    Args:
        model: Trained logistic regression model
        preprocessor: Dictionary containing preprocessing objects
        data_file: Path to the data file (optional if df is provided)
        df: DataFrame to make predictions on (optional if data_file is provided)
        num_rows: Number of rows to load from data_file (None for all rows)
        return_probabilities: Whether to return probabilities instead of class predictions
        
    Returns:
        dict: Dictionary containing predictions and processed features
    """
    # Load data if DataFrame not provided
    if df is None:
        if data_file is None:
            raise ValueError("Either data_file or df must be provided")
        print(f"Loading data from {data_file}" + (f" (first {num_rows} rows)" if num_rows else ""))
        df = load_data(data_file, num_rows=num_rows)
    
    # Extract feature lists from preprocessor
    numeric_features = preprocessor.get('numeric_features')
    categorical_features = preprocessor.get('categorical_features')
    
    print(f"Processing data with {len(numeric_features or [])} numeric and {len(categorical_features or [])} categorical features")
    
    # Process features using the preprocessor
    feature_result = process_features(df, numeric_features, categorical_features, preprocessor)
    X = feature_result['X']
    
    print(f"Feature set shape: {X.shape}")
    
    # Make predictions
    if return_probabilities:
        predictions = model.predict_proba(X)[:, 1]
        print("Generated probability predictions")
    else:
        predictions = model.predict(X)
        print("Generated class predictions")
    
    # Convert predictions to original labels if encoder is available
    if 'target_encoder' in preprocessor and not return_probabilities:
        predictions = preprocessor['target_encoder'].inverse_transform(predictions)
        print("Converted predictions to original labels")
    
    # Return results
    return {
        'predictions': predictions,
        'X': X,
        'feature_names': feature_result['feature_names'],
        'df': df
    }

def test_with_lr(
    model,
    preprocessor,
    data_file=None,
    df=None,
    num_rows=None,
    target_column='disposition'
):
    """
    Test a trained logistic regression model and evaluate its performance against true labels
    
    Args:
        model: Trained logistic regression model
        preprocessor: Dictionary containing preprocessing objects
        data_file: Path to the data file (optional if df is provided)
        df: DataFrame to test on (optional if data_file is provided)
        num_rows: Number of rows to load from data_file (None for all rows)
        target_column: Name of the target column containing true labels
        
    Returns:
        dict: Dictionary containing predictions, metrics, and processed features
    """
    # Load data if DataFrame not provided
    if df is None:
        if data_file is None:
            raise ValueError("Either data_file or df must be provided")
        print(f"Loading data from {data_file}" + (f" (first {num_rows} rows)" if num_rows else ""))
        df = load_data(data_file, num_rows=num_rows)
    
    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")
    
    # Extract feature lists from preprocessor
    numeric_features = preprocessor.get('numeric_features')
    categorical_features = preprocessor.get('categorical_features')
    
    print(f"Processing data with {len(numeric_features or [])} numeric and {len(categorical_features or [])} categorical features")
    
    # Process features using the preprocessor
    feature_result = process_features(df, numeric_features, categorical_features, preprocessor)
    X = feature_result['X']
    
    print(f"Feature set shape: {X.shape}")
    
    # Encode target variable using the encoder from preprocessor
    if 'target_encoder' in preprocessor:
        encoder = preprocessor['target_encoder']
        y_true = encoder.transform(df[target_column])
    else:
        # If no encoder in preprocessor, create a new one (not ideal, but fallback)
        print("Warning: No target encoder found in preprocessor. Creating a new one.")
        encoder = LabelEncoder()
        y_true = encoder.fit_transform(df[target_column])
    
    # Make predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    print("Generated predictions and probabilities")
    
    # Evaluate model
    print("Evaluating model performance...")
    metrics, markdown = evaluate_model(
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
        'predictions': predictions,
        'probabilities': y_prob,
        'metrics': metrics,
        'markdown': markdown,
        'X': X,
        'y_true': y_true,
        'feature_names': feature_result['feature_names'],
        'df': df
    }

def save_lr_model(model_data, save_dir, model_name='lr_model'):
    """
    Save the logistic regression model and its preprocessor to disk
    
    Args:
        model_data: Dictionary containing model, preprocessor, and other data (as returned by train_lr_model)
        save_dir: Directory to save the model files
        model_name: Base name for the saved model files
        
    Returns:
        dict: Dictionary containing paths to saved files
    """
    import os
    import pickle
    import json
    from datetime import datetime
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Paths for saved files
    model_path = os.path.join(save_dir, f"{model_name}_{timestamp}.pkl")
    preprocessor_path = os.path.join(save_dir, f"{model_name}_preprocessor_{timestamp}.pkl")
    metadata_path = os.path.join(save_dir, f"{model_name}_metadata_{timestamp}.json")
    
    # Extract components to save
    model = model_data.get('model')
    preprocessor = model_data.get('preprocessor')
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    # Save preprocessor
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor saved to {preprocessor_path}")
    
    # Create metadata
    metadata = {
        'model_type': 'LogisticRegression',
        'timestamp': timestamp,
        'feature_count': len(model_data.get('feature_names', [])),
        'metrics': model_data.get('metrics', {}),
        'best_params': model_data.get('best_params', {}),
        'model_path': os.path.basename(model_path),
        'preprocessor_path': os.path.basename(preprocessor_path)
    }
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")
    
    return {
        'model_path': model_path,
        'preprocessor_path': preprocessor_path,
        'metadata_path': metadata_path,
        'timestamp': timestamp
    }

def load_lr_model(model_path, preprocessor_path):
    """
    Load a saved logistic regression model and its preprocessor from disk
    
    Args:
        model_path: Path to the saved model file
        preprocessor_path: Path to the saved preprocessor file
        
    Returns:
        dict: Dictionary containing loaded model and preprocessor
    """
    import pickle
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {model_path}")
    
    # Load preprocessor
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    print(f"Preprocessor loaded from {preprocessor_path}")
    
    return {
        'model': model,
        'preprocessor': preprocessor
    }
