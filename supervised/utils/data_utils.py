import os
import hashlib
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm as tqdm_func
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from utils.complexity import ComplexityFeatures


def get_cache_key(texts, sample_size=None):
    """
    Generate a unique cache key based on the input texts and sample size
    """
    # Sort texts to ensure same key for same content regardless of order
    sorted_texts = sorted([str(t) for t in texts])
    # Create a string combining texts and sample size
    content = ''.join(sorted_texts) + str(sample_size)
    # Generate MD5 hash as cache key
    return hashlib.md5(content.encode()).hexdigest()


def load_data(file_path, num_rows=None):
    """
    Load data from CSV file with optional row sampling and NA removal
    
    Args:
        file_path (str): Path to the CSV file
        num_rows (int, optional): Number of rows to sample. If None, all rows are returned
        
    Returns:
        pd.DataFrame: Processed DataFrame with NaN values removed from chiefcomplaint
    """
    # Load specific columns
    df = pd.read_csv(file_path)
    
    # Sample rows if num_rows is specified
    if num_rows is not None:
        df = df.sample(n=num_rows, random_state=42)
    
    # Remove NaN values and reset index
    df = df.dropna(subset=['chiefcomplaint']).reset_index(drop=True)
    
    return df

def process_data_for_xgboost(embeddings, column, pca_obj=None, n_components=80, training=False):
    """
    Process data for XGBoost model, including scaling, PCA transformation, and label encoding.

    Args:
        embeddings: Embeddings to process
        column: Target column (disposition)
        pca_obj: PCA object to use for transformation. If None and training=True, a new one will be created.
        n_components: Number of PCA components (only used if creating a new PCA object)
        training: Whether this is for training (True) or inference (False)

    Returns:
        Dictionary containing processed data and preprocessing objects
    """
    # Scale the embeddings
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)

    # Apply PCA transformation
    if pca_obj is None and training:
        # Create a new PCA object for training
        pca_obj = PCA(n_components=n_components, random_state=42)
        X_pca = pca_obj.fit_transform(X_scaled)
    elif pca_obj is not None:
        # Use existing PCA object for inference
        X_pca = pca_obj.transform(X_scaled)
    else:
        raise ValueError("For inference, a pre-trained PCA object must be provided")

    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(column)

    return {
        'X': X_pca,
        'y': y,
        'scaler': scaler,
        'encoder': encoder,
        'pca': pca_obj
    }

def process_data_for_rf(df):
    """
    Extract features, scale them, and prepare train/test splits
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'chiefcomplaint' and 'disposition' columns
        
    Returns:
        dict: Dictionary containing processed data and preprocessing objects
    """
    # Define the expected feature names
    feature_names = [
        'entropy',
        'lexical_complexity',
        'pos_complexity',
        'medical_entities',
        'text_length',
        'word_count'
    ]
    
    # Check if the complexity features are already present in the DataFrame
    cc_feature_names = ['cc_entropy', 'cc_lexical_complexity', 'cc_pos_complexity', 
                       'cc_med_entity_count', 'cc_length', 'cc_word_count']
    
    features_df = pd.DataFrame()
    
    # If complexity features already exist in the DataFrame, use them
    if all(feature in df.columns for feature in cc_feature_names):
        print("Using existing complexity features from DataFrame...")
        # Extract and rename the features to match the expected names
        features_df['cc_entropy'] = df['cc_entropy']
        features_df['cc_lexical_complexity'] = df['cc_lexical_complexity']
        features_df['cc_pos_complexity'] = df['cc_pos_complexity']
        features_df['cc_medical_entities'] = df['cc_med_entity_count']
        features_df['cc_text_length'] = df['cc_length']
        features_df['cc_word_count'] = df['cc_word_count']
    else:
        # If features don't exist, compute them from scratch
        feature_extractor = ComplexityFeatures()

        # Generate cache key based on input texts and sample size
        cache_key = get_cache_key(df['chiefcomplaint'], len(df))
        cache_file = os.path.join('models', f'complexity_features_{cache_key}.joblib')

        # Try to load cached features
        if os.path.exists(cache_file):
            print("Loading cached complexity features...")
            features = joblib.load(cache_file)
        else:
            print("Computing complexity features...")
            features = [feature_extractor.text_features(text) for text in tqdm_func(df['chiefcomplaint'].values, desc="RF Features")]
            # Cache the features
            os.makedirs('models', exist_ok=True)
            joblib.dump(features, cache_file)
            print(f"Cached complexity features to {cache_file}")

        # Create features DataFrame and handle NaN values
        features_df = pd.DataFrame(features, columns=feature_names)
    
    # Handle any NaN values and reset index
    features_df = features_df.dropna().reset_index(drop=True)
    
    # Ensure the DataFrame has the same length as the input
    if len(features_df) != len(df):
        print(f"Warning: Features length ({len(features_df)}) doesn't match input data length ({len(df)})")
        # Adjust the input DataFrame to match
        df = df.iloc[features_df.index].reset_index(drop=True)

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    features_df = pd.DataFrame(features_scaled, columns=feature_names)

    # Encode target variable
    y = df['disposition']
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return {
        'X': features_df,
        'y': y_encoded,
        'scaler': scaler,
        'encoder': encoder,
        'feature_names': feature_names
    }

def fit_preprocessor(df: pd.DataFrame):
    """
    Fit preprocessor to the data
    
    Args:
        df (pd.DataFrame): Input DataFrame with all features
    
    Returns:
        tuple: (scaler, categorical_columns) - Fitted scaler and categorical column names
    """
    
    
    numeric_features = [
        'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp',
        'pain', 'shock_index', 'sirs', 'anchor_age', 'acuity',
        'cc_entropy', 'cc_lexical_complexity', 'cc_pos_complexity',
        'cc_med_entity_count', 'cc_length', 'cc_word_count'
    ]
    categorical_features = [
        'hr_category', 'resp_category', 'pulse_ox_category', 'sbp_category',
        'temp_category', 'dbp_category', 'pain_category', 'day_shift',
        'age_category', 'gender', 'arrival_transport'
    ]

    scaler = StandardScaler()
    scaler.fit(df[numeric_features])

    categorical_dummies = pd.get_dummies(df[categorical_features], columns=categorical_features)
    categorical_columns = categorical_dummies.columns.tolist()

    encoder = LabelEncoder()
    encoder.fit(df['disposition'] == 'ADMITTED')

    return {
        'scaler': scaler,
        'categorical_columns': categorical_columns,
        'encoder': encoder,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }


def transform_data(df: pd.DataFrame, preprocessor: dict):
    """
    Transform data using fitted preprocessor
    
    Args:
        df (pd.DataFrame): Input DataFrame with all features
        preprocessor (dict): Fitted preprocessor with scaler, encoder, and categorical column names
    
    Returns:
        tuple: (X, y, feature_names) - Transformed data and feature names
    """
    
    numeric_features = preprocessor['numeric_features']
    categorical_features = preprocessor['categorical_features']

    X_numeric_scaled = pd.DataFrame(
        preprocessor['scaler'].transform(df[numeric_features]),
        columns=numeric_features,
        index=df.index
    )

    X_categorical_encoded = pd.get_dummies(df[categorical_features], columns=categorical_features)

    # Ensure all columns from training are present in subset
    for col in preprocessor['categorical_columns']:
        if col not in X_categorical_encoded:
            X_categorical_encoded[col] = 0

    # Ensure column order consistency
    X_categorical_encoded = X_categorical_encoded[preprocessor['categorical_columns']]

    X = pd.concat([X_numeric_scaled, X_categorical_encoded], axis=1)

    y = (df['disposition'] == 'ADMITTED').astype(int)
    y_encoded = preprocessor['encoder'].transform(y)

    return {
        'X': X,
        'y': y_encoded,
        'feature_names': X.columns.tolist()
    }

def process_data_for_lr(df: pd.DataFrame, preprocessor=None, mode='train'):
    """
    Processes a DataFrame for logistic regression by scaling numeric features and encoding categorical variables.
    
    Args:
        df (pd.DataFrame): Input data containing features and target.
        preprocessor (dict, optional): Preprocessing object returned from training mode; required in inference mode.
        mode (str): Specifies operation mode, either 'train' or 'inference'.
        
    Returns:
        dict: Dictionary containing processed data:
            - 'X': (pd.DataFrame) Processed feature DataFrame.
            - 'y': (np.array) Encoded target variable array.
            - 'feature_names': (list) Names of processed feature columns.
            - 'preprocessor': (dict, only in 'train' mode) Object containing fitted scaler, encoders, and metadata for reuse.
            
    Raises:
        ValueError: If preprocessor is not provided in 'inference' mode.
    """
    if mode == 'train':
        preprocessor = fit_preprocessor(df)
        data = transform_data(df, preprocessor)
        return {
            'X': data['X'],
            'y': data['y'],
            'feature_names': data['feature_names'],
            'preprocessor': preprocessor
        }
    elif mode == 'inference':
        if preprocessor is None:
            raise ValueError("preprocessor must be provided in inference mode")
        data = transform_data(df, preprocessor)
        return {
            'X': data['X'],
            'y': data['y'],
            'feature_names': data['feature_names']
        }

def process_data_for_lr_old(df: pd.DataFrame):
    """
    Preprocesses the DataFrame by scaling numeric features and encoding categorical features.

    Args:
        df (pd.DataFrame): The input data.

    Returns:
        X (pd.DataFrame): Processed feature set.
        y (pd.Series): Encoded target variable.
        feature_names (list): List of feature names after encoding.
    """
   
    numeric_features = [
        'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp',
        'pain', 'shock_index', 'sirs', 'anchor_age', 'acuity',
    ]
    categorical_features = [
        'hr_category', 'resp_category', 'pulse_ox_category', 'sbp_category',
        'temp_category', 'dbp_category', 'pain_category', 'day_shift',
        'age_category', 'gender', 'arrival_transport'
    ]

    # Scale numeric features
    scaler = StandardScaler()
    X_numeric_scaled = pd.DataFrame(
        scaler.fit_transform(df[numeric_features]),
        columns=numeric_features,
        index=df.index
    )

    # One-hot encode categorical features
    X_categorical_encoded = pd.get_dummies(df[categorical_features], columns=categorical_features)

    # Combine numeric and categorical features
    X = pd.concat([X_numeric_scaled, X_categorical_encoded], axis=1)

    feature_names = X.columns.tolist()

    # Encode target variable
    y = (df['disposition'] == 'ADMITTED').astype(int)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return {
        'X': X,
        'y': y_encoded,
        'feature_names': feature_names,
        'scaler': scaler,
        'encoder': encoder
    }