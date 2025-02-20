"""
import joblib
import json
from xgboost import XGBClassifier

# Load the models
pca = joblib.load('models/pca_transformer.joblib')
xgb = XGBClassifier()
xgb.load_model('models/xgboost_model.json')

# Load tokenizer config
with open('models/tokenizer_config.json', 'r') as f:
    tokenizer_config = json.load(f)
"""

import os
import json
import warnings

from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import torch

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve
    )
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from xgboost import XGBClassifier


warnings.filterwarnings('ignore')

nlp = spacy.load('en_core_web_sm')
tqdm.pandas()


# --- Complexity Features Class ---
class ComplexityFeatures:
    def calculate_entropy(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0
        words = text.lower().split()
        counts = Counter(words)
        total = sum(counts.values())
        probs = [count / total for count in counts.values()]
        return -sum(p * np.log2(p) for p in probs)

    def lexical_complexity(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0
        words = text.lower().split()
        return len(set(words)) / len(words) if words else 0

    def pos_complexity(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0
        doc = nlp(text)
        return sum(1 for token in doc if token.pos_ in ["ADJ", "NOUN"])

    def medical_entity_count(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0
        doc = nlp(text)
        return len(doc.ents)

    def text_features(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return [0]*6
        return [
            self.calculate_entropy(text),
            self.lexical_complexity(text),
            self.pos_complexity(text),
            self.medical_entity_count(text),
            len(text),
            len(text.split())
        ]

# --- Helper Functions for Logistic Regression ---
def categorize_hr(hr):
    if hr < 60:
        return "bradycardic"
    elif hr > 100:
        return "tachycardic"
    return "normal"

def categorize_resp(resp):
    if resp < 12:
        return "low"
    elif resp > 20:
        return "high"
    return "normal"

def categorize_pulseox(po):
    return "low" if po < 92 else "normal"

def categorize_bp(bp):
    if bp < 90:
        return "low"
    elif bp > 140:
        return "high"
    return "normal"

def categorize_temp(temp):
    if temp < 97.0:
        return "hypothermic"
    elif temp > 99.5:
        return "febrile"
    return "normal"

def categorize_dbp(dbp):
    if dbp < 60:
        return "low"
    elif dbp > 90:
        return "high"
    return "normal"

def categorize_pain(pain):
    if pain == 0:
        return "no pain"
    elif pain <= 3:
        return "mild"
    elif pain <= 6:
        return "moderate"
    return "severe"

def categorize_acuity(acuity):
    try:
        return f"acuity_{int(acuity)}"
    except Exception:
        return "unknown"

def categorize_age(age):
    try:
        age = float(age)
    except:
        return "unknown"
    if age < 18:
        return "child"
    elif age < 36:
        return "young_adult"
    elif age < 56:
        return "adult"
    elif age < 76:
        return "middle_aged"
    else:
        return "senior"

def is_daytime(time):
    return 7 <= time.hour < 19

def calculate_sirs(temp, hr, resp):
    count = 0
    if (temp > 38) or (temp < 36):
        count += 1
    if hr > 90:
        count += 1
    if resp > 20:
        count += 1
    return 1 if count >= 2 else 0

# --- Ensemble Pipeline Class ---
class EnsemblePipeline:
    def __init__(self, data_dir="mimic-iv-ed-2.2", exp_samples=5000):
        self.data_dir = data_dir
        self.exp_samples = exp_samples
        self.models = {}  # to store trained models by name
        self.le = None   # Label encoder
        # Set up device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            print(f"Memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load CSVs for all pipelines
    def load_data(self):
        edstays = pd.read_csv(os.path.join(self.data_dir, 'edstays.csv'))
        triage = pd.read_csv(os.path.join(self.data_dir, 'triage.csv'))
        patients = pd.read_csv(os.path.join(self.data_dir, 'patients.csv'))
        return triage, edstays, patients

    # Clean and prepare data for all pipelines
    def clean_data(self, num_samples=None):
        # Load data
        triage, edstays, patients = self.load_data()

        # Merge datasets
        df = pd.merge(triage, edstays, on=['subject_id', 'stay_id'])
        patients_subset = patients[['subject_id', 'anchor_age']]
        df = pd.merge(df, patients_subset, on='subject_id', how='left')

        # Sample if requested
        if num_samples is not None:
            df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)

        # Map dispositions
        disposition_mapping = {
            'ELOPED': 'HOME',
            'LEFT AGAINST MEDICAL ADVICE': 'HOME',
            'LEFT WITHOUT BEING SEEN': 'HOME',
            'TRANSFER': 'ADMITTED',
            'HOME': 'HOME',
            'ADMITTED': 'ADMITTED'
        }
        df = df[df['disposition'] != 'OTHER'].copy()
        df['disposition'] = df['disposition'].map(disposition_mapping)

        # Convert pain to numeric and handle missing values
        df['pain'] = pd.to_numeric(df['pain'], errors='coerce')
        df = df.dropna(subset=['heartrate', 'resprate', 'o2sat', 'sbp',
                              'temperature', 'dbp', 'pain', 'acuity', 
                              'anchor_age', 'disposition', 'chiefcomplaint']).reset_index(drop=True)

        # Create derived features
        df['shock_index'] = df['heartrate'] / df['sbp']
        df['intime'] = pd.to_datetime(df['intime'])
        df['day_shift'] = df['intime'].apply(is_daytime)
        df['sirs'] = df.apply(lambda row: calculate_sirs(row['temperature'], 
                                                        row['heartrate'], 
                                                        row['resprate']), axis=1)

        # Create categorical features
        df['hr_category'] = df['heartrate'].apply(categorize_hr)
        df['resp_category'] = df['resprate'].apply(categorize_resp)
        df['pulse_ox_category'] = df['o2sat'].apply(categorize_pulseox)
        df['sbp_category'] = df['sbp'].apply(categorize_bp)
        df['temp_category'] = df['temperature'].apply(categorize_temp)
        df['dbp_category'] = df['dbp'].apply(categorize_dbp)
        df['pain_category'] = df['pain'].apply(categorize_pain)
        df['acuity_category'] = df['acuity'].apply(categorize_acuity)
        df['age_category'] = df['anchor_age'].apply(categorize_age)

        # Create label encoding for disposition
        self.le = LabelEncoder()
        df['disposition_enc'] = self.le.fit_transform(df['disposition'])

        return df

    # Extract and scale complexity features
    def extract_complexity_features(self, df):
        extractor = ComplexityFeatures()
        print("Extracting complexity features...")
        features = df['chiefcomplaint'].progress_apply(extractor.text_features).tolist()
        feature_names = ['entropy', 'lexical_complexity', 'pos_complexity',
                         'medical_entities', 'text_length', 'word_count']
        features_df = pd.DataFrame(features, columns=feature_names)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        features_df = pd.DataFrame(features_scaled, columns=feature_names)
        return features_df, df['disposition_enc']

    # Extract text data for transformer embeddings
    def extract_embedding_data(self, df):
        texts = df['chiefcomplaint'].tolist()
        return texts, df['disposition_enc']

    # Train Random Forest with undersampling on complexity features
    def train_random_forest(self, X_train, y_train, X_test, y_test, selected_features=['entropy', 'pos_complexity', 'medical_entities', 'text_length']):
        X_train_sel = X_train[selected_features]
        X_test_sel = X_test[selected_features]
        # Check class distribution before undersampling
        class_counts = np.bincount(y_train)
        min_class_size = min(class_counts)
        
        if min_class_size > 0:
            try:
                # Create sampling strategy dictionary
                sampling_strategy = {
                    0: min_class_size,  # Keep all samples of minority class
                    1: min_class_size   # Undersample majority class to match minority
                }
                
                rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
                X_train_res, y_train_res = rus.fit_resample(X_train_sel, y_train)
                print(f"Undersampling complete. New class distribution: {np.bincount(y_train_res)}")
            except Exception as e:
                print(f"Warning: Undersampling failed - {str(e)}. Using original data.")
                print(f"Original class distribution: {class_counts}")
                X_train_res, y_train_res = X_train_sel, y_train
        else:
            print("Warning: Empty class detected. Using original data.")
            print(f"Class distribution: {class_counts}")
            X_train_res, y_train_res = X_train_sel, y_train
            
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_res, y_train_res)
        y_pred = rf.predict(X_test_sel)
        y_proba = rf.predict_proba(X_test_sel)
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        target_names = [str(c) for c in self.le.classes_]
        metrics = {
            'auc_roc': auc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'report': classification_report(y_test, y_pred, target_names=target_names)
        }
        self.models['random_forest'] = {
            'model': rf,
            'features': selected_features,
            'X_test': X_test,
            'y_test': y_test,
            'y_proba': y_proba
        }
        print("Random Forest Performance:")
        print("AUC-ROC:", auc)
        print("F1 Macro:", f1_macro)
        print("F1 Weighted:", f1_weighted)
        print("\nClassification Report:\n", metrics['report'])
        return metrics

    # Compute transformer embeddings
    def get_embeddings(self, texts, tokenizer, model, pooling_method='cls', max_length=512, 
                      device='cpu', batch_size=32, cache_key=None, num_samples=None, force_regenerate=False):
        """Get embeddings for texts, with caching based on sample size."""
        if cache_key:
            # Include sample size in cache key if specified
            sample_suffix = f"_{num_samples}" if num_samples else ""
            cache_dir = os.path.join('models', 'embeddings')
            cache_file = os.path.join(cache_dir, f'{cache_key}{sample_suffix}_embeddings.npy')
            
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
            # Check if cache exists and matches current sample size
            if os.path.exists(cache_file) and not force_regenerate:
                try:
                    cached_embeddings = np.load(cache_file)
                    if len(cached_embeddings) == len(texts):
                        print(f"Loading cached embeddings from {cache_file}")
                        return cached_embeddings
                    else:
                        print(f"Cached embeddings size mismatch. Regenerating...")
                except Exception as e:
                    print(f"Error loading cache: {e}. Regenerating embeddings...")
            
        # Generate embeddings
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, max_length=max_length, padding=True, 
                             truncation=True, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                if pooling_method == 'cls':
                    batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
                else:  # mean pooling
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = batch_embeddings.cpu().numpy()
                
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        # Save to cache if cache_key provided
        if cache_key:
            np.save(cache_file, embeddings)
            print(f"Saved embeddings to {cache_file}")
        
        return embeddings

    # Train XGBoost with undersampling on transformer embeddings
    def train_xgboost_embedding(self, texts_train, y_train, texts_test, y_test, 
                              pooling_method='cls', pca_components=20, test_size=0.2, 
                              random_state=42):
        """Train XGBoost on embeddings with proper cache handling."""
        print("\nLoading BGE-M3 model...")
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        model = AutoModel.from_pretrained("BAAI/bge-m3")
        model.to(self.device)
        print("Model loaded successfully on", self.device)

        # Get number of samples for cache key
        num_samples = len(texts_train) + len(texts_test)
        
        print(f"\nGenerating embeddings for {len(texts_train)} training texts...")
        embeddings_train = self.get_embeddings(texts_train, tokenizer, model, 
                                             pooling_method=pooling_method,
                                             device=self.device, 
                                             batch_size=16,
                                             cache_key='train',
                                             num_samples=num_samples)
        
        print(f"\nGenerating embeddings for {len(texts_test)} test texts...")
        embeddings_test = self.get_embeddings(texts_test, tokenizer, model, 
                                            pooling_method=pooling_method,
                                            device=self.device, 
                                            batch_size=16,
                                            cache_key='test',
                                            num_samples=num_samples)
        
        print(f"\nReducing dimensionality with PCA (n_components={pca_components})...")
        pbar = tqdm(total=4, desc="Training XGBoost", unit="step")
        
        pca = PCA(n_components=pca_components, random_state=random_state)
        X_train_reduced = pca.fit_transform(embeddings_train)
        X_test_reduced = pca.transform(embeddings_test)
        pbar.update(1)
        pbar.set_postfix({"step": "PCA complete"})
        
        # Check class distribution and set up undersampling strategy
        class_counts = np.bincount(y_train)
        min_class_size = min(class_counts)
        maj_class_size = max(class_counts)
        
        if min_class_size > 0:
            try:
                # Create sampling strategy dictionary
                sampling_strategy = {
                    0: min_class_size,  # Keep all samples of minority class
                    1: min_class_size   # Undersample majority class to match minority
                }
                
                rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
                X_train_res, y_train_res = rus.fit_resample(X_train_reduced, y_train)
                print(f"Undersampling complete. New class distribution: {np.bincount(y_train_res)}")
            except Exception as e:
                print(f"Warning: Undersampling failed - {str(e)}. Using original data.")
                print(f"Original class distribution: {class_counts}")
                X_train_res, y_train_res = X_train_reduced, y_train
        else:
            print("Warning: Empty class detected. Using original data.")
            print(f"Class distribution: {class_counts}")
            X_train_res, y_train_res = X_train_reduced, y_train
            
        pbar.update(1)
        pbar.set_postfix({"step": "Undersampling complete"})
        
        # Configure XGBoost for GPU if available
        tree_method = 'gpu_hist' if self.device == 'cuda' else 'auto'
        xgb = XGBClassifier(
            random_state=random_state,
            eval_metric='logloss',
            tree_method=tree_method,  # Use GPU acceleration if available
            predictor='gpu_predictor' if self.device == 'cuda' else 'auto',  # Use GPU for prediction if available
            n_jobs=-1  # Use all available CPU cores
        )
        
        print(f"Training XGBoost using {tree_method} method...")
        xgb.fit(X_train_res, y_train_res)
        pbar.update(1)
        pbar.set_postfix({"step": "XGBoost training complete"})
        
        # Save XGBoost model
        model_path = os.path.join('models', 'xgboost_model.json')
        xgb.save_model(model_path)
        print(f"Saved XGBoost model to {model_path}")
        
        # Save tokenizer configuration
        tokenizer_path = os.path.join('models', 'tokenizer_config.json')
        tokenizer_config = {
            'model_name': 'BAAI/bge-m3',
            'max_length': 512,
            'pooling_method': pooling_method
        }
        with open(tokenizer_path, 'w') as f:
            json.dump(tokenizer_config, f)
        print(f"Saved tokenizer configuration to {tokenizer_path}")
        
        y_pred = xgb.predict(X_test_reduced)
        y_proba = xgb.predict_proba(X_test_reduced)
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        f1 = f1_score(y_test, y_pred, average='macro')
        acc = accuracy_score(y_test, y_pred)
        target_names = [str(c) for c in self.le.classes_]
        metrics = {
            'auc_roc': auc,
            'f1': f1,
            'accuracy': acc,
            'report': classification_report(y_test, y_pred, target_names=target_names)
        }
        print("\nXGBoost Performance:")
        print("AUC-ROC:", auc)
        print("F1 Score:", f1)
        print("Accuracy:", acc)
        print("\nClassification Report:\n", metrics['report'])
        if len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            plt.figure(figsize=(8,6))
            plt.plot(fpr, tpr, label=f'XGBoost (AUC={auc:.2f})')
            plt.plot([0,1], [0,1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve - XGBoost")
            plt.legend()
            plt.grid(True)
            plt.show()
        self.models['xgboost'] = {
            'model': xgb,
            'pca': pca,
            'X_test': X_test_reduced,
            'y_test': y_test,
            'y_proba': y_proba
        }
        self.save_models(pca, xgb)
        return metrics

    # Train Logistic Regression on vitals and extra features using undersampling and oversampling
    def train_logistic_regression_vitals(self, df_train, df_test, total_samples=2000, test_size=0.2, random_state=42):
        # Load datasets: edstays, triage, and patients
        edstays_df = pd.read_csv(os.path.join(self.data_dir, 'edstays.csv'))
        triage_df = pd.read_csv(os.path.join(self.data_dir, 'triage.csv'))
        patients_df = pd.read_csv(os.path.join(self.data_dir, 'patients.csv'))

        # Merge triage and edstays, then merge patients (using anchor_age)
        df = pd.merge(triage_df, edstays_df, on=['subject_id', 'stay_id'])
        patients_subset = patients_df[['subject_id', 'anchor_age']]
        df = pd.merge(df, patients_subset, on='subject_id', how='left')

        n_samples = min(total_samples, len(df))
        df = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)

        disposition_mapping = {
            'ELOPED': 'HOME',
            'LEFT AGAINST MEDICAL ADVICE': 'HOME',
            'LEFT WITHOUT BEING SEEN': 'HOME',
            'TRANSFER': 'ADMITTED',
            'HOME': 'HOME',
            'ADMITTED': 'ADMITTED'
        }
        df = df[df['disposition'] != 'OTHER'].copy()
        df['disposition'] = df['disposition'].map(disposition_mapping)

        df['pain'] = pd.to_numeric(df['pain'], errors='coerce')
        df = df.dropna(subset=['heartrate', 'resprate', 'o2sat', 'sbp',
                               'temperature', 'dbp', 'pain', 'acuity', 'anchor_age', 'disposition']).reset_index(drop=True)
        # Feature Engineering
        df['shock_index'] = df['heartrate'] / df['sbp']
        df['intime'] = pd.to_datetime(df['intime'])
        df['day_shift'] = df['intime'].apply(is_daytime)
        df['sirs'] = df.apply(lambda row: calculate_sirs(row['temperature'], row['heartrate'], row['resprate']), axis=1)

        df['hr_category'] = df['heartrate'].apply(categorize_hr)
        df['resp_category'] = df['resprate'].apply(categorize_resp)
        df['pulse_ox_category'] = df['o2sat'].apply(categorize_pulseox)
        df['sbp_category'] = df['sbp'].apply(categorize_bp)
        df['temp_category'] = df['temperature'].apply(categorize_temp)
        df['dbp_category'] = df['dbp'].apply(categorize_dbp)
        df['pain_category'] = df['pain'].apply(categorize_pain)
        df['acuity_category'] = df['acuity'].apply(categorize_acuity)
        df['age_category'] = df['anchor_age'].apply(categorize_age)

        extra_features = ['race', 'gender', 'arrival_transport']
        categorical_features = ["hr_category", "resp_category", "pulse_ox_category", "sbp_category",
                                "temp_category", "dbp_category", "pain_category", "acuity_category", "day_shift", "age_category"]
        for feat in extra_features:
            if feat in df.columns:
                categorical_features.append(feat)
                numeric_features = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "shock_index", "sirs", "anchor_age"]

        X_train = df_train[numeric_features + categorical_features]
        y_train = df_train["disposition"]
        X_test = df_test[numeric_features + categorical_features]
        y_test = df_test["disposition"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
            ]
        )
        
        logreg_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced"))
        ])

        # Undersampling
        rus = RandomUnderSampler(random_state=random_state)
        X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
        logreg_pipeline.fit(X_train_under, y_train_under)
        y_pred_under = logreg_pipeline.predict(X_test)
        y_prob_under = logreg_pipeline.predict_proba(X_test)[:, 1]
        le_target = LabelEncoder()
        y_test_enc = le_target.fit_transform(y_test)
        y_pred_enc_under = le_target.transform(y_pred_under)
        report_under = classification_report(y_test, y_pred_under, target_names=logreg_pipeline.classes_)
        auc_roc_under = roc_auc_score(y_test_enc, y_prob_under)
        f1_under = f1_score(y_test_enc, y_pred_enc_under, pos_label=1)
        acc_under = accuracy_score(y_test_enc, y_pred_enc_under)
        recall_under = recall_score(y_test_enc, y_pred_enc_under, pos_label=1)
        metrics_under = {
            "classification_report": report_under,
            "auc_roc": auc_roc_under,
            "f1": f1_under,
            "accuracy": acc_under,
            "recall": recall_under
        }
        print("Logistic Regression (Undersampling) Classification Report:")
        print(report_under)
        print(f"AUC-ROC: {auc_roc_under:.3f}")
        print(f"F1 Score (ADMITTED): {f1_under:.3f}")
        print(f"Accuracy: {acc_under:.3f}")
        print(f"Recall (ADMITTED): {recall_under:.3f}\n")

        # Oversampling
        ros = RandomOverSampler(random_state=random_state)
        X_train_over, y_train_over = ros.fit_resample(X_train, y_train)
        logreg_pipeline.fit(X_train_over, y_train_over)
        y_pred_over = logreg_pipeline.predict(X_test)
        y_prob_over = logreg_pipeline.predict_proba(X_test)[:, 1]
        y_pred_enc_over = le_target.transform(y_pred_over)
        report_over = classification_report(y_test, y_pred_over, target_names=logreg_pipeline.classes_)
        auc_roc_over = roc_auc_score(y_test_enc, y_prob_over)
        f1_over = f1_score(y_test_enc, y_pred_enc_over, pos_label=1)
        acc_over = accuracy_score(y_test_enc, y_pred_enc_over)
        recall_over = recall_score(y_test_enc, y_pred_enc_over, pos_label=1)
        metrics_over = {
            "classification_report": report_over,
            "auc_roc": auc_roc_over,
            "f1": f1_over,
            "accuracy": acc_over,
            "recall": recall_over
        }
        print("Logistic Regression (Oversampling) Classification Report:")
        print(report_over)
        print(f"AUC-ROC: {auc_roc_over:.3f}")
        print(f"F1 Score (ADMITTED): {f1_over:.3f}")
        print(f"Accuracy: {acc_over:.3f}")
        print(f"Recall (ADMITTED): {recall_over:.3f}")

        self.models['logreg_under'] = {
            'model': logreg_pipeline,
            'X_test': X_test,
            'y_test': y_test,
            'y_proba': y_prob_under
        }
        self.models['logreg_over'] = {
            'model': logreg_pipeline,
            'X_test': X_test,
            'y_test': y_test,
            'y_proba': y_prob_over
        }
        return {"undersample": metrics_under, "oversample": metrics_over}

    # Ensemble prediction: average probabilities from available models
    def ensemble_predict(self):
        if not self.models:
            raise ValueError("No models available for ensemble.")
        probs = []
        y_tests = []
        for key, mod in self.models.items():
            proba = mod['y_proba']
            # Handle 1D output (binary probability) or 2D output
            if proba.ndim == 1:
                probs.append(proba)
            elif proba.shape[1] == 2:
                probs.append(proba[:, 1])
            else:
                probs.append(proba[:, 0])
            y_tests.append(mod['y_test'])
        probs = np.column_stack(probs)
        ensemble_prob = np.mean(probs, axis=1)
        y_true = y_tests[0]
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)
        if len(np.unique(y_true)) == 2:
            auc = roc_auc_score(y_true, ensemble_prob)
        else:
            auc = roc_auc_score(y_true, np.column_stack([1-ensemble_prob, ensemble_prob]), multi_class='ovr')
        f1 = f1_score(y_true, ensemble_pred, average='macro')
        target_names = [str(c) for c in self.le.classes_]
        report = classification_report(y_true, ensemble_pred, target_names=target_names)
        print("\nEnsemble Performance:")
        print("AUC-ROC: %.4f" % auc)
        print("F1 Macro: %.4f" % f1)
        print("\nClassification Report:\n%s" % report)
        return {
            'auc_roc': auc,
            'f1_macro': f1,
            'report': report
        }

    # Run full pipeline including all models
    def run_pipeline(self, exp_run=True, pooling_method='cls', pca_components=20):
        # Load and clean data
        num_samples = self.exp_samples if exp_run else None
        df = self.clean_data(num_samples=num_samples)

        # Create a single train/test split to be used by all models
        train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df['disposition_enc'])
        df_train = df.loc[train_idx].reset_index(drop=True)
        df_test = df.loc[test_idx].reset_index(drop=True)

        # Initialize progress bar for model training
        models_to_train = ['Random Forest', 'XGBoost', 'Logistic Regression']
        model_pbar = tqdm(models_to_train, desc="Training models")

        # Train Random Forest
        model_pbar.set_description("Training Random Forest on complexity features")
        X_train_complex, y_train = self.extract_complexity_features(df_train)
        X_test_complex, y_test = self.extract_complexity_features(df_test)
        rf_metrics = self.train_random_forest(X_train_complex, y_train, X_test_complex, y_test)
        model_pbar.update(1)

        # Train XGBoost
        model_pbar.set_description("Training XGBoost on transformer embeddings")
        texts_train, y_train_emb = self.extract_embedding_data(df_train)
        texts_test, y_test_emb = self.extract_embedding_data(df_test)
        xgb_metrics = self.train_xgboost_embedding(texts_train, y_train_emb, texts_test, y_test_emb,
                                                 pooling_method=pooling_method,
                                                 pca_components=pca_components)
        model_pbar.update(1)

        # Train Logistic Regression using the same test set
        model_pbar.set_description("Training Logistic Regression on vitals and extra features")
        logreg_metrics = self.train_logistic_regression_vitals(df_train, df_test)
        model_pbar.update(1)
        model_pbar.close()

        # Ensemble predictions
        ensemble_metrics = self.ensemble_predict()
        return {
            'random_forest': rf_metrics,
            'xgboost': xgb_metrics,
            'logreg': logreg_metrics,
            'ensemble': ensemble_metrics
        }
        
    def save_models(self, pca, xgb_model, model_dir='models'):
        """Save PCA and XGBoost models."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save PCA model
        pca_path = os.path.join(model_dir, 'pca_transformer.joblib')
        joblib.dump(pca, pca_path)
        print(f"Saved PCA model to {pca_path}")
        
        # Save XGBoost model
        xgb_path = os.path.join(model_dir, 'xgboost_model.json')
        xgb_model.save_model(xgb_path)
        print(f"Saved XGBoost model to {xgb_path}")
        
    def load_models(self, model_dir='models'):
        """Load saved PCA and XGBoost models."""
        pca_path = os.path.join(model_dir, 'pca_transformer.joblib')
        xgb_path = os.path.join(model_dir, 'xgboost_model.json')
        
        if not os.path.exists(pca_path) or not os.path.exists(xgb_path):
            raise FileNotFoundError("Model files not found. Please train the models first.")
        
        # Load PCA model
        pca = joblib.load(pca_path)
        print(f"Loaded PCA model from {pca_path}")
        
        # Load XGBoost model
        xgb_model = XGBClassifier()
        xgb_model.load_model(xgb_path)
        print(f"Loaded XGBoost model from {xgb_path}")
        
        return pca, xgb_model
    
    def predict(self, texts, pca=None, xgb_model=None):
        """Make predictions using saved models."""
        if pca is None or xgb_model is None:
            pca, xgb_model = self.load_models()
        
        # Load tokenizer and model
        print("Loading BGE-M3 model...")
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        model = AutoModel.from_pretrained("BAAI/bge-m3")
        model.to(self.device)
        print("Model loaded successfully on", self.device)
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.get_embeddings(texts, tokenizer, model,
                                       device=self.device,
                                       batch_size=16)
        
        # Apply PCA transformation
        print("Applying PCA transformation...")
        X_transformed = pca.transform(embeddings)
        
        # Make predictions
        print("Making predictions...")
        predictions = xgb_model.predict(X_transformed)
        probabilities = xgb_model.predict_proba(X_transformed)
        
        return predictions, probabilities

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run the pipeline with specified sample size')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-n', '--num_samples', type=int, help='Number of samples to use for experiment')
    group.add_argument('-f', '--full', action='store_true', help='Run on full dataset')
    
    args = parser.parse_args()
    
    # Set up GPU memory growth to avoid taking all GPU memory at once
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
    
    pipeline = EnsemblePipeline()
    
    if args.full:
        print("\nRunning pipeline on full dataset...")
        results = pipeline.run_pipeline(exp_run=False, pooling_method='cls', pca_components=20)
    else:
        print(f"\nRunning experimental run using first {args.num_samples} samples...")
        pipeline.exp_samples = args.num_samples
        results = pipeline.run_pipeline(exp_run=True, pooling_method='cls', pca_components=20)
