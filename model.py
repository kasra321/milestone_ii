import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import joblib
from tqdm.auto import tqdm
import spacy
import warnings
from xgboost import XGBClassifier
from collections import Counter

class ComplexityFeatures:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        
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
        doc = self.nlp(text)
        return sum(1 for token in doc if token.pos_ in ["ADJ", "NOUN"])

    def medical_entity_count(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0
        doc = self.nlp(text)
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

class EnsemblePipeline:
    def __init__(self, model_dir="models"):
        """Initialize the pipeline with model directory path."""
        self.model_dir = model_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print("GPU available: %s" % torch.cuda.get_device_name(0))
            print("Memory available: %.2f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
        self.complexity_features = ComplexityFeatures()
        
    def load_models(self):
        """Load PCA and XGBoost models from disk."""
        try:
            # Load PCA model
            pca_path = os.path.join(self.model_dir, 'pca_transformer.joblib')
            pca = joblib.load(pca_path)
            print("Loaded PCA model from %s" % pca_path)
            
            # Load XGBoost model
            xgb_path = os.path.join(self.model_dir, 'xgboost_model.json')
            xgb = XGBClassifier()
            xgb.load_model(xgb_path)
            print("Loaded XGBoost model from %s" % xgb_path)
            
            return pca, xgb
        except Exception as e:
            raise Exception("Error loading models: %s" % str(e))
    
    def get_embeddings(self, texts, tokenizer, model, batch_size=16):
        """Generate embeddings for input texts."""
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, max_length=512, padding=True, 
                             truncation=True, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def get_features(self, texts):
        """Extract complexity features for input texts."""
        features = []
        for text in tqdm(texts, desc="Extracting features"):
            features.append(self.complexity_features.text_features(text))
        return np.array(features)
    
    def predict(self, texts, pca=None, xgb_model=None):
        """Make predictions for input texts."""
        if pca is None or xgb_model is None:
            pca, xgb_model = self.load_models()
        
        # Load tokenizer and model
        print("Loading BGE-M3 model...")
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        model = AutoModel.from_pretrained("BAAI/bge-m3")
        model.to(self.device)
        print("Model loaded successfully on %s" % self.device)
        
        # Generate embeddings
        print("Generating embeddings for %d texts..." % len(texts))
        embeddings = self.get_embeddings(texts, tokenizer, model)
        
        # Apply PCA transformation
        embeddings_pca = pca.transform(embeddings)
        
        # Make predictions
        probabilities = xgb_model.predict_proba(embeddings_pca)
        predictions = xgb_model.predict(embeddings_pca)
        
        return predictions, probabilities

def main():
    """Make predictions for sample chief complaints."""
    texts = [
        "Patient presents with severe chest pain radiating to left arm, shortness of breath",
        "No acute distress, routine follow-up for medication refill",
        "Fever and cough for 3 days, concerned about COVID"
    ]
    
    # Initialize pipeline
    pipeline = EnsemblePipeline()
    
    try:
        # Try to load and use existing models
        predictions, probabilities = pipeline.predict(texts)
        print("\nPredictions for demo texts:")
        print("Class labels: 0 = HOME/DISCHARGE, 1 = ADMITTED")
        print("------------------------------------------")
        for text, pred, prob in zip(texts, predictions, probabilities):
            print("\nText: %s" % text)
            print("Prediction: %s (class %d)" % ('ADMITTED' if pred == 1 else 'HOME', pred))
            print("Probabilities: HOME=%.3f, ADMITTED=%.3f" % (prob[0], prob[1]))
            
    except Exception as e:
        print("\nError: %s" % str(e))
        print("\nPlease run the training pipeline first to generate the models:")
        print("1. Run pipeline.py to train and save the models")
        print("2. Make sure the models are saved in the 'models' directory")
        print("3. Try running this script again")

if __name__ == "__main__":
    main()
