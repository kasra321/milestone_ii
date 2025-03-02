import os
import pickle
import numpy as np
import hashlib
import joblib

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Set random seed
RANDOM_STATE = 42
SAMPLE_SIZE = 1000
DEFAULT_PCA_COMPONENTS = 20
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# disable "This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable."
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class EmbeddingExtractor:
    """
    Extracts text embeddings from a transformer model.

    Attributes:
        tokenizer (transformers.AutoTokenizer): Tokenizer for the transformer model.
        model (transformers.AutoModel): Transformer model for extracting embeddings.
        device (str): Device to run the model on. Defaults to CUDA if available.
        embeddings (array-like): Last computed embeddings.
        cache_dir (str): Directory to store cached embeddings. Defaults to 'embeddings'.
    """

    def __init__(self, model_name=EMBEDDING_MODEL_NAME, device=None, cache_dir='embeddings'):
        """
        Initialize the EmbeddingExtractor.
        
        Args:
            model_name (str): Name of the transformer model to use
            device (str, optional): Device to run model on. Defaults to CUDA if available
            cache_dir (str, optional): Directory to store cached embeddings
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.embeddings = None
        self.cache_dir = cache_dir
        # Create cache directory and any necessary parent directories
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Using device: {self.device}")
        print(f"Cache directory: {os.path.abspath(self.cache_dir)}")

    def _get_cache_key(self, texts, sample_size=None):
        """
        Generate a unique cache key based on the input texts and sample size.
        
        Args:
            texts (list): List of texts to generate cache key from
            sample_size (int, optional): Size of the sample, if using subset
            
        Returns:
            str: MD5 hash of the sorted texts and sample size
        """
        sorted_texts = sorted([str(t) for t in texts])
        content = ''.join(sorted_texts) + str(sample_size)
        return hashlib.md5(content.encode()).hexdigest()

    def get_or_create_embeddings(self, df, column='chiefcomplaint'):
        """
        Try to load existing embeddings, create new ones if not found or if shapes don't match.
        
        Args:
            df (pd.DataFrame): DataFrame containing text column
            column (str): Name of the column containing text data. Defaults to 'chiefcomplaint'
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        texts = df[column].tolist()
        cache_key = self._get_cache_key(texts, len(df))
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, f'embeddings_{cache_key}.joblib')
        
        try:
            if os.path.exists(cache_file):
                print(f"Loading cached embeddings from {cache_file}...")
                embeddings = joblib.load(cache_file)
                if embeddings.shape[0] != len(df):
                    raise ValueError("Cached embeddings shape doesn't match data")
                print(f"Loaded existing embeddings, shape: {embeddings.shape}")
            else:
                raise FileNotFoundError("Cache file not found")
                
        except (FileNotFoundError, ValueError) as e:
            print(f"Creating new embeddings... Reason: {str(e)}")
            embeddings = self.transform(texts)
            print(f"Created embeddings, shape: {embeddings.shape}")
            
            # Double-check directory exists before saving
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            try:
                joblib.dump(embeddings, cache_file)
                print(f"Cached embeddings saved to {cache_file}")
            except Exception as save_error:
                print(f"Warning: Failed to save embeddings cache: {str(save_error)}")
        
        return embeddings

    def get_embedding(self, text):
        """
        Get embedding for a single text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding

    def transform(self, texts):
        """
        Transform a list of texts into embeddings.
        
        Args:
            texts (list): List of texts to transform
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        embeddings = []
        for text in tqdm(texts, desc="Extracting embeddings"):
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return np.vstack(embeddings)

    def save(self, embeddings):
        """
        Save embeddings to a pickle file.
        
        Args:
            embeddings (numpy.ndarray): Embeddings to save
            
        Returns:
            str: Status message
        """
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embeddings must be a numpy array.")
    
        n = embeddings.shape[0]
        os.makedirs(self.cache_dir, exist_ok=True)
        file_path = os.path.join(self.cache_dir, f"embeddings-{n}.pkl")
        
        # Check if file already exists
        if os.path.exists(file_path):
            return "Already saved, loading existing"
        
        # Save if file doesn't exist
        try:
            with open(file_path, "wb") as f:
                pickle.dump(embeddings, f)
            return f"Saved new embeddings to: {file_path}"
        except Exception as e:
            return f"Error saving embeddings: {str(e)}"
    
    def load(self, n=1000):
        """
        Load embeddings from a pickle file.
        
        Args:
            n (int): Number of embeddings to load
            
        Returns:
            numpy.ndarray: Loaded embeddings
        """
        file_path = os.path.join(self.cache_dir, f"embeddings-{n}.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at {file_path}")
        with open(file_path, "rb") as f:
            embeddings = pickle.load(f)
        return embeddings
