import os
import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from google.oauth2 import service_account

def _setup_plotting():
    """Configure default plotting settings"""
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    warnings.filterwarnings('ignore')
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def _setup_pandas():
    """Configure pandas display settings"""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

def setup(source='demo'):
    """
    Main setup function that initializes everything
    
    Args:
        source (str): Data source - either 'demo' for local demo data or 'gcp' for BigQuery data
        
    Returns:
        MIMICDF: A configured MIMICDF instance
    """
    _setup_plotting()
    _setup_pandas()
    
    # Import your custom modules
    from src.mimicdf import MIMICDF
    from src.preprocessing.data_preprocessor import DataCleaner, FeatureEngineer
    
    # Initialize MIMIC database connection based on source parameter
    if source == 'gcp':
        mimicdf = MIMICDF.create_connection()
    elif source == 'demo':
        mimicdf = MIMICDF.create_demo()
    else:
        print(f"Invalid source '{source}'. Defaulting to demo data.")
        mimicdf = MIMICDF.create_demo()
        
    return mimicdf

# Only expose what's needed in notebooks
__all__ = [
    # Common libraries
    'pd', 'np', 'plt', 'sns',
    # Setup function
    'setup'
]