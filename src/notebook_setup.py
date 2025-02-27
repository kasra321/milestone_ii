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

def setup():
    """Main setup function that initializes everything"""
    _setup_plotting()
    _setup_pandas()
    
    # Import your custom modules
    from src.mimicdf import MIMICDF
    from src.preprocessing.data_preprocessor import DataCleaner, FeatureEngineer
    
    # Initialize MIMIC database connection
    mimicdf = MIMICDF.create_connection()
    return mimicdf

# Only expose what's needed in notebooks
__all__ = [
    # Common libraries
    'pd', 'np', 'plt', 'sns',
    # Setup function
    'setup'
]