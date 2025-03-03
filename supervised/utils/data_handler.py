import os
import logging
import numpy as np
import pandas as pd
import tqdm
from sklearn.preprocessing import LabelEncoder

from .helper_functions import (
    categorize_hr, categorize_resp, categorize_pulseox, categorize_bp,
    categorize_temp, categorize_dbp, categorize_pain, categorize_acuity,
    categorize_age, is_daytime, calculate_sirs
)
from .pain_utils import clean_pain_value
from .complexity import ComplexityFeatures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataHandler():

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.label_encoder = LabelEncoder()

        # Numeric features
        self.lr_numeric_features = [
            "temperature", "heartrate", "resprate", "o2sat", "sbp", 
            "dbp", "pain", "shock_index", "sirs", "anchor_age"
        ]

        # Categorical features
        self.lr_categorical_features = [
            "hr_category", "resp_category", "pulse_ox_category", "sbp_category",
            "temp_category", "dbp_category", "pain_category", "acuity_category", 
            "day_shift", "age_category", 'race', 'gender', 'arrival_transport'
        ]

        self.lr_all_features = self.lr_categorical_features + self.lr_numeric_features


    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the data from directory"""
        logger.info('Loading data from directory %s', self.data_dir)
        edstays = pd.read_csv(os.path.join(self.data_dir, 'edstays.csv'))
        triage = pd.read_csv(os.path.join(self.data_dir, 'triage.csv'))
        patients = pd.read_csv(os.path.join(self.data_dir, 'patients.csv'))
        logger.info('Successfully loaded edstays (%d rows), triage (%d rows), and patients (%d rows)', 
                   len(edstays), len(triage), len(patients))
        return triage, edstays, patients

    def merge_data(self, triage: pd.DataFrame, edstays: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
        """Merge the data tables """
        logger.info('Merging triage and edstays data')
        df = pd.merge(triage, edstays, on=['subject_id', 'stay_id'])
        logger.info('Merging patient data for anchor_age')
        patients_subset = patients[['subject_id', 'anchor_age']]
        merged_df = pd.merge(df, patients_subset, on='subject_id', how='left')
        logger.info('Data merge complete. Final shape: %s', merged_df.shape)
        return merged_df

    def clean_data(self, df) -> pd.DataFrame:
        """Clean data by converting numeric fields and handling missing values"""
        logger.info('Starting data cleaning process')
        
        # First map dispositions since we need this before handling missing values
        df = self.map_dispositions(df)
        
        # Handle missing values and imputation
        df = self.handle_missing_values(df)
        logger.info('Data cleaning complete')
        
        return df

    def map_dispositions(self, df):
        """Map dispositions"""
        logger.info('Mapping disposition values')

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
        logger.info('Disposition mapping complete. Unique values: %s', df['disposition'].unique())

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values through appropriate imputation strategies"""
        logger.info('Analyzing missing values before imputation')
        
        # Create a copy to avoid chained assignment warnings
        df = df.copy()
        
        # Handle acuity first - convert to numeric and replace NaN with -1
        logger.info('Converting acuity to numeric and handling missing values')
        df['acuity'] = pd.to_numeric(df['acuity'], errors='coerce')
        missing_acuity = df['acuity'].isnull().sum()
        df['acuity'] = df['acuity'].fillna(-1)
        logger.info('Replaced %d missing acuity values with -1', missing_acuity)
        
        # These columns we'll impute with median/mode
        vital_signs = ['heartrate', 'resprate', 'o2sat', 'sbp',
                      'temperature', 'dbp', 'pain', 'anchor_age']
        
        # Define valid ranges for vital signs
        vital_ranges = {
            'heartrate': (0, 300),    # beats per minute
            'resprate': (0, 99),      # breaths per minute
            'o2sat': (0, 100),        # percentage
            'sbp': (0, 400),          # mmHg
            'temperature': (30, 115),  # Fahrenheit
            'dbp': (0, 350),          # mmHg
            'pain': (0, 20),           # pain scale
            'anchor_age': (0, 300)     # years
        }
        
        # These columns we'll still require (can't meaningfully impute)
        required_columns = ['anchor_age', 'disposition', 'chiefcomplaint']
        
        # Log missing value analysis
        all_columns = vital_signs + required_columns
        for col in all_columns:
            null_count = df[col].isnull().sum()
            null_percent = (null_count / len(df)) * 100
            logger.info('Column %s has %d missing values (%.2f%%)', col, null_count, null_percent)
        
        # Analyze and drop rows with missing required values one by one
        logger.info('Analyzing required columns for missing values:')
        initial_rows = len(df)
        rows_dropped_by_col = {}
        
        for col in required_columns:
            rows_before = len(df)
            df = df.dropna(subset=[col]).reset_index(drop=True)
            rows_dropped = rows_before - len(df)
            rows_dropped_by_col[col] = rows_dropped
            if rows_dropped > 0:
                logger.info('Dropped %d rows (%.2f%%) due to missing %s', 
                          rows_dropped, 
                          (rows_dropped / rows_before) * 100,
                          col)
        
        total_dropped = initial_rows - len(df)
        logger.info('Summary of dropped rows:')
        for col, dropped in rows_dropped_by_col.items():
            logger.info('  - %s: %d rows (%.2f%% of total drops)', 
                       col, dropped, 
                       (dropped / total_dropped * 100) if total_dropped > 0 else 0)
        logger.info('Total rows dropped: %d (%.2f%% of original data)', 
                   total_dropped, 
                   (total_dropped / initial_rows * 100))
        
        # Special handling for pain values
        logger.info('Cleaning pain values')
        initial_unclear = (df['pain'] == -1).sum() if 'pain' in df else 0
        df['pain'] = df['pain'].apply(clean_pain_value)
        final_unclear = (df['pain'] == -1).sum()
        logger.info('Pain value cleaning: %d unclear values converted to NaN',
                   initial_unclear - final_unclear)
        
        # Clean vital signs before imputation
        logger.info('Cleaning vital signs before imputation')
        for col, (min_val, max_val) in vital_ranges.items():
            if col != 'pain':  # Skip pain as it's already cleaned
                # Convert to numeric if not already
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Count out of range values
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if out_of_range > 0:
                    logger.info('Found %d values outside valid range for %s [%d-%d]', 
                              out_of_range, col, min_val, max_val)
                    # Set out of range values to NaN
                    df.loc[(df[col] < min_val) | (df[col] > max_val), col] = None
        
        # Now handle vital signs through imputation
        logger.info('Imputing missing vital signs')
        for col in vital_signs:
            if col == 'pain':  # Pain is discrete 0-10, but may be documented as higher
                # Convert any remaining -1 values to NaN before imputation
                df.loc[df['pain'] == -1, 'pain'] = np.nan
                df.loc[df['pain'].isnull(), 'pain'] = 4.4
                logger.info('Imputed pain with mean value: %s', 4.4)
            else:  # Other vitals are continuous
                median_value = df[col].median()
                df.loc[df[col].isnull(), col] = median_value
                logger.info('Imputed %s with median value: %.2f', col, median_value)
        
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features needed for the model"""
        logger.info('Creating features')
        
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Create categorical features
        df['hr_category'] = df['heartrate'].apply(categorize_hr)
        df['resp_category'] = df['resprate'].apply(categorize_resp)
        df['pulse_ox_category'] = df['o2sat'].apply(categorize_pulseox)
        df['sbp_category'] = df['sbp'].apply(categorize_bp)
        df['temp_category'] = df['temperature'].apply(categorize_temp)
        df['dbp_category'] = df['dbp'].apply(categorize_dbp)
        df['pain_category'] = df['pain'].apply(categorize_pain)
        df['acuity_category'] = df['acuity'].apply(categorize_acuity) #
        df['age_category'] = df['anchor_age'].apply(categorize_age)
        df['day_shift'] = pd.to_datetime(df['intime']).apply(is_daytime)
        df['shock_index'] = df['heartrate'] / df['sbp']
        df['sirs'] = df.apply(lambda row: calculate_sirs(row['temperature'], 
                                                        row['heartrate'], 
                                                        row['resprate']), axis=1)

        # Add complexity features only for chiefcomplaint column
        if 'chiefcomplaint' in df.columns:
            logger.info('Calculating complexity features for chief complaints')
            complexity = ComplexityFeatures()
            
            # Create a copy of the chiefcomplaint series and handle non-string values
            df = df.dropna(subset=['chiefcomplaint']).reset_index(drop=True)
            chief_complaints = df['chiefcomplaint'].copy()
            
            # Calculate simple features using vectorized operations (faster)
            df['cc_length'] = chief_complaints.str.len()
            df['cc_word_count'] = chief_complaints.str.split().str.len()
            
            # Initialize feature columns with zeros
            for feature in ['cc_entropy', 'cc_lexical_complexity', 'cc_pos_complexity', 'cc_med_entity_count']:
                df[feature] = 0.0
            
            # Process each valid complaint with progress bar
            for i, idx in enumerate(tqdm.tqdm(df.index, desc="Processing complexity features")):
                text = chief_complaints.loc[idx]
                
                # Calculate features
                df.at[idx, 'cc_entropy'] = complexity.calculate_entropy(text)
                df.at[idx, 'cc_lexical_complexity'] = complexity.lexical_complexity(text)
                df.at[idx, 'cc_pos_complexity'] = complexity.pos_complexity(text)
                df.at[idx, 'cc_med_entity_count'] = complexity.medical_entity_count(text)
                
            # Log completion once at the end
            added_columns = ['cc_entropy', 'cc_lexical_complexity', 'cc_pos_complexity', 
                          'cc_med_entity_count', 'cc_length', 'cc_word_count']
            
            logger.info(f"Added {len(added_columns)} complexity feature columns for chief complaints")
        
        logger.info('Feature creation complete')
        
        return df

    def create_validation_set(self, df: pd.DataFrame, n_validation: int = 5000, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split out a validation set from the main dataset.
        
        Args:
            df: Input DataFrame
            n_validation: Number of rows for validation set
            random_state: Random seed for reproducibility
            
        Returns:
            tuple: (train_df, validation_df)
        """
        logger.info('Creating validation set with %d samples', n_validation)
        
        # Randomly sample validation set
        validation_df = df.sample(n=n_validation, random_state=random_state)
        
        # Remove validation samples from training set
        train_df = df.drop(validation_df.index)
        
        logger.info('Split complete. Train size: %d, Validation size: %d', 
                   len(train_df), len(validation_df))
        
        return train_df, validation_df

    def handle(self, n_validation: int = 5000) -> pd.DataFrame:
        """
        Load, merge, clean and prepare the data
        
        Args:
            n_validation: Number of rows for validation set (default: 5000)
            
        Returns:
            pd.DataFrame: Training dataframe
        """
        # Load data
        triage, edstays, patients = self.load_data()
        
        # Merge data
        df = self.merge_data(triage, edstays, patients)
        
        # Clean data
        df = self.clean_data(df)
        
        # Create features
        df = self.create_features(df)
        
        # Split into train and validation sets
        train_df, validation_df = self.create_validation_set(df, n_validation=n_validation)
        
        # Save the datasets
        logger.info('Saving merged training data to %s', os.path.join(self.data_dir, 'train.csv'))
        train_df.to_csv(os.path.join(self.data_dir, 'train.csv'), index=False)
        
        logger.info('Saving validation data to %s', os.path.join(self.data_dir, 'validation.csv'))
        validation_df.to_csv(os.path.join(self.data_dir, 'validation.csv'), index=False)
        
        return train_df

    def main(self, n_validation: int = 5000):
        """
        Main entry point
        
        Args:
            n_validation: Number of rows for validation set (default: 5000)
        """
        df = self.handle(n_validation=n_validation)
        return df


if __name__ == "__main__":
    data_handler = DataHandler()
    train_df = data_handler.handle()
    logger.info('Data processing complete. Training data shape: %s', train_df.shape)
