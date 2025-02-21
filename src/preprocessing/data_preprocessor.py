"""
This module contains components for data preprocessing pipeline:

DataCleaner
    - Basic data cleaning
    - Categorical validation
    - Vital signs cleaning
    - Blood pressure validation

FeatureEngineer (to be implemented)
    - Time-based features
    - Interaction features
    - Domain-specific features

DataTransformer (to be implemented)
    - Scaling
    - Encoding
    - Dimensionality reduction

DataPruner (to be implemented)
    - Outlier removal
    - Feature selection
    - Sample balancing
"""

import numpy as np
import pandas as pd

class DataCleaner:
    def __init__(self, data):
        self.data = data.copy()
        
        # Define valid values for categorical variables
        self.valid_transport = ['AMBULANCE', 'WALK IN']
        self.valid_disposition = ['HOME', 'ADMITTED']
        
        # Define expected ranges for vital signs
        self.vital_ranges = {
            'temperature': (56, 111.4),   # Fahrenheit
            'heartrate': (13, 250),       # beats per minute
            'resprate': (1, 97),          # breaths per minute
            'o2sat': (8, 100),            # percentage
            'sbp': (5, 299),              # mmHg
            'dbp': (0, 297),              # mmHg
            'pain': (0, 10),              # pain scale
            'acuity': (1, 5)              # acuity scale
        }

        self.data_clean = self.prepare_data()

    def prepare_data(self):
        """Main method to clean and prepare the data"""
        df = self.data.copy()
        
        # Clean categorical variables
        df = self._clean_categorical(df)
        
        # Clean vital signs
        df = self._clean_vitals(df)
        
        # Clean blood pressure (special case due to relationship between sbp/dbp)
        df = self._clean_blood_pressure(df)
        
        return df

    def _clean_categorical(self, df):
        """Clean categorical variables"""
        # Filter to valid transport modes
        df.loc[~df['arrival_transport'].isin(self.valid_transport), 'arrival_transport'] = np.nan
        
        # Filter to valid dispositions
        df.loc[~df['disposition'].isin(self.valid_disposition), 'disposition'] = np.nan
        
        return df

    def _clean_vitals(self, df):
        """Clean vital signs using physiological ranges"""
        for vital, (min_val, max_val) in self.vital_ranges.items():
            if vital not in ['sbp', 'dbp']:  # Blood pressure handled separately
                df.loc[~df[vital].between(min_val, max_val), vital] = np.nan
        return df

    def _clean_blood_pressure(self, df):
        """Clean blood pressure readings using statistical and physiological rules"""
        # Mark invalid where dbp >= sbp or outside physiological ranges
        invalid_bp = (
            (df['dbp'] >= df['sbp']) | 
            ~df['sbp'].between(*self.vital_ranges['sbp']) |
            ~df['dbp'].between(*self.vital_ranges['dbp'])
        )
        
        # Set invalid readings to NaN
        df.loc[invalid_bp, ['sbp', 'dbp']] = np.nan
        
        return df
    

class FeatureEngineer:
    def __init__(self, data):
        self.data = data.copy()

    def engineer_features(self):
        """Main method to engineer all features"""
        df = self.data.copy()
        
        # Add clinical metrics first (needed for some categories)
        df = self._add_clinical_metrics(df)
        
        # Add vital sign categories
        df = self._add_vital_categories(df)
        
        # Add demographic features
        df = self._add_demographic_features(df)
        
        # Add clinical scores
        df = self._add_clinical_scores(df)
        
        # Add temporal features
        df = self._add_temporal_features(df)
        
        return df

    def _add_clinical_metrics(self, df):
        """Add calculated clinical metrics"""
        # All metrics as float
        df['map'] = df['dbp'] + (df['sbp'] - df['dbp'])/3
        df['pulse_pressure'] = df['sbp'] - df['dbp']
        df['shock_index'] = df['heartrate'] / df['sbp']
        df['rate_pressure_product'] = df['heartrate'] * df['sbp']
        df['temp_delta'] = df['temperature'] - 98.6
        
        return df

    def _add_vital_categories(self, df):
        """Add categorical features for vital signs"""
        # Binary features as int (1/0), with NA handling
        df['is_tachycardic'] = df['heartrate'].gt(100).fillna(False).astype(int)
        df['is_bradycardic'] = df['heartrate'].lt(60).fillna(False).astype(int)
        df['is_hypoxic'] = df['o2sat'].lt(92).fillna(False).astype(int)
        df['is_hypertensive'] = df['sbp'].gt(140).fillna(False).astype(int)
        df['is_hypotensive'] = df['sbp'].lt(90).fillna(False).astype(int)
        df['is_febrile'] = df['temperature'].gt(99.5).fillna(False).astype(int)
        df['is_hypothermic'] = df['temperature'].lt(97.0).fillna(False).astype(int)
        
        # Categorical features with np.nan for missing
        df['hr_category'] = df['heartrate'].apply(self._categorize_hr)
        df['resp_category'] = df['resprate'].apply(self._categorize_resp)
        df['o2_category'] = df['o2sat'].apply(self._categorize_pulseox)
        df['sbp_category'] = df['sbp'].apply(self._categorize_bp)
        df['dbp_category'] = df['dbp'].apply(self._categorize_dbp)
        df['temp_category'] = df['temperature'].apply(self._categorize_temp)
        df['pain_category'] = df['pain'].apply(self._categorize_pain)
        
        return df

    def _add_demographic_features(self, df):
        """Add demographic-based features"""
        df['age_group'] = df['age_at_ed'].apply(self._categorize_age)
        df['acuity_group'] = df['acuity'].apply(self._categorize_acuity)
        return df

    def _add_clinical_scores(self, df):
        """Add clinical scoring features"""
        # Binary feature as int (1/0)
        df['has_sirs'] = df.apply(
            lambda x: self._calculate_sirs(
                x['temperature'], 
                x['heartrate'], 
                x['resprate']
            ), axis=1
        ).astype(int)
        return df

    def _add_temporal_features(self, df):
        """Add time-based features"""
        # Binary feature as int (1/0)
        df['is_daytime'] = (
            (df['hour'] >= 7) & (df['hour'] < 19)
        ).fillna(False).astype('Int64')
        
        # Only calculate week_minutes if all components are present
        mask = df['dow'].notna() & df['hour'].notna() & df['minute'].notna()
        df['week_minutes'] = np.nan  # initialize with NaN
        
        if mask.any():
            # Map full day names to numbers (Monday = 0)
            day_map = {
                'Monday': 0,
                'Tuesday': 1,
                'Wednesday': 2,
                'Thursday': 3,
                'Friday': 4,
                'Saturday': 5,
                'Sunday': 6
            }
            
            # Calculate minutes for valid rows
            days = df.loc[mask, 'dow'].map(day_map)
            minutes = (
                days * 24 * 60 +  # days to minutes
                df.loc[mask, 'hour'] * 60 +  # hours to minutes
                df.loc[mask, 'minute']  # minutes
            )
            
            # Assign float values first
            df.loc[mask, 'week_minutes'] = minutes.round()
        
        # Convert to nullable integer type at the end
        df['week_minutes'] = df['week_minutes'].astype('Int64')
        
        return df

    def _categorize_hr(self, hr):
        if pd.isna(hr): return np.nan
        if hr < 60: return "bradycardic"
        if hr > 100: return "tachycardic"
        return "normal"

    def _categorize_resp(self, resp):
        if pd.isna(resp): return np.nan
        if resp < 12: return "low"
        if resp > 20: return "high"
        return "normal"

    def _categorize_pulseox(self, po):
        if pd.isna(po): return np.nan
        return "low" if po < 92 else "normal"

    def _categorize_bp(self, bp):
        if pd.isna(bp): return np.nan
        if bp < 90: return "low"
        if bp > 140: return "high"
        return "normal"

    def _categorize_temp(self, temp):
        if pd.isna(temp): return np.nan
        if temp < 97.0: return "hypothermic"
        if temp > 99.5: return "febrile"
        return "normal"

    def _categorize_dbp(self, dbp):
        if pd.isna(dbp): return np.nan
        if dbp < 60: return "low"
        if dbp > 90: return "high"
        return "normal"

    def _categorize_pain(self, pain):
        if pd.isna(pain): return np.nan
        if pain == 0: return "no pain"
        if pain <= 3: return "mild"
        if pain <= 6: return "moderate"
        return "severe"

    def _categorize_acuity(self, acuity):
        try:
            return f"acuity_{int(acuity)}"
        except:
            return np.nan

    def _categorize_age(self, age):
        try:
            age = float(age)
        except:
            return np.nan
        if age < 18: return "child"
        if age < 36: return "young_adult"
        if age < 56: return "adult"
        if age < 76: return "middle_aged"
        return "senior"

    def _calculate_sirs(self, temp, hr, resp):
        if pd.isna(temp) or pd.isna(hr) or pd.isna(resp):
            return 0
        count = 0
        if (temp > 38) or (temp < 36): count += 1
        if hr > 90: count += 1
        if resp > 20: count += 1
        return 1 if count >= 2 else 0