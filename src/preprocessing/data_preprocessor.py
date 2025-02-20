import numpy as np
import pandas as pd

class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        # Define valid categories
        self.valid_transport = ['AMBULANCE', 'WALK-IN']
        self.valid_disposition = ['HOME', 'ADMITTED']
        
        # Define column name mappings
        self.column_map = {
            'transport': ['transport', 'transport_mode', 'arrival_transport'],
            'disposition': ['disposition', 'disposition_type', 'discharge_disposition'],
            'age_at_ed': ['age_at_ed', 'age', 'patient_age'],
            'dbp': ['dbp', 'diastolic_bp'],
            'sbp': ['sbp', 'systolic_bp'],
            'o2sat': ['o2sat', 'oxygen_saturation'],
            'resprate': ['resprate', 'respiratory_rate'],
            'temperature': ['temperature', 'temp'],
            'heartrate': ['heartrate', 'heart_rate'],
            'los_minutes': ['los_minutes', 'length_of_stay_minutes']
        }
        
        # Validate and set actual column names
        self.actual_columns = self._get_actual_column_names()
    
    def _get_actual_column_names(self):
        """Maps standard column names to actual column names in the dataset"""
        actual_columns = {}
        for standard_name, possible_names in self.column_map.items():
            found_names = [name for name in possible_names if name in self.data.columns]
            if found_names:
                actual_columns[standard_name] = found_names[0]
            else:
                print(f"Warning: No matching column found for {standard_name}")
                actual_columns[standard_name] = None
        return actual_columns

    def clean_blood_pressure(self, data):
        """Clean blood pressure readings using statistical and physiological rules"""
        dbp_col = self.actual_columns['dbp']
        sbp_col = self.actual_columns['sbp']
        
        if not (dbp_col and sbp_col):
            return data
            
        invalid_bp = ~(
            (data[dbp_col] > 0) & 
            (data[sbp_col] > 0) & 
            (data[dbp_col] <= 300) & 
            (data[sbp_col] <= 350) &
            (data[dbp_col] < data[sbp_col])
        )
        data.loc[invalid_bp, [dbp_col, sbp_col]] = np.nan
        return data
    
    def clean_o2sat(self, data):
        """Clean oxygen saturation readings"""
        data.loc[~((data['o2sat'] >= 0) & (data['o2sat'] <= 100)), 'o2sat'] = np.nan
        return data
    
    def clean_resprate(self, data):
        """Clean respiratory rate using statistical approach"""
        data.loc[~((data['resprate'] > 0) & (data['resprate'] < 300)), 'resprate'] = np.nan
        return data
    
    def clean_temperature(self, data):
        """Clean temperature readings"""
        data.loc[~((data['temperature'] > 50) & (data['temperature'] < 120)), 'temperature'] = np.nan
        return data
    
    def clean_heartrate(self, data):
        """Clean heart rate readings"""
        data.loc[~((data['heartrate'] > 0) & (data['heartrate'] < 300)), 'heartrate'] = np.nan
        return data
    
    def clean_los(self, data):
        """Clean los readings"""
        data.loc[~(data['los_minutes'] > 0), 'los_minutes'] = np.nan
        return data
    
    def filter_ed_quality_cohort(self, data):
        """
        Filters ED cohort based on data quality assessment and inclusion criteria.
        
        Inclusion criteria:
        - Transport: AMBULANCE and WALK-IN only
        - Disposition: HOME and ADMITTED only
        - Excludes age_at_ed due to systematic missingness
        """
        original_size = len(data)
        
        # Get actual column names
        transport_col = self.actual_columns['transport']
        disposition_col = self.actual_columns['disposition']
        age_col = self.actual_columns['age_at_ed']
        
        if not (transport_col and disposition_col):
            raise ValueError("Required columns 'transport' and 'disposition' not found in data")
        
        # Apply filters
        cohort = data[
            (data[transport_col].isin(self.valid_transport)) &
            (data[disposition_col].isin(self.valid_disposition))
        ].copy()
        
        # Drop age if it exists
        if age_col and age_col in cohort.columns:
            cohort = cohort.drop(columns=[age_col])
        
        # Log cohort selection results
        final_size = len(cohort)
        print(f"Cohort Selection Results:")
        print(f"Original samples: {original_size:,}")
        print(f"Final samples: {final_size:,}")
        print(f"Retained: {(final_size/original_size)*100:.1f}%")
        
        return cohort
    
    def prepare_data(self):
        """Main pipeline for data preparation
        Returns:
            DataFrame with cleaned vital signs and filtered cohort
        """
        try:
            # First filter the cohort
            filtered_data = self.filter_ed_quality_cohort(self.data)
            
            # Apply cleaning functions in sequence
            cleaning_functions = [
                self.clean_blood_pressure,
                self.clean_o2sat,
                self.clean_resprate, 
                self.clean_temperature,
                self.clean_heartrate,
                self.clean_los
            ]
            
            for clean_func in cleaning_functions:
                filtered_data = clean_func(filtered_data)
                
            return filtered_data
            
        except Exception as e:
            raise Exception(f"Error in data preparation pipeline: {str(e)}")