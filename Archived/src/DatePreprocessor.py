import numpy as np

class DataPreprocessor:
    def __init__(self, preprocessed_data):
        self.data = preprocessed_data
        
    def clean_blood_pressure(self, data):
        """Clean blood pressure readings using statistical and physiological rules"""
        # Create mask for invalid blood pressure values
        invalid_bp = ~(
            (data['dbp'] > 0) & 
            (data['sbp'] > 0) & 
            (data['dbp'] <= 300) & 
            (data['sbp'] <= 350) &
            (data['dbp'] < data['sbp'])
        )
        # Replace invalid values with np.nan
        data.loc[invalid_bp, ['dbp', 'sbp']] = np.nan
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
    
    def prepare_data(self):
        """Main pipeline for data preparation"""
        data = self.data.ed_data()  # Get raw data
        
        # Apply cleaning functions
        data = self.clean_blood_pressure(data)
        data = self.clean_o2sat(data)
        data = self.clean_resprate(data)
        data = self.clean_temperature(data)
        data = self.clean_heartrate(data)
        data = self.clean_los(data)

        return data