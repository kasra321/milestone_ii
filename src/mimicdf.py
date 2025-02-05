import os
from pathlib import Path
import pandas as pd
from .gcp_utils import get_bigquery_client
from typing import Literal
from google.cloud import bigquery
from dotenv import load_dotenv
import tempfile
import requests
from time import sleep

# Load environment variables
load_dotenv()

# Project root directory (2 levels up from this file: src -> project root)
ROOT_DIR = Path(__file__).resolve().parent.parent
# Demo data directory
DEMO_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'demo')


# GCP settings
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
PHYSIONET_PROJECT = 'physionet-data'  # Project where MIMIC data is hosted

# Update the constants section
MIMIC_DATASETS = {
    'ed': 'mimiciv_ed',
    'hosp': 'mimiciv_hosp',
    'derived': 'mimiciv_derived'
}

class MIMICDF:

    """
    MIMICDF (MIMIC Database Interface)
    
    A class to interface with the MIMIC-IV Emergency Department database, supporting both
    local demo data and Google BigQuery access.

    Available Methods:
        edstays(): Get base cohort from edstays with demographics
        demographics(): Get subject demographics including gender, race, and age
        age(): Get age data for subjects
        diagnosis(): Get diagnosis codes and descriptions
        edstays_time(): Get edstays time series data including day of week and length of stay
        med_records(): Get medication records for a specific stay_id
        medications(): Get combined medication records from pyxis and medrecon
        pyxis(): Get medication dispensing records
        triage(): Get triage assessments
        vitals(): Get vital signs data
        model_data(): Get preprocessed data ready for modeling
        clear_cache(): Clear cached dataframes to free memory

    Args:
        source (str): Data source - either 'demo' for local demo data or 'gcp' for BigQuery data
        credentials (google.oauth2.credentials.Credentials, optional): Google Cloud credentials for BigQuery access

    Example:
        # For demo data
        mimic = MIMICDF(source='demo')
        
        # For BigQuery data
        credentials = service_account.Credentials.from_service_account_file('path/to/credentials.json')
        mimic = MIMICDF(source='gcp', credentials=credentials)
        
        # Get ED stays data
        edstays_df = mimic.edstays()
    """
    # constructor functions

    def __init__(self, source: Literal['demo', 'gcp'] = 'demo', credentials=None):
        """
        Initialize MIMICDF with either demo data or BigQuery data
        
        Args:
            source: Either 'demo' for local demo data or 'gcp' for BigQuery data
            credentials: Optional Google Cloud credentials object
        """
        if source not in ['demo', 'gcp']:
            raise ValueError("source must be either 'demo' or 'gcp'")
            
        self.source = source
        self.data_path = DEMO_DATA_DIR if source == 'demo' else None
        self._cache = {}
        
        if source == 'gcp':
            self.bq_client = get_bigquery_client(GCP_PROJECT_ID, credentials)
            
        # Add dataset mapping for each table
        self.table_dataset_map = {
            # ED tables
            'edstays': 'ed',
            'vitalsign': 'ed',
            'triage': 'ed',
            'pyxis': 'ed',
            'medrecon': 'ed',
            'diagnosis': 'ed',
            # HOSP tables - add your hosp tables here
            'patients': 'hosp',
            # derived tables
            'age': 'derived',
        }
        
    def _load(self, table_name: str) -> pd.DataFrame:
        """Internal method to load and cache dataframes."""
        if table_name not in self._cache:
            if self.source == 'demo':
                file_path = os.path.join(self.data_path, f"{table_name}.csv")
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"Demo file not found: {file_path}. "
                        "Please check if the demo files are present."
                    )
                self._cache[table_name] = pd.read_csv(file_path)
            else:  # bigquery
                if table_name not in self.table_dataset_map:
                    raise ValueError(f"Unknown table: {table_name}. Please add it to table_dataset_map")
                    
                dataset = MIMIC_DATASETS[self.table_dataset_map[table_name]]
                self._cache[table_name] = self.bq_client.get_table(
                    PHYSIONET_PROJECT, 
                    dataset,
                    table_name
                )
                    
        print(f"Table loaded: {table_name}")
        return self._cache[table_name]


    def clear_cache(self):
        """Clear cached dataframes to free memory."""
        print("Clearing cache")
        self._cache.clear()

    # data processing functions

    def edstays(self) -> pd.DataFrame:
        """Base cohort from edstays with demographics."""
        df = self._load('edstays')
        df[['intime', 'outtime']] = df[['intime', 'outtime']].apply(pd.to_datetime)
        return df

    def vitals(self, include_triage: bool = True) -> pd.DataFrame:
        """Vital signs"""
        return self._load('vitalsign')

    def triage(self) -> pd.DataFrame:
        """Triage assessments."""
        
        def map_pain(x):
            try:
                # Convert to float and clip between 0 and 10
                val = float(x)
                val = min(val, 10)  # Clip at 10
                if val < 0:
                    return pd.NA
                return int(val)  # Round down by converting to int
            except (ValueError, TypeError):
                return pd.NA

        def map_acuity(x):
            try:
                return int(x)
            except (ValueError, TypeError):
                return pd.NA

        def convert_to_float(x):
            try:
                return float(x)
            except (ValueError, TypeError):
                return pd.NA

        
        df = self._load('triage')

        # Convert numerical columns to float
        vital_columns = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']
        for col in vital_columns:
            df[col] = df[col].apply(convert_to_float).astype('Float64')

        # Apply the mapping to create new integer column
        df['pain'] = df['pain'].apply(map_pain).astype('Int64')
        df['acuity'] = df['acuity'].apply(map_acuity).astype('Int64')
        return df

    def pyxis(self) -> pd.DataFrame:
        """Medication dispensing records."""
        df = self._load('pyxis')
        df['charttime'] = pd.to_datetime(df['charttime'])
        return df

    def medrecon(self) -> pd.DataFrame:
        """Medication reconciliation records."""
        df = self._load('medrecon')
        df['charttime'] = pd.to_datetime(df['charttime'])
        return df

    def diagnosis(self) -> pd.DataFrame:
        """Diagnosis codes and descriptions."""
        return self._load('diagnosis')

    def ndc_mapping(self) -> pd.DataFrame:
        """NDC to ATC-4 mapping."""
        df = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'ndc_mapping.csv'))
        df.columns = ['ndc', 'atc4']
        df['ndc'] = df['ndc'].str.replace('-', '')
        return df

    def medications(self) -> pd.DataFrame:
        """Combined medication records from both pyxis and medrecon."""
        pyxis = self.pyxis()
        medrecon = self.medrecon()
        
        pyxis['source'] = 'pyxis'
        medrecon['source'] = 'medrecon'
        
        common_cols = ['subject_id', 'stay_id', 'charttime', 'name', 'source']
        return pd.concat([
            pyxis[common_cols],
            medrecon[common_cols]
        ], ignore_index=True)


    def vitals_triage(self, include_triage: bool = True) -> pd.DataFrame:
        """Vital signs with optional triage vitals inclusion."""
        vitals = self._load('vitalsign')
        vitals['charttime'] = pd.to_datetime(vitals['charttime'])
        
        if not include_triage:
            return vitals
            
        triage = self._load('triage')
        vitals['source'] = 'vitalsign'
        triage['source'] = 'triage'
        # Join admission time from edstays for triage
        edstays = self._load('edstays')
        triage = triage.merge(edstays[['stay_id', 'intime']], on='stay_id', how='left')
        triage['charttime'] = triage['intime']
        
        vital_cols = ['subject_id', 'stay_id', 'charttime', 'temperature', 
                     'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'source']
        combined = pd.concat([
            vitals[vital_cols],
            triage[vital_cols]
        ], ignore_index=True)
        
        return combined.sort_values(['subject_id', 'stay_id', 'charttime'])

    def demographics(self) -> pd.DataFrame:
        """Subjects demographics"""
        
        # load edstays
        edstays = self.edstays()
        edstays_columns = ['subject_id', 'gender', 'race']
        filtered_edstays = edstays[edstays_columns]

        # load age
        age = self._load('age')
        age_columns = ['subject_id', 'age']
        filtered_age = age[age_columns]

        # merge edstays and age
        demographics = filtered_edstays.merge(filtered_age, on='subject_id', how='left')
        demographics = filtered_edstays.groupby('subject_id').first().reset_index()
        
        # Create a mapping dictionary
        race_mapping = {
            # White categories
            'WHITE': 'WHITE',
            'WHITE - BRAZILIAN': 'WHITE',
            'WHITE - OTHER EUROPEAN': 'WHITE',
            'WHITE - RUSSIAN': 'WHITE',
            'WHITE - EASTERN EUROPEAN': 'WHITE',
            'PORTUGUESE': 'WHITE',
            
            # Black categories
            'BLACK/AFRICAN AMERICAN': 'BLACK',
            'BLACK/CAPE VERDEAN': 'BLACK',
            'BLACK/CARIBBEAN ISLAND': 'BLACK',
            'BLACK/AFRICAN': 'BLACK',
            
            # Asian categories
            'ASIAN': 'ASIAN',
            'ASIAN - SOUTH EAST ASIAN': 'ASIAN',
            'ASIAN - CHINESE': 'ASIAN',
            'ASIAN - KOREAN': 'ASIAN',
            'ASIAN - ASIAN INDIAN': 'ASIAN',
            
            # Hispanic categories
            'HISPANIC/LATINO - SALVADORAN': 'HISPANIC',
            'HISPANIC/LATINO - CUBAN': 'HISPANIC',
            'HISPANIC/LATINO - DOMINICAN': 'HISPANIC',
            'HISPANIC/LATINO - PUERTO RICAN': 'HISPANIC',
            'HISPANIC/LATINO - GUATEMALAN': 'HISPANIC',
            'HISPANIC/LATINO - MEXICAN': 'HISPANIC',
            'HISPANIC/LATINO - COLUMBIAN': 'HISPANIC',
            'HISPANIC/LATINO - HONDURAN': 'HISPANIC',
            'HISPANIC/LATINO - CENTRAL AMERICAN': 'HISPANIC',
            'HISPANIC OR LATINO': 'HISPANIC',
            'SOUTH AMERICAN': 'HISPANIC',
            
            # Other categories
            'MULTIPLE RACE/ETHNICITY': 'OTHER',
            'UNABLE TO OBTAIN': 'OTHER',
            'UNKNOWN': 'OTHER',
            'PATIENT DECLINED TO ANSWER': 'OTHER',
            'OTHER': 'OTHER',
            'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'OTHER',
            'AMERICAN INDIAN/ALASKA NATIVE': 'OTHER'
        }

        # Apply the mapping to race column
        demographics['race'] = demographics['race'].map(race_mapping)
        return demographics

    def edstays_time(self) -> pd.DataFrame:
        """Edstays time series"""
        edstays = self.edstays()
        
        # Create time-related columns
        df = pd.DataFrame()
        df['stay_id'] = edstays['stay_id']
        df['dow'] = edstays['intime'].dt.day_name()
        df['hour'] = edstays['intime'].dt.hour
        df['los_minutes'] = (edstays['outtime'] - edstays['intime']).dt.total_seconds() / 60
        
        return df

    def med_records(self, stay_id: int) -> pd.DataFrame:
        """
        Get all medication records from pyxis for a specific stay_id.
        
        Args:
            stay_id: The ED stay identifier
            
        Returns:
            pd.DataFrame: Medication records for the specified stay sorted by charttime
        """
        pyxis_df = self.pyxis()
        stay_meds = pyxis_df[pyxis_df['stay_id'] == stay_id].sort_values('charttime')
        return stay_meds

    def age(self) -> pd.DataFrame:
        """Age of subjects"""
        age = self._load('age')
        return age

    def ed_data(self) -> pd.DataFrame:
        """ED data with proper handling of nullable integers and time features"""
        print("Loading edstays...")
        edstays = self.edstays()
        
        print("Loading age data...")
        age = self.age()
        
        print("Loading time features...")
        edstays_time = self.edstays_time()
        
        print("Loading triage data...")
        triage = self.triage()
        
        print("Processing age calculations...")
        # Start with just the columns we need from edstays
        df = edstays[['subject_id', 'hadm_id', 'stay_id', 'gender', 'race', 
                      'arrival_transport', 'disposition', 'intime', 'outtime']]
        
        # Clean up arrival_transport
        df['arrival_transport'] = df['arrival_transport'].apply(
            lambda x: 'WALK IN' if x == 'WALK IN'
            else 'AMBULANCE' if x == 'AMBULANCE'
            else 'OTHER'
        )
        
        # Clean up disposition
        df['disposition'] = df['disposition'].apply(
            lambda x: 'HOME' if x == 'HOME'
            else 'ADMITTED' if x == 'ADMITTED'
            else 'OTHER'
        )
        
        # Merge with age data
        anchor_age = age[['subject_id', 'anchor_age', 'anchor_year']]
        df = df.merge(anchor_age, on='subject_id', how='left')
        
        print("Calculating ED visit age...")
        df['age_at_ed'] = df['anchor_age'] + (df['intime'].dt.year - df['anchor_year'])
        df['age_at_ed'] = df['age_at_ed'].astype('Int64')
        
        print("Merging time features...")
        df = df.merge(edstays_time, on='stay_id', how='left')
        
        print("Merging triage features...")
        triage_features = triage.drop(['subject_id', 'chiefcomplaint'], axis=1)
        df = df.merge(triage_features, on='stay_id', how='left')
        
        print("Cleaning up columns...")
        df = df.drop(['intime', 'outtime', 'anchor_year', 'anchor_age', 'hadm_id'], axis=1)
        
        print(f'\n Dataframe shape: {df.shape} \n')
        print('Dataframe info: \n')
        print(df.info())

        return df