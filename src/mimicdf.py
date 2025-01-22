import os
from pathlib import Path
import pandas as pd
from .gcp_utils import get_bigquery_client
from typing import Literal
from google.cloud import bigquery
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Project root directory (2 levels up from this file: src -> project root)
ROOT_DIR = Path(__file__).resolve().parent.parent
# Demo data directory
DEMO_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'demo')

# GCP settings
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
PHYSIONET_PROJECT = 'physionet-data'  # Project where MIMIC data is hosted
MIMIC_DATASET = 'mimiciv_ed'  # The exact BigQuery dataset name

class MIMICDF:
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
                self._cache[table_name] = self.bq_client.get_table(
                    PHYSIONET_PROJECT, 
                    MIMIC_DATASET, 
                    table_name
                )
                    
        print(f"Table loaded: {table_name}")
        return self._cache[table_name]


    def clear_cache(self):
        """Clear cached dataframes to free memory."""
        print("Clearing cache")
        self._cache.clear()


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
                return int(val)  # Round down by converting to int
            except (ValueError, TypeError):
                # Return -1 for non-numeric values or missing (None/NaN)
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

    def subjects_demographics(self) -> pd.DataFrame:
        """Subjects demographics"""
        edstays = self.edstays()
        columns = ['subject_id', 'gender', 'race']
        filtered_edstays = edstays[columns]
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
        columns = ['stay_id', 'subject_id', 'intime', 'outtime', 'dow', 'hour', 'los_minutes']
        edstays['dow'] = edstays['intime'].dt.day_name()
        edstays['hour'] = edstays['intime'].dt.hour
        edstays['los_minutes'] = (edstays['outtime'] - edstays['intime']).dt.total_seconds() / 60
        edstays_time = edstays[columns]
        return edstays_time
