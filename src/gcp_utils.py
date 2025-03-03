from google.cloud import bigquery
from google.oauth2 import service_account
from google.auth import default
from google.api_core import exceptions
import os
import pandas as pd
from typing import Optional, Union

class BigQueryAccessError(Exception):
    """Custom exception for BigQuery access issues"""
    pass

class BigQueryClient:
    def __init__(self, project_id: str, credentials=None):
        """Initialize BigQuery client"""
        self.project_id = project_id
        try:
            # Use provided credentials if available
            self.client = bigquery.Client(
                project=project_id,
                credentials=credentials
            )
            
            # Test access specifically to MIMIC dataset
            test_query = """
            SELECT COUNT(*) as count
            FROM `physionet-data.mimiciv_ed.edstays` 
            LIMIT 1
            """
            result = self.client.query(test_query).result()
            print("Successfully connected to MIMIC-IV ED dataset")
            
        except exceptions.Forbidden as e:
            auth_instructions = """
Access Denied. Please ensure:
1. You have completed the PhysioNet credentialing process
2. You have been granted access to MIMIC-IV ED dataset
3. You are using the same Google account that has access to BigQuery

To re-authenticate:
   gcloud auth application-default login
"""
            raise BigQueryAccessError(f"{str(e)}\n\n{auth_instructions}")
        except Exception as e:
            raise BigQueryAccessError(f"Failed to initialize BigQuery client: {str(e)}")

    def query_to_dataframe(self, query: str) -> pd.DataFrame:
        """Execute a BigQuery query and return results as DataFrame"""
        try:
            return self.client.query(query).to_dataframe()
        except exceptions.Forbidden as e:
            raise BigQueryAccessError(f"Permission denied when executing query: {str(e)}")
        except Exception as e:
            raise BigQueryAccessError(f"Query execution failed: {str(e)}")
    
    def get_table(self, project: str, dataset: str, table: str) -> pd.DataFrame:
        """Get a full table from BigQuery"""
        query = f"""
        SELECT *
        FROM `{project}.{dataset}.{table}`
        """
        try:
            return self.query_to_dataframe(query)
        except Exception as e:
            raise BigQueryAccessError(
                f"Failed to access table {project}.{dataset}.{table}: {str(e)}"
            )

def get_bigquery_client(
    project_id: str, 
    credentials: Union[str, service_account.Credentials, None] = None
) -> BigQueryClient:
    """
    Create a BigQuery client with credentials
    
    Args:
        project_id: GCP project ID
        credentials: Either a path to credentials file (str) or a Credentials object
    """
    if isinstance(credentials, str):
        # If credentials is a string, treat it as a path
        credentials = service_account.Credentials.from_service_account_file(credentials)
    elif credentials is None:
        # Try to use application default credentials
        credentials, _ = default()
    return BigQueryClient(project_id, credentials)