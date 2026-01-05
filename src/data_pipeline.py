import pandas as pd
import numpy as np
from config import Config

class DataPipeline:
    def __init__(self):
        self.data_path = Config.DATA_PATH

    def load_data(self):
        """Loads data from the CSV file."""
        try:
            df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {self.data_path}")
            return None

    def preprocess(self, df):
        """Basic preprocessing steps."""
        # Rename columns for easier access
        df = df.rename(columns={
            'step': 'step',
            'type': 'type',
            'amount': 'amount',
            'nameOrig': 'name_orig',
            'oldbalanceOrg': 'old_balance_org',
            'newbalanceOrig': 'new_balance_orig',
            'nameDest': 'name_dest',
            'oldbalanceDest': 'old_balance_dest',
            'newbalanceDest': 'new_balance_dest',
            'isFraud': 'is_fraud',
            'isFlaggedFraud': 'is_flagged_fraud'
        })
        
        # Drop isFlaggedFraud as it's a rule-based feature we don't need for training
        if 'is_flagged_fraud' in df.columns:
            df = df.drop('is_flagged_fraud', axis=1)

        return df
