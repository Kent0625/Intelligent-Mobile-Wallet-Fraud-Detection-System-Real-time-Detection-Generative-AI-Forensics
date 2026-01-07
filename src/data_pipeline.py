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
            print("Falling back to Synthetic Data Generation...")
            return self.generate_synthetic_data()

    def generate_synthetic_data(self, n_rows=1000):
        """Generates synthetic data mimicking PaySim structure."""
        np.random.seed(42)
        
        data = {
            'step': np.random.randint(1, 744, n_rows),
            'type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], n_rows),
            'amount': np.round(np.random.uniform(10, 10000, n_rows), 2),
            'nameOrig': [f'C{np.random.randint(1000000, 9999999)}' for _ in range(n_rows)],
            'oldbalanceOrg': np.round(np.random.uniform(0, 100000, n_rows), 2),
            'newbalanceOrig': np.zeros(n_rows),
            'nameDest': [f'M{np.random.randint(1000000, 9999999)}' for _ in range(n_rows)],
            'oldbalanceDest': np.round(np.random.uniform(0, 100000, n_rows), 2),
            'newbalanceDest': np.zeros(n_rows),
            'isFraud': np.random.choice([0, 1], n_rows, p=[0.9, 0.1]),
            'isFlaggedFraud': np.zeros(n_rows)
        }
        
        df = pd.DataFrame(data)
        
        # Simple logic to make balances consistent for non-fraud, inconsistent for fraud
        # This helps the model (which looks for balance errors) actually detect something.
        
        # Vectorized update for speed
        # Default: Everything matches
        df['newbalanceOrig'] = df['oldbalanceOrg'] - df['amount']
        df['newbalanceDest'] = df['oldbalanceDest'] + df['amount']
        
        # Fix negatives
        df['newbalanceOrig'] = df['newbalanceOrig'].clip(lower=0)
        
        # Sabotage the math for Frauds (creating "balance errors")
        frauds = df['isFraud'] == 1
        df.loc[frauds, 'newbalanceOrig'] = df.loc[frauds, 'oldbalanceOrg'] # Money didn't leave?
        df.loc[frauds, 'newbalanceDest'] = df.loc[frauds, 'oldbalanceDest'] # Money didn't arrive?
        
        return df

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
