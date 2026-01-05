import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from config import Config

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()

    def create_features(self, df):
        """Creates new features and encodes categorical ones."""
        df = df.copy()

        # Balance Errors
        # Origin: old - amount = new (expected)
        # Error = new - (old - amount)
        df['error_balance_orig'] = df['new_balance_orig'] + df['amount'] - df['old_balance_org']
        
        # Destination: old + amount = new (expected)
        # Error = new - (old + amount)
        df['error_balance_dest'] = df['old_balance_dest'] + df['amount'] - df['new_balance_dest']

        # One-Hot Encoding for 'type'
        # We perform get_dummies. Note: In production, we need to ensure columns align.
        # For simplicity in this demo, we assume all types are present or handle alignment later.
        df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=True)

        return df

    def select_features(self, df):
        """Selects features for the model."""
        # Drop non-numeric identifiers
        drop_cols = ['step', 'name_orig', 'name_dest', 'is_fraud']
        features = df.drop([c for c in drop_cols if c in df.columns], axis=1)
        return features

    def fit_transform_scaler(self, X):
        """Fits and transforms the scaler. Saves feature names."""
        # Save the feature names to ensure consistency
        self.feature_names = X.columns.tolist()
        joblib.dump(self.feature_names, Config.SCALER_PATH.replace('.pkl', '_columns.pkl'))
        
        X_scaled = self.scaler.fit_transform(X)
        joblib.dump(self.scaler, Config.SCALER_PATH)
        return X_scaled

    def transform_scaler(self, X):
        """Transforms data using the loaded scaler and enforces column order."""
        try:
             self.scaler = joblib.load(Config.SCALER_PATH)
             self.feature_names = joblib.load(Config.SCALER_PATH.replace('.pkl', '_columns.pkl'))
        except:
             pass 

        # Enforce column order and fill missing with 0
        if hasattr(self, 'feature_names'):
            # Add missing cols
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            # Drop extra cols
            X = X[self.feature_names]
            
        return self.scaler.transform(X)
