import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from config import Config

class FeatureEngineer:
    def __init__(self):
        # We will use a ColumnTransformer to handle numeric and categorical features separately
        # But for manual control and transparency in this project, we'll manage them explicitly.
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.numeric_features = [] # To be populated during fit
        self.categorical_features = ['type']

    def create_features(self, df):
        """Creates new features. Does NOT encode categorical ones yet."""
        df = df.copy()

        # Balance Errors
        # Origin: old - amount = new (expected)
        # Error = new - (old - amount)
        df['error_balance_orig'] = df['new_balance_orig'] + df['amount'] - df['old_balance_org']
        
        # Destination: old + amount = new (expected)
        # Error = new - (old + amount)
        df['error_balance_dest'] = df['old_balance_dest'] + df['amount'] - df['new_balance_dest']
        
        # Temporal Feature: Step is roughly hours (1 step = 1 hour in PaySim)
        # We can extract 'hour of day' assuming step 1 is 00:00 or similar cyclic pattern
        if 'step' in df.columns:
            df['hour_of_day'] = df['step'] % 24

        return df

    def select_features(self, df):
        """Selects features for the model. Drops IDs but KEEPS 'type' for encoding."""
        # Drop non-numeric identifiers and target
        drop_cols = ['step', 'name_orig', 'name_dest', 'is_fraud', 'is_flagged_fraud']
        # We keep 'type' because we will encode it in fit/transform
        features = df.drop([c for c in drop_cols if c in df.columns], axis=1)
        return features

    def fit(self, X):
        """Fits the encoder and scaler on the training data."""
        # Identify numeric columns (excluding the categorical ones)
        self.numeric_features = [c for c in X.columns if c not in self.categorical_features]
        
        # Fit OneHotEncoder on categorical features
        self.encoder.fit(X[self.categorical_features])
        
        # Transform categorical to get shape
        encoded_cols = self.encoder.get_feature_names_out(self.categorical_features)
        
        # Prepare numeric data for Scaler fit
        X_numeric = X[self.numeric_features]
        
        # We can verify X_numeric is all numeric
        self.scaler.fit(X_numeric)
        
        # Save artifacts
        self.save_state()
        return self

    def transform(self, X):
        """Transforms new data using the fitted encoder and scaler."""
        # 1. Encode Categorical
        X_encoded = self.encoder.transform(X[self.categorical_features])
        encoded_cols = self.encoder.get_feature_names_out(self.categorical_features)
        df_encoded = pd.DataFrame(X_encoded, columns=encoded_cols, index=X.index)
        
        # 2. Scale Numeric
        X_numeric = X[self.numeric_features]
        X_scaled_array = self.scaler.transform(X_numeric)
        df_numeric_scaled = pd.DataFrame(X_scaled_array, columns=self.numeric_features, index=X.index)
        
        # 3. Concatenate
        X_final = pd.concat([df_numeric_scaled, df_encoded], axis=1)
        
        return X_final

    def fit_transform(self, X):
        """Fits and transforms in one step."""
        self.fit(X)
        return self.transform(X)

    def save_state(self):
        """Saves the encoder, scaler, and feature lists."""
        state = {
            'scaler': self.scaler,
            'encoder': self.encoder,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features
        }
        joblib.dump(state, Config.SCALER_PATH)

    def load_state(self):
        """Loads the encoder and scaler."""
        try:
            state = joblib.load(Config.SCALER_PATH)
            self.scaler = state['scaler']
            self.encoder = state['encoder']
            self.numeric_features = state['numeric_features']
            self.categorical_features = state['categorical_features']
            return True
        except FileNotFoundError:
            return False
