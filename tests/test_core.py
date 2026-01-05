import sys
import os
import pandas as pd
import numpy as np
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer
from fraud_model import FraudModel

@pytest.fixture
def mock_data():
    """Creates a small mock dataset for testing."""
    data = {
        'step': [1, 1, 1],
        'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT'],
        'amount': [100.0, 200.0, 300.0],
        'nameOrig': ['C1', 'C2', 'C3'],
        'oldbalanceOrg': [1000.0, 2000.0, 3000.0],
        'newbalanceOrig': [900.0, 1800.0, 2700.0],
        'nameDest': ['M1', 'M2', 'M3'],
        'oldbalanceDest': [0.0, 0.0, 0.0],
        'newbalanceDest': [0.0, 0.0, 0.0],
        'isFraud': [0, 0, 1],
        'isFlaggedFraud': [0, 0, 0]
    }
    return pd.DataFrame(data)

def test_pipeline_preprocess(mock_data):
    """Test data preprocessing (renaming, dropping columns)."""
    pipeline = DataPipeline()
    processed_df = pipeline.preprocess(mock_data)
    
    expected_cols = ['step', 'type', 'amount', 'name_orig', 'old_balance_org', 
                     'new_balance_orig', 'name_dest', 'old_balance_dest', 
                     'new_balance_dest', 'is_fraud']
    
    # Check if all expected columns are present
    for col in expected_cols:
        assert col in processed_df.columns
    
    # Check if dropped column is gone
    assert 'is_flagged_fraud' not in processed_df.columns

def test_feature_engineering(mock_data):
    """Test feature creation and encoding."""
    pipeline = DataPipeline()
    engineer = FeatureEngineer()
    
    processed_df = pipeline.preprocess(mock_data)
    features_df = engineer.create_features(processed_df)
    
    # Check for new features
    assert 'error_balance_orig' in features_df.columns
    assert 'error_balance_dest' in features_df.columns
    
    # Check one-hot encoding
    # We expect type_PAYMENT, type_TRANSFER, type_CASH_OUT (minus one if drop_first=True)
    # Since we have 3 types, drop_first=True means 2 dummy columns if all present, 
    # but pandas get_dummies depends on input.
    # Just check if 'type' column is gone
    assert 'type' not in features_df.columns

def test_model_training_and_prediction(mock_data):
    """Test model training and prediction flow."""
    pipeline = DataPipeline()
    engineer = FeatureEngineer()
    model = FraudModel()
    
    # Prepare data
    processed_df = pipeline.preprocess(mock_data)
    features_df = engineer.create_features(processed_df)
    
    y = features_df['is_fraud']
    X = engineer.select_features(features_df)
    
    # Fill NA if any (one-hot might introduce them if not careful, but here should be fine)
    X = X.fillna(0)
    
    # Scale
    X_scaled = engineer.fit_transform_scaler(X)
    
    # Train
    model.train(X_scaled, y)
    
    # Predict
    preds = model.predict(X_scaled)
    
    assert len(preds) == len(mock_data)
    assert all(p in [0, 1] for p in preds)

