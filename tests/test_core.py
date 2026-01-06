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

def test_feature_creation(mock_data):
    """Test feature creation (before encoding)."""
    pipeline = DataPipeline()
    engineer = FeatureEngineer()
    
    processed_df = pipeline.preprocess(mock_data)
    features_df = engineer.create_features(processed_df)
    
    # Check for new features
    assert 'error_balance_orig' in features_df.columns
    assert 'error_balance_dest' in features_df.columns
    assert 'hour_of_day' in features_df.columns
    
    # Check type is STILL there (not encoded yet)
    assert 'type' in features_df.columns

def test_feature_transformation(mock_data):
    """Test full transformation (encoding + scaling)."""
    pipeline = DataPipeline()
    engineer = FeatureEngineer()
    
    processed_df = pipeline.preprocess(mock_data)
    features_df = engineer.create_features(processed_df)
    X = engineer.select_features(features_df)
    
    # Fit and Transform
    X_transformed = engineer.fit_transform(X)
    
    # Check output format
    assert isinstance(X_transformed, pd.DataFrame)
    
    # Check if encoded columns exist (OneHotEncoder should create type_PAYMENT etc.)
    # Note: Column names might vary depending on OHE version but usually "type_PAYMENT"
    cols = X_transformed.columns.tolist()
    assert any('type_' in col for col in cols)
    
    # Check if original 'type' string column is gone
    assert 'type' not in X_transformed.columns
    
    # Check if numeric features are scaled (roughly) - hard to check exact values without manual math
    # but we can check existence
    assert 'amount' in X_transformed.columns
    assert 'error_balance_orig' in X_transformed.columns

def test_schema_validation(mock_data):
    """Test that incoming data adheres to expected schema (types)."""
    pipeline = DataPipeline()
    df = pipeline.preprocess(mock_data)
    
    # Define expected schema
    required_columns = {
        'step': 'int', # or int64
        'type': 'object', # string
        'amount': 'float',
        'name_orig': 'object',
        'old_balance_org': 'float',
        'new_balance_orig': 'float',
        'name_dest': 'object',
        'old_balance_dest': 'float',
        'new_balance_dest': 'float'
    }
    
    # Validate
    for col, expected_type in required_columns.items():
        assert col in df.columns, f"Missing column: {col}"
        
        # Simple type check
        if expected_type == 'float':
            assert pd.api.types.is_float_dtype(df[col]), f"Column {col} should be float"
        elif expected_type == 'int':
            assert pd.api.types.is_integer_dtype(df[col]), f"Column {col} should be int"
        elif expected_type == 'object':
            assert pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]), f"Column {col} should be string/object"

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
    
    # Fit Transform
    X_processed = engineer.fit_transform(X)
    
    # Train
    model.train(X_processed, y)
    
    # Predict
    preds = model.predict(X_processed)
    
    assert len(preds) == len(mock_data)
    assert all(p in [0, 1] for p in preds)