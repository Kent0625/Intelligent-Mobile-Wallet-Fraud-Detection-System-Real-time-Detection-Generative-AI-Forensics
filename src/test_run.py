import os
import sys

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer
from fraud_model import FraudModel

def test_full_pipeline():
    print(f"Checking data path: {Config.DATA_PATH}")
    if not os.path.exists(Config.DATA_PATH):
        print("FAIL: Data file still not found!")
        return

    pipeline = DataPipeline()
    engineer = FeatureEngineer()
    model = FraudModel()

    print("Attempting to load data...")
    df = pipeline.load_data()
    if df is not None:
        print(f"SUCCESS: Data loaded. Shape: {df.shape}")
        
        print("Preprocessing sample...")
        df_sample = df.head(1000)
        df_clean = pipeline.preprocess(df_sample)
        df_features = engineer.create_features(df_clean)
        X = engineer.select_features(df_features)
        X_scaled = engineer.fit_transform_scaler(X)
        
        print("Testing model training...")
        model.train(X_scaled)
        model.save_model()
        
        if os.path.exists(Config.MODEL_PATH):
            print(f"SUCCESS: Model saved at {Config.MODEL_PATH}")
        else:
            print("FAIL: Model file not created.")
    else:
        print("FAIL: pipeline.load_data() returned None")

if __name__ == "__main__":
    test_full_pipeline()
