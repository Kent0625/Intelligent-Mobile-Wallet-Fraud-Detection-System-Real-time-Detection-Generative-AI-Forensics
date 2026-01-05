import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from config import Config
from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer
from fraud_model import FraudModel

def evaluate():
    print("Loading data...")
    pipeline = DataPipeline()
    engineer = FeatureEngineer()
    
    # Load all data (or a large chunk)
    df = pipeline.load_data()
    if df is None:
        return

    # Use a larger sample for meaningful evaluation, but keep it manageable
    print("Sampling data for evaluation (200,000 records)...")
    # Ensure we include frauds in the sample for valid testing
    frauds = df[df['isFraud'] == 1]
    non_frauds = df[df['isFraud'] == 0].sample(n=200000, random_state=42)
    df_sample = pd.concat([frauds, non_frauds]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Sample shape: {df_sample.shape}")
    print(f"Fraud count in sample: {df_sample['isFraud'].sum()}")

    print("Preprocessing and Feature Engineering...")
    df_clean = pipeline.preprocess(df_sample)
    df_features = engineer.create_features(df_clean)
    X = engineer.select_features(df_features)
    y = df_sample['isFraud']

    # Split into Train and Validation sets
    # Note: Isolation Forest is unsupervised, so we train on X_train (often mostly normal data)
    # But here we just split to evaluate on unseen data.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print("Scaling features...")
    # Fit scaler on train, transform both
    X_train_scaled = engineer.fit_transform_scaler(X_train)
    X_val_scaled = engineer.transform_scaler(X_val)

    print("Training Model (Random Forest)...")
    model = FraudModel()
    model.train(X_train_scaled, y_train)

    print("Evaluating on Validation Set...")
    # Predict
    y_pred = model.predict(X_val_scaled)

    # Metrics
    print("\n--- Model Performance Report ---")
    print(classification_report(y_val, y_pred, target_names=['Legit', 'Fraud']))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)

    # Save metrics to file
    with open("model_evaluation.txt", "w") as f:
        f.write("Model Performance Report\n")
        f.write(classification_report(y_val, y_pred, target_names=['Legit', 'Fraud']))
        f.write("\nConfusion Matrix\n")
        f.write(str(cm))
    
    print("\nEvaluation complete. Results saved to model_evaluation.txt")

if __name__ == "__main__":
    evaluate()
