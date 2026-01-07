import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, average_precision_score, precision_recall_curve
from config import Config
from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer
from fraud_model import FraudModel

def evaluate():
    print("Loading data...")
    pipeline = DataPipeline()
    engineer = FeatureEngineer()
    
    # Load all data
    df = pipeline.load_data()
    if df is None:
        return

    print("Preprocessing and Feature Engineering...")
    df_clean = pipeline.preprocess(df)
    df_features = engineer.create_features(df_clean)
    
    # Select features (keeps 'type' and creates 'hour_of_day' later in transform)
    X = engineer.select_features(df_features)
    y = df_features['is_fraud']

    # 1. Split into Train and Test (Hold-out) - Stratified to maintain ratio
    # We use a large test set to capture rare fraud events
    print("Splitting data (Train: 70%, Test: 30%)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"Train set shape: {X_train_raw.shape}")
    print(f"Test set shape: {X_test_raw.shape}")

    # 2. Prepare Training Data (Downsampling Majority Class)
    # We train on balanced data to help the model learn patterns
    print("Balancing training data...")
    train_data = pd.concat([X_train_raw, y_train], axis=1)
    frauds = train_data[train_data['is_fraud'] == 1]
    non_frauds = train_data[train_data['is_fraud'] == 0]
    
    # Downsample non-frauds to match frauds (or a multiple, e.g. 1:1 or 1:10)
    # 1:1 balance is standard for initial training
    non_frauds_downsampled = non_frauds.sample(n=len(frauds), random_state=42)
    
    train_balanced = pd.concat([frauds, non_frauds_downsampled]).sample(frac=1, random_state=42)
    
    X_train_balanced = train_balanced.drop('is_fraud', axis=1)
    y_train_balanced = train_balanced['is_fraud']
    
    print(f"Balanced Training Data: {X_train_balanced.shape} (Fraud: {y_train_balanced.sum()})")

    # 3. Fit Scaler/Encoder on RAW Training Data (The "True" Distribution)
    print("Fitting Scaler/Encoder on True Distribution...")
    engineer.fit(X_train_raw)  # Fit on Imbalanced

    # 4. Transform the Balanced Data for the Model
    print("Transforming Balanced Training Data...")
    X_train_processed = engineer.transform(X_train_balanced) # Transform Balanced
    
    # 5. Transform Test Data (Imbalanced - Real world scenario)
    print("Transforming Test Data...")
    X_test_processed = engineer.transform(X_test_raw)

    print("Training Model (Random Forest)...")
    model = FraudModel()
    model.train(X_train_processed, y_train_balanced)
    
    # Save the trained model and engineer state
    model.save_model()

    print("Evaluating on Imbalanced Test Set (Real-world simulation)...")
    # Predict
    y_pred = model.predict(X_test_processed)
    # Predict Proba for AUPRC
    y_prob = model.model.predict_proba(X_test_processed)[:, 1]

    # Metrics
    print("\n--- Model Performance Report (Test Set) ---")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # AUPRC
    auprc = average_precision_score(y_test, y_prob)
    print(f"\nArea Under Precision-Recall Curve (AUPRC): {auprc:.4f}")

    # Save metrics to file
    with open("model_evaluation.txt", "w") as f:
        f.write("Model Performance Report (Imbalanced Test Set)\n")
        f.write(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))
        f.write("\nConfusion Matrix\n")
        f.write(str(cm))
        f.write(f"\nAUPRC: {auprc:.4f}\n")
    
    print("\nEvaluation complete. Results saved to model_evaluation.txt")

if __name__ == "__main__":
    evaluate()
