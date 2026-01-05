import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer
from fraud_model import FraudModel
import os

def audit():
    print("Loading data for audit...")
    pipeline = DataPipeline()
    engineer = FeatureEngineer()
    
    # Load sample to save time, but large enough to be representative
    df = pipeline.load_data()
    if df is None: return
    
    # We use a balanced-ish subsample for visualization to make charts readable
    # But calculate stats on the full set (or large chunk)
    print("Creating audit dataset...")
    frauds = df[df['isFraud'] == 1]
    non_frauds = df[df['isFraud'] == 0].sample(n=100000, random_state=42)
    df_audit = pd.concat([frauds, non_frauds])
    
    print(f"Audit Dataset: {df_audit.shape}")
    print(f"Fraud Rate: {df_audit['isFraud'].mean():.4f}")

    # 1. Feature Engineering (to see what the model sees)
    df_clean = pipeline.preprocess(df_audit)
    df_features = engineer.create_features(df_clean)
    
    # 2. Correlation Check
    print("\n--- Correlation with Target (is_fraud) ---")
    corr = df_features.select_dtypes(include=[np.number]).corr()['is_fraud'].sort_values(ascending=False)
    print(corr)
    
    # 3. Model Feature Importance
    print("\n--- Feature Importance Analysis ---")
    model = FraudModel()
    if os.path.exists(Config.MODEL_PATH):
        model.load_model()
        if hasattr(model.model, 'feature_importances_'):
            # Get feature names
            # We must recreate the exact features used in training
            # In training, we dropped columns and then One-Hot Encoded
            # The pipeline does get_dummies. We need to match that.
            
            # Use the scaler's feature_names_in_ if available (sklearn > 1.0)
            if hasattr(engineer.scaler, 'feature_names_in_'):
                feature_names = engineer.scaler.feature_names_in_
            else:
                # Fallback: recreate X to get columns
                X = engineer.select_features(df_features)
                feature_names = X.columns
            
            importances = model.model.feature_importances_
            
            feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feat_imp = feat_imp.sort_values(by='importance', ascending=False)
            print(feat_imp)
            
            # Save plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feat_imp)
            plt.title('Random Forest Feature Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            print("Saved feature_importance.png")
        else:
            print("Model does not support feature importance (might not be fitted or wrong type).")
    else:
        print("Model file not found. Please train the model first.")

    # 4. Outlier / Distribution Check (Boxplots)
    print("\n--- Distribution Check (Top Features) ---")
    # Let's check the top 3 features from our intuition or correlation
    top_features = ['amount', 'error_balance_orig', 'old_balance_org']
    
    for feat in top_features:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x='is_fraud', y=feat, data=df_features)
        plt.title(f'Distribution of {feat} by Class')
        plt.savefig(f'dist_{feat}.png')
        print(f"Saved dist_{feat}.png")
        
        # Print summary stats
        print(f"\nStats for {feat}:")
        print(df_features.groupby('is_fraud')[feat].describe())

if __name__ == "__main__":
    audit()
