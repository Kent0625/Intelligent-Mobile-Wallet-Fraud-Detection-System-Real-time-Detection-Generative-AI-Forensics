import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from config import Config

class FraudModel:
    def __init__(self):
        # Switch to Random Forest for better performance on this dataset
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=Config.RANDOM_STATE, 
            n_jobs=-1,
            class_weight='balanced' # Important for imbalanced data
        )

    def train(self, X_train, y_train):
        """Trains the Random Forest model."""
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def predict(self, X):
        """Predicts fraud. Returns 1 for fraud, 0 for normal."""
        return self.model.predict(X)

    def save_model(self):
        """Saves the model to disk."""
        joblib.dump(self.model, Config.MODEL_PATH)
        print(f"Model saved to {Config.MODEL_PATH}")

    def load_model(self):
        """Loads the model from disk."""
        self.model = joblib.load(Config.MODEL_PATH)
        print("Model loaded.")
