import os
from dotenv import load_dotenv

load_dotenv()

# Get the absolute path to the project root (one level up from src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "PS_20174392719_1491204439457_log.csv")
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fraud_detector.pkl")
    SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    RANDOM_STATE = 42
    CONTAMINATION = 0.002 # Approximate fraud rate in dataset
