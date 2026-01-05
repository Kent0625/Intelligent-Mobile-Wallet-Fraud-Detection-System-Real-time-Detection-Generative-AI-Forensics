from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

def list_models():
    client = genai.Client(api_key=api_key)
    print("Available models:")
    for m in client.models.list():
        print(f"Name: {m.name}, ID: {m.name.split('/')[-1]}")

if __name__ == "__main__":
    list_models()
