from google import genai
from config import Config

class ForensicAgent:
    def __init__(self):
        if Config.GEMINI_API_KEY:
            self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
            self.model_id = "gemini-2.0-flash"
        else:
            print("Warning: GEMINI_API_KEY not found. GenAI features will be disabled.")
            self.client = None

    def analyze_transaction(self, transaction_data, is_fraud_pred):
        """
        Generates a forensic report for a transaction.
        """
        if not self.client:
            return "GenAI Analysis Unavailable (Missing API Key)."

        status = "SUSPICIOUS" if is_fraud_pred == 1 else "NORMAL"
        
        prompt = f"""
        You are a financial fraud forensics expert. 
        Analyze the following mobile wallet transaction which has been flagged as {status} by our ML system.

        Transaction Details:
        {transaction_data}

        Task:
        1. Explain why this transaction might be considered {status} based on the values (e.g. huge amount, zero balance change, etc.).
        2. Provide a recommendation for the fraud analyst (e.g. "Call customer", "Block account", "Ignore").
        3. Keep it concise (under 100 words).
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                return "⚠️ Analysis Skipped: API Rate Limit Reached. (Free tier quota exceeded)"
            return f"Error generating analysis: {e}"