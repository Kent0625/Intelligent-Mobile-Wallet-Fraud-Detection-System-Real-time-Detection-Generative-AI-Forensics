from huggingface_hub import InferenceClient
from config import Config

class ForensicAgent:
    def __init__(self):
        # Use Hugging Face Inference API (Free Tier)
        # Qwen 2.5 is robust and widely supported on the free tier
        self.model_id = "Qwen/Qwen2.5-7B-Instruct" 
        
        if Config.HF_TOKEN:
            self.client = InferenceClient(model=self.model_id, token=Config.HF_TOKEN)
        else:
            print("Warning: HF_TOKEN not found. GenAI features will be disabled.")
            self.client = None

    def analyze_transaction(self, transaction_data, is_fraud_pred, feature_context=None):
        """
        Generates a forensic report using HF Chat Completion API (Conversational).
        """
        if not self.client:
            return "GenAI Analysis Unavailable (Missing HF_TOKEN)."

        status = "SUSPICIOUS" if is_fraud_pred == 1 else "NORMAL"
        
        # Format context
        context_str = ""
        if feature_context:
            context_str = "\nKey Risk Indicators:\n"
            for k, v in feature_context.items():
                context_str += f"- {k}: {v}\n"
        
        # Prompt content
        user_content = f"""You are a financial fraud forensics expert. 
Analyze this transaction flagged as {status}.

Transaction Details:
{transaction_data}
{context_str}

Task:
1. Explain why it is {status} (e.g. theft if error_balance_orig < 0).
2. Recommend action (Block/Call/Ignore).
3. Be extremely concise (max 50 words)."""

        messages = [
            {"role": "user", "content": user_content}
        ]

        try:
            # Use chat_completion for Instruct/Chat models
            response = self.client.chat_completion(
                messages,
                max_tokens=150,
                temperature=0.5
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Analysis Error: {str(e)}"
