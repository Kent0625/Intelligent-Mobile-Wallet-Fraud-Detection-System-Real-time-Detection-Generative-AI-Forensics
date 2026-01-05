from forensic_agent import ForensicAgent

def test_gemini():
    print("Testing Gemini Forensic Agent...")
    agent = ForensicAgent()
    
    # Mock a fraud transaction
    mock_tx = {
        'type': 'TRANSFER',
        'amount': 500000.0,
        'old_balance_org': 500000.0,
        'new_balance_orig': 0.0,
        'name_orig': 'C123',
        'name_dest': 'C456'
    }
    
    print("Sending request to Gemini...")
    analysis = agent.analyze_transaction(mock_tx, is_fraud_pred=1)
    
    print("\n--- Gemini Analysis ---")
    print(analysis)
    print("------------------------")

if __name__ == "__main__":
    test_gemini()
