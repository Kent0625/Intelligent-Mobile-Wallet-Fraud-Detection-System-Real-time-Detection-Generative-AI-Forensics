import streamlit as st
import pandas as pd
import time
import os
import shap
import matplotlib.pyplot as plt
from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer
from fraud_model import FraudModel
from forensic_agent import ForensicAgent
from realtime_simulator import TransactionStream
from config import Config

st.set_page_config(page_title="Intelligent Fraud Detection", layout="wide")

st.title("🛡️ Intelligent Mobile Wallet Fraud Detection System")
st.markdown("### Real-time Detection & Generative AI Forensics")

# Sidebar
st.sidebar.header("Control Panel")
api_key = st.sidebar.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY") or "")
if api_key:
    Config.GEMINI_API_KEY = api_key
    os.environ["GEMINI_API_KEY"] = api_key

# Initialize components
@st.cache_resource
def load_components():
    pipeline = DataPipeline()
    engineer = FeatureEngineer()
    model = FraudModel()
    explainer = None
    
    # Try to load existing model
    if os.path.exists(Config.MODEL_PATH):
        try:
            model.load_model()
            if engineer.load_state():
                print("Feature Engineer state loaded.")
            else:
                print("Warning: Feature Engineer state not found.")
            
            # Initialize SHAP Explainer
            # We use a background dataset for reference if possible, but for TreeExplainer on RF it's self-contained
            # Using a small background sample can speed it up but TreeExplainer handles RF well.
            explainer = shap.TreeExplainer(model.model)
            print("SHAP Explainer initialized.")
        except Exception as e:
            print(f"Could not load model/explainer: {e}")
            
    agent = ForensicAgent()
    return pipeline, engineer, model, agent, explainer

pipeline, engineer, model, agent, explainer = load_components()

# Training Section
if not os.path.exists(Config.MODEL_PATH):
    st.warning("⚠️ Model not found.")
    if st.button("Train Model (This may take a minute)"):
        with st.spinner("Loading and Preprocessing Data..."):
            df = pipeline.load_data()
            if df is not None:
                # Use a subset for speed in demo but ensure it's balanced-ish or large enough
                df_sample = df.sample(n=50000, random_state=42) 
                df_clean = pipeline.preprocess(df_sample)
                df_features = engineer.create_features(df_clean)
                # Extract labels for Supervised Learning
                y = df_features['is_fraud']
                X = engineer.select_features(df_features)
                
                # FIT and Transform
                X_scaled = engineer.fit_transform(X)
                
                with st.spinner("Training Random Forest..."):
                    model.train(X_scaled, y)
                    model.save_model()
                st.success("Model trained and saved! Please reload the app.")
            else:
                st.error("Could not load data. Check path.")

# Simulation Section
if os.path.exists(Config.MODEL_PATH):
    st.sidebar.success("Model Status: Ready ✅")
    
    if st.button("Start Live Simulation"):
        st.subheader("Live Transaction Stream")
        
        # Metrics placeholders
        col1, col2, col3 = st.columns(3)
        with col1:
            total_metric = st.empty()
        with col2:
            fraud_metric = st.empty()
        with col3:
            alert_metric = st.empty()
            
        transaction_log = st.empty()
        
        # Stream logic
        streamer = TransactionStream()
        # Load a mix of fraud and non-fraud for demo
        try:
            full_df = pd.read_csv(Config.DATA_PATH, nrows=10000) # Read chunk
            # Hack to ensure we see some frauds in the stream
            frauds = full_df[full_df['isFraud'] == 1].head(10)
            normals = full_df[full_df['isFraud'] == 0].head(90)
            demo_df = pd.concat([frauds, normals]).sample(frac=1).reset_index(drop=True)
            streamer.df = demo_df
        except:
            st.error("Data file not found for simulation.")
            st.stop()
        
        total_count = 0
        fraud_count = 0
        
        rows = []
        
        for tx in streamer.stream():
            total_count += 1
            
            # Preprocess single transaction
            tx_df = pd.DataFrame([tx])
            tx_clean = pipeline.preprocess(tx_df)
            tx_features = engineer.create_features(tx_clean)
            
            # Select features
            X_input = engineer.select_features(tx_features)
            
            # Transform (handling encoding/scaling safely)
            X_scaled = engineer.transform(X_input)
            
            # Predict
            pred = model.predict(X_scaled)[0]
            
            status = "🔴 FRAUD" if pred == 1 else "🟢 LEGIT"
            
            if pred == 1:
                fraud_count += 1
                
                # Context for GenAI
                context = {
                    "error_balance_orig": tx_features['error_balance_orig'].iloc[0],
                    "error_balance_dest": tx_features['error_balance_dest'].iloc[0],
                    "amount": tx['amount']
                }
                
                # Trigger Agent
                with st.expander(f"🚨 ALERT: Transaction {tx['step']} Detected!", expanded=True):
                    col_gen, col_shap = st.columns([1, 1])
                    
                    with col_gen:
                        st.markdown("### 🤖 GenAI Forensic Analysis")
                        analysis = agent.analyze_transaction(tx, pred, feature_context=context)
                        st.write(analysis)

                    with col_shap:
                        st.markdown("### 📊 SHAP Explainability")
                        if explainer:
                            try:
                                # Calculate SHAP values for this instance
                                # X_scaled is a DataFrame, so shap preserves feature names
                                shap_values = explainer(X_scaled)
                                
                                # Check if shap_values has multiple class outputs (common for classifiers)
                                # Shape is typically (n_samples, n_features, n_classes)
                                if len(shap_values.shape) == 3:
                                    # Select the explanation for the Fraud class (Index 1) for the first sample (Index 0)
                                    explanation = shap_values[0, :, 1]
                                else:
                                    # Binary case or regression where only one output is returned
                                    explanation = shap_values[0]

                                # Waterfall plot
                                fig, ax = plt.subplots()
                                shap.plots.waterfall(explanation, show=False)
                                st.pyplot(fig)
                                plt.close(fig)
                            except Exception as e:
                                st.error(f"Could not generate SHAP plot: {e}")
            
            # Update metrics
            total_metric.metric("Transactions Processed", total_count)
            fraud_metric.metric("Frauds Detected", fraud_count)
            
            # Update log
            rows.insert(0, {"Step": tx['step'], "Type": tx['type'], "Amount": tx['amount'], "Status": status})
            transaction_log.dataframe(pd.DataFrame(rows).head(10))
            
            # Slow down to prevent API rate limits (Free tier is ~15 req/min)
            time.sleep(2.0)

else:
    st.info("Please train the model to start simulation.")