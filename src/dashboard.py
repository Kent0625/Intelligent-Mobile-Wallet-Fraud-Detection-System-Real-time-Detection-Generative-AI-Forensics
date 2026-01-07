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

# Custom CSS for extra stability
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.5rem !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Intelligent Mobile Wallet Fraud Detection System")
st.markdown("### Real-time Detection & Generative AI Forensics")

# Sidebar
st.sidebar.header("Control Panel")
hf_token = st.sidebar.text_input("Hugging Face Token", type="password", value=os.getenv("HF_TOKEN") or "")
if hf_token:
    Config.HF_TOKEN = hf_token
    os.environ["HF_TOKEN"] = hf_token

# Initialize components
@st.cache_resource
def load_components():
    pipeline = DataPipeline()
    engineer = FeatureEngineer()
    model = FraudModel()
    explainer = None
    
    if os.path.exists(Config.MODEL_PATH):
        try:
            model.load_model()
            if engineer.load_state():
                print("Feature Engineer state loaded.")
            else:
                print("Warning: Feature Engineer state not found.")
            explainer = shap.TreeExplainer(model.model)
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
                df_sample = df.sample(n=50000, random_state=42) 
                df_clean = pipeline.preprocess(df_sample)
                df_features = engineer.create_features(df_clean)
                y = df_features['is_fraud']
                X = engineer.select_features(df_features)
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
        
        # 1. FIXED TOP METRICS
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1: total_metric = st.empty()
        with m_col2: fraud_metric = st.empty()
        with m_col3: status_metric = st.empty()
        
        st.divider()
        
        col_monitor, col_log = st.columns([0.6, 0.4])
        
        # 2. FIXED HEIGHT SECURITY MONITOR (Left Column)
        with col_monitor:
            st.markdown("### 🔍 Security Monitor")
            # This container has a HARD fixed height. Content inside scrolls if needed.
            # Page layout will NEVER shift because this box size is immutable.
            alert_container = st.container(height=500)
            
        # 3. FIXED HEIGHT LOG (Right Column)
        with col_log:
            st.markdown("### 📜 Log")
            log_container = st.container(height=500)
            log_table = log_container.empty()
        
        # Stream logic
        streamer = TransactionStream()
        
        # Load data (Real or Synthetic Fallback)
        full_df = pipeline.load_data()
        
        if full_df is not None:
            # Ensure we have both classes for a good demo mix
            if 'isFraud' in full_df.columns:
                 # Raw data (synthetic or real) has 'isFraud'
                 fraud_col = 'isFraud'
            elif 'is_fraud' in full_df.columns:
                 # Preprocessed might have 'is_fraud'
                 fraud_col = 'is_fraud'
            else:
                 fraud_col = None

            if fraud_col:
                frauds = full_df[full_df[fraud_col] == 1].head(10)
                normals = full_df[full_df[fraud_col] == 0].head(90)
                
                # If synthetic generator didn't produce enough of one class, just use what we have
                if len(frauds) < 10:
                    frauds = full_df[full_df[fraud_col] == 1]
                
                demo_df = pd.concat([frauds, normals]).sample(frac=1).reset_index(drop=True)
                streamer.df = demo_df
            else:
                st.warning("Data schema mismatch. Streaming raw data.")
                streamer.df = full_df.head(100)
        else:
            st.error("Could not generate or load data.")
            st.stop()
        
        total_count = 0
        fraud_count = 0
        rows = []
        
        # Initial Safe State
        with alert_container:
            st.success("✅ System Secure. Monitoring active...")
            st.markdown("Awaiting transaction stream...")
        
        for tx in streamer.stream():
            total_count += 1
            
            # Processing
            tx_df = pd.DataFrame([tx])
            tx_clean = pipeline.preprocess(tx_df)
            tx_features = engineer.create_features(tx_clean)
            X_input = engineer.select_features(tx_features)
            X_scaled = engineer.transform(X_input)
            pred = model.predict(X_scaled)[0]
            
            status = "🔴 FRAUD" if pred == 1 else "🟢 LEGIT"
            
            # Update Metrics
            total_metric.metric("Transactions", total_count)
            fraud_metric.metric("Frauds Detected", fraud_count)
            status_metric.info(f"Processing Step: {tx['step']}")

            # Update Log
            rows.insert(0, {"Step": tx['step'], "Type": tx['type'], "Amt": tx['amount'], "Status": status})
            log_table.dataframe(pd.DataFrame(rows).head(15), use_container_width=True)
            
            if pred == 1:
                fraud_count += 1
                
                context = {
                    "error_orig": round(tx_features['error_balance_orig'].iloc[0], 2),
                    "error_dest": round(tx_features['error_balance_dest'].iloc[0], 2),
                    "amount": tx['amount']
                }
                
                # RENDER ALERT INSIDE FIXED CONTAINER
                # We clear the container content first to remove the "Safe" message
                alert_container.empty()
                with alert_container:
                    st.error(f"🚨 FRAUD DETECTED: Transaction {tx['step']}")
                    
                    st.markdown("#### 🤖 AI Forensic Analysis")
                    with st.spinner("Consulting AI..."):
                        analysis = agent.analyze_transaction(tx, pred, feature_context=context)
                    st.info(analysis)
                    
                    st.markdown("#### 📊 Root Cause (SHAP)")
                    if explainer:
                        try:
                            shap_values = explainer(X_scaled)
                            if len(shap_values.shape) == 3:
                                explanation = shap_values[0, :, 1]
                            else:
                                explanation = shap_values[0]
                            
                            fig, ax = plt.subplots(figsize=(6,3), dpi=100)
                            shap.plots.waterfall(explanation, show=False, max_display=5)
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                        except:
                            st.warning("SHAP Visualization Unavailable")
                
                time.sleep(5.0)
                
                # Reset to Safe State
                alert_container.empty()
                with alert_container:
                    st.success(f"✅ Threat Mitigated. Resuming scan...")
                    st.markdown(f"Last Alert: Step {tx['step']}")

            else:
                # Optional: Don't update alert container on every safe frame to save rendering
                # Just sleep fast
                time.sleep(0.1)

else:
    st.info("Please train the model to start simulation.")