# Intelligent Mobile Wallet Fraud Detection System

A real-time fraud detection pipeline combining Machine Learning with Generative AI forensics for automated transaction analysis.

![Dashboard Preview](dashboard_preview.png)

## Overview
This system processes mobile money transactions to detect fraudulent activity in real-time. It uses a supervised learning model to flag suspicious transactions and a Large Language Model (LLM) to provide an explainable forensic report for analysts.

## Key Features
*   **Real-time Inference:** Low-latency detection using a specialized Random Forest model.
*   **Generative AI Forensics:** Automated narrative analysis of fraud using Hugging Face's Inference API (Qwen 2.5).
*   **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations) values visualize specific feature contributions for every decision.
*   **Production Pipeline:** Robust stateful preprocessing using `OneHotEncoder` and `StandardScaler` to handle live data streams.

## Technical Architecture
*   **Model:** Random Forest Classifier (Scikit-learn) trained on the PaySim dataset.
*   **LLM Integration:** `huggingface_hub` InferenceClient (Qwen/Qwen2.5-7B-Instruct).
*   **Frontend:** Streamlit for live monitoring and alert visualization.
*   **Evaluation:** Optimized for AUPRC (Area Under Precision-Recall Curve) on highly imbalanced data (1:1000).

## Quick Start

### 1. Installation
```bash
git clone https://github.com/Kent0625/Intelligent-Mobile-Wallet-Fraud-Detection-System-Real-time-Detection-Generative-AI-Forensics.git
cd Intelligent-Mobile-Wallet-Fraud-Detection-System-Real-time-Detection-Generative-AI-Forensics
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file with your Hugging Face token (required for AI analysis):
```bash
HF_TOKEN=your_token_here
```

### 3. Run Application
```bash
streamlit run src/dashboard.py
```

## Project Structure
*   `src/`: Source code for data pipelines, feature engineering, and the dashboard.
*   `models/`: Serialized model artifacts (`.pkl`).
*   `tests/`: Unit tests for schema validation and pipeline integrity.
