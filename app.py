import os
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

# -----------------------
# App Configuration
# -----------------------
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

st.title("💳 Credit Card Fraud Detection")
st.write("This app uses a trained model to detect fraudulent transactions.")

# -----------------------
# Load Model
# -----------------------
MODEL_PATH = "model.h5"
try:
    model = load_model(MODEL_PATH)
    st.sidebar.success("✅ Model loaded successfully")
except Exception as e:
    st.sidebar.error("❌ Could not load model. Please ensure model.h5 is in the repo.")
    st.stop()

# -----------------------
# Load Dataset
# -----------------------
DATA_PATHS = ["data/creditcard.csv", "creditcard.csv"]

df = None
for path in DATA_PATHS:
    if os.path.exists(path):
        df = pd.read_csv(path)
        break

if df is None:
    st.sidebar.warning("⚠️ Sample dataset not found. Please upload it below.")
    uploaded_file = st.file_uploader("Upload creditcard.csv", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset uploaded successfully!")
else:
    st.sidebar.success("✅ Sample dataset loaded")

# -----------------------
# Feature Names
# -----------------------
FEATURES = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

# -----------------------
# Prediction Function
# -----------------------
def predict_transaction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array, verbose=0)[0][0]
    return prediction

# -----------------------
# Input Section
# -----------------------
st.header("🔍 Enter Transaction Details")

user_input = []
cols = st.columns(3)  # Split UI into 3 columns
for i, feature in enumerate(FEATURES):
    with cols[i % 3]:
        value = st.number_input(f"{feature}", value=0.0, format="%.5f")
        user_input.append(value)

if st.button("Predict Fraud"):
    prob = predict_transaction(user_input)
    if prob > 0.5:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Probability: {prob:.2f})")

# -----------------------
# Example Transactions
# -----------------------
st.header("📊 Example Transactions")
if df is not None:
    fraud_samples = df[df["Class"] == 1].sample(n=2, random_state=42)
    legit_samples = df[df["Class"] == 0].sample(n=2, random_state=42)

    st.subheader("Fraudulent Transactions")
    st.dataframe(fraud_samples[FEATURES + ["Class"]])

    st.subheader("Legitimate Transactions")
    st.dataframe(legit_samples[FEATURES + ["Class"]])
else:
    st.info("Upload `creditcard.csv` to view example transactions.")
