import numpy as np
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model

# Load model
model = load_model("model.h5")

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter transaction details below to check if it’s **Fraudulent 🚨 or Safe ✅**")

# --- User Input ---
feature_names = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18","V19",
    "V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

user_data = []
with st.form("fraud_form"):
    cols = st.columns(3)
    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            val = st.number_input(f"{feature}", value=0.0, format="%.4f")
            user_data.append(val)
    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    try:
        # Convert to numpy
        input_array = np.array([user_data], dtype=np.float32)

        # --- Automatic reshaping based on model input ---
        expected_shape = model.input_shape
        st.info(f"Model expects input shape: {expected_shape}")

        # If model expects (None, 30, 1) → add extra dimension
        if len(expected_shape) == 3:
            input_array = input_array.reshape((1, expected_shape[1], expected_shape[2]))
        elif len(expected_shape) == 2:
            input_array = input_array.reshape((1, expected_shape[1]))
        
        # Predict
        prediction = model.predict(input_array)
        prediction_value = float(prediction.flatten()[0])

        # Threshold
        label = "🚨 Fraudulent Transaction" if prediction_value > 0.5 else "✅ Safe Transaction"

        # --- UI Display ---
        st.subheader("🔎 Prediction Result")
        if prediction_value > 0.5:
            st.error(f"**{label}** (Confidence: {prediction_value:.4f})")
        else:
            st.success(f"**{label}** (Confidence: {prediction_value:.4f})")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# --- Example Section ---
st.markdown("---")
st.subheader("📊 Example Transactions")

if st.checkbox("Show sample fraud & non-fraud examples"):
    try:
        df = pd.read_csv("data/creditcard.csv")
        sample_nonfraud = df[df["Class"] == 0].sample(3)
        sample_fraud = df[df["Class"] == 1].sample(3)
        
        st.write("✅ **Non-Fraudulent Transactions**")
        st.dataframe(sample_nonfraud[feature_names + ["Class"]])
        
        st.write("🚨 **Fraudulent Transactions**")
        st.dataframe(sample_fraud[feature_names + ["Class"]])
    except:
        st.warning("Sample dataset not found. Please ensure `data/creditcard.csv` is uploaded.")
