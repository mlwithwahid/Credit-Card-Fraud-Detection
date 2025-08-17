import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import tensorflow as tf

# ---------------------------
# Load Model and Scaler
# ---------------------------
MODEL_PATH = "models/fraud_model.h5"
SCALER_PATH = "models/scaler.pkl"

model, scaler = None, None

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        st.warning("⚠️ Model not found. Please train and save the model first.")

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        st.warning("⚠️ Scaler not found. Please ensure scaler.pkl is saved.")
except Exception as e:
    st.error(f"⚠️ Error loading model/scaler: {e}")

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("💳 Credit Card Fraud Detection")
st.write("Predict fraudulent transactions using a trained TensorFlow model.")

# Example dataset for testing
EXAMPLE_PATH = "data/creditcard.csv"
df_example = None
if os.path.exists(EXAMPLE_PATH):
    df_example = pd.read_csv(EXAMPLE_PATH)
    st.success("✅ Example dataset loaded successfully.")
else:
    st.warning("⚠️ Example dataset not found. Upload data/creditcard.csv if needed.")

# ---------------------------
# Input Section
# ---------------------------
st.subheader("🔹 Enter Transaction Details")

uploaded_file = st.file_uploader("Upload a CSV file (same format as training data)", type=["csv"])

input_data = None

if uploaded_file:
    try:
        df_input = pd.read_csv(uploaded_file)

        # Drop columns not used in training
        if "Class" in df_input.columns:
            df_input = df_input.drop("Class", axis=1)
        if "Time" in df_input.columns:
            df_input = df_input.drop("Time", axis=1)

        input_data = df_input
        st.success("✅ File uploaded and processed successfully.")

    except Exception as e:
        st.error(f"⚠️ Error reading uploaded file: {e}")

else:
    # Manual entry for testing (first row from dataset if available)
    if df_example is not None:
        sample = df_example.drop(["Time", "Class"], axis=1).iloc[0].to_dict()
        input_values = {}
        for col, val in sample.items():
            input_values[col] = st.number_input(f"{col}", value=float(val))
        input_data = pd.DataFrame([input_values])

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔍 Predict Fraud"):
    if model is None or scaler is None:
        st.error("⚠️ Model/Scaler not loaded. Cannot make predictions.")
    elif input_data is None:
        st.warning("⚠️ Please upload data or enter values manually.")
    else:
        try:
            # Scale input
            input_scaled = scaler.transform(input_data)

            # Predict
            prediction = model.predict(input_scaled)[0][0]

            if prediction > 0.5:
                st.error(f"🚨 Fraudulent Transaction Detected (score: {prediction:.4f})")
            else:
                st.success(f"✅ Legitimate Transaction (score: {prediction:.4f})")

        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")
