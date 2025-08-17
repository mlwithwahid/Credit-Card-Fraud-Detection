import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("model.h5")

# Feature names (30 inputs)
FEATURES = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18","V19",
    "V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

st.title("💳 Credit Card Fraud Detection")
st.write("Predict whether a transaction is **Fraudulent** or **Non-Fraudulent** using a deep learning model.")

# --- Input method selection ---
st.sidebar.header("Choose Input Method")
input_method = st.sidebar.radio("Select how to provide input:", ["Manual Input", "Sample Examples"])

# --- Manual Input Section ---
if input_method == "Manual Input":
    st.header("Enter Transaction Details")

    user_data = []
    for feature in FEATURES:
        value = st.number_input(f"{feature}", value=0.0, format="%.5f")
        user_data.append(value)

    input_array = np.array([user_data])

    if st.button("🔍 Predict"):
        prediction = model.predict(input_array)[0][0]
        label = "🚨 Fraudulent" if prediction > 0.5 else "✅ Non-Fraudulent"
        st.success(f"Prediction: {label} (score: {prediction:.4f})")

# --- Sample Example Section ---
else:
    st.header("Use Example Transactions")

    try:
        fraud_df = pd.read_csv("data/fraud_examples.csv")
        nonfraud_df = pd.read_csv("data/nonfraud_examples.csv")

        example_type = st.radio("Choose example type:", ["Fraud", "Non-Fraud"])
        if example_type == "Fraud":
            sample = fraud_df.sample(1).iloc[0]
        else:
            sample = nonfraud_df.sample(1).iloc[0]

        st.write("### Selected Example")
        st.dataframe(pd.DataFrame([sample]))

        input_array = np.array([sample.values])

        if st.button("🔍 Predict Example"):
            prediction = model.predict(input_array)[0][0]
            label = "🚨 Fraudulent" if prediction > 0.5 else "✅ Non-Fraudulent"
            st.success(f"Prediction: {label} (score: {prediction:.4f})")

    except FileNotFoundError:
        st.error("⚠️ Please upload `fraud_examples.csv` and `nonfraud_examples.csv` inside a `data/` folder in your repo.")

st.sidebar.info("ℹ️ Tip: Use `Manual Input` for custom transactions or `Sample Examples` to quickly test with real data.")
