import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

# Load model & scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/fraud_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return model, scaler
    except:
        return None, None

model, scaler = load_model()

st.title("💳 Credit Card Fraud Detection")
st.write("Upload a transaction file (CSV) or enter details manually to check for fraud.")

# ---------------- User Upload ----------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "Class" in df.columns:
        X = df.drop("Class", axis=1)
        y = df["Class"]
    else:
        X = df
        y = None

    if scaler:
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        df["Prediction"] = preds

        st.subheader("🔎 Predictions")
        st.dataframe(df.head())

        if y is not None:
            st.subheader("✅ Accuracy on uploaded file")
            acc = (preds == y).mean()
            st.write(f"Accuracy: {acc:.2%}")
    else:
        st.error("⚠️ Model not found. Please train the model first.")

# ---------------- Manual Input ----------------
st.subheader("Or Enter Transaction Details")
feature_count = 30  # Adjust based on dataset
inputs = []

for i in range(feature_count):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(val)

if st.button("Predict Transaction"):
    X_manual = np.array(inputs).reshape(1, -1)
    if scaler and model:
        X_manual_scaled = scaler.transform(X_manual)
        pred = model.predict(X_manual_scaled)[0]
        st.success("⚠️ Fraudulent Transaction!" if pred == 1 else "✅ Legitimate Transaction")
    else:
        st.error("⚠️ Model not loaded.")

# ---------------- Example Transactions ----------------
st.subheader("📊 Example Transactions")

if os.path.exists("data/creditcard.csv"):
    try:
        df = pd.read_csv("data/creditcard.csv")
        legit_samples = df[df["Class"] == 0].sample(n=2, random_state=42)
        fraud_samples = df[df["Class"] == 1].sample(n=2, random_state=42)

        st.write("✅ Legitimate Transactions")
        st.dataframe(legit_samples)

        st.write("⚠️ Fraudulent Transactions")
        st.dataframe(fraud_samples)
    except Exception as e:
        st.warning("⚠️ Could not load example dataset. Please check the file.")
else:
    st.info("ℹ️ Example dataset not found. Please upload `data/creditcard.csv` in GitHub if you want to display examples.")
