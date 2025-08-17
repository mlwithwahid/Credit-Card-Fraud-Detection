import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

# Load model & scaler
# ---------------- Load or Train Model ----------------
st.subheader("🤖 Model Status")

import pickle
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "model.h5"

def demo_dataset():
    data = {
        "V1": [0.1, -1.2, 1.5, -0.3, 0.7, -2.1],
        "V2": [1.3, -0.4, 0.7, 2.1, -1.5, 0.9],
        "V3": [-0.2, 0.8, -1.5, 0.6, 1.1, -0.7],
        "Amount": [50.0, 200.0, 500.0, 1200.0, 300.0, 750.0],
        "Class": [0, 0, 1, 1, 0, 1]
    }
    return pd.DataFrame(data)

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        st.success("✅ Model loaded successfully.")
    except:
        st.error("⚠️ Error loading model. Re-training fallback model...")
        df_demo = demo_dataset()
        X, y = df_demo.drop("Class", axis=1), df_demo["Class"]
        model = RandomForestClassifier().fit(X, y)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        st.success("✅ Fallback model trained & saved.")
else:
    st.warning("⚠️ Model not found. Training fallback model...")
    df_demo = demo_dataset()
    X, y = df_demo.drop("Class", axis=1), df_demo["Class"]
    model = RandomForestClassifier().fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    st.success("✅ Fallback model trained & saved.")

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
# ---------------- Example Transactions ----------------
st.subheader("📊 Example Transactions")

def demo_examples():
    data = {
        "V1": [0.1, -1.2, 1.5, -0.3],
        "V2": [1.3, -0.4, 0.7, 2.1],
        "V3": [-0.2, 0.8, -1.5, 0.6],
        "Amount": [50.0, 200.0, 500.0, 1200.0],
        "Class": [0, 0, 1, 1]  # 0 = Legit, 1 = Fraud
    }
    return pd.DataFrame(data)

if os.path.exists("data/creditcard.csv"):
    try:
        df = pd.read_csv("data/creditcard.csv")
        legit_samples = df[df["Class"] == 0].sample(n=2, random_state=42)
        fraud_samples = df[df["Class"] == 1].sample(n=2, random_state=42)
    except:
        df = demo_examples()
        legit_samples = df[df["Class"] == 0]
        fraud_samples = df[df["Class"] == 1]
else:
    df = demo_examples()
    legit_samples = df[df["Class"] == 0]
    fraud_samples = df[df["Class"] == 1]

st.write("✅ Legitimate Transactions")
st.dataframe(legit_samples)

st.write("⚠️ Fraudulent Transactions")
st.dataframe(fraud_samples)
