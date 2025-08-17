import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow import keras

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="💳 Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection App")
st.markdown("Detect fraudulent transactions using your trained TensorFlow model.")

MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.pkl"

# Expected features (with Time)
EXPECTED_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# -----------------------------
# Helper Functions
# -----------------------------
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def align_features(df):
    # Ensure all expected features exist
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    # Reorder columns
    return df[EXPECTED_FEATURES]

def demo_dataset():
    data = {
        "Time": [1000, 2000, 3000, 4000, 5000, 6000],
        "V1": [0.1, -1.2, 1.5, -0.3, 0.7, -2.1],
        "V2": [1.3, -0.4, 0.7, 2.1, -1.5, 0.9],
        "V3": [-0.2, 0.8, -1.5, 0.6, 1.1, -0.7],
        "Amount": [50.0, 200.0, 500.0, 1200.0, 300.0, 750.0],
        "Class": [0, 0, 1, 1, 0, 1],
    }
    return pd.DataFrame(data)

# -----------------------------
# Load Model & Scaler
# -----------------------------
st.sidebar.subheader("🤖 Model & Settings")
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.sidebar.error("⚠️ Model or Scaler not found. Please ensure 'model.h5' and 'scaler.pkl' are uploaded.")
else:
    st.sidebar.success("✅ Model and Scaler loaded successfully.")

threshold = st.sidebar.slider("Fraud Detection Threshold", 0.0, 1.0, 0.5, 0.01)

# -----------------------------
# Upload Dataset
# -----------------------------
st.subheader("📂 Upload or Use Demo Dataset")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully.")
else:
    st.info("ℹ️ Using demo dataset (for testing only).")
    df = demo_dataset()

st.write("### Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Fraud Detection
# -----------------------------
if model is not None and scaler is not None:
    try:
        df_aligned = align_features(df.copy())

        # Save true labels if available
        y_true = df["Class"].values if "Class" in df.columns else None

        # Scale & predict
        X_scaled = scaler.transform(df_aligned)
        y_prob = model.predict(X_scaled).flatten()
        y_pred = (y_prob > threshold).astype(int)

        df_results = df.copy()
        df_results["Prediction"] = y_pred

        fraud_count = int(sum(y_pred))
        st.success(f"🚨 Fraudulent Transactions Detected: {fraud_count}")

        st.write("### 📝 Predictions Preview")
        st.dataframe(df_results.head(20))

        # Download predictions
        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", csv, "fraud_predictions.csv", "text/csv")

        # -----------------------------
        # Visualizations
        # -----------------------------
        st.subheader("📊 Visualizations")

        # Fraud vs Non-Fraud count
        fig, ax = plt.subplots()
        sns.countplot(x=y_pred, ax=ax, palette="viridis")
        ax.set_title("Predicted Fraud vs Non-Fraud Distribution")
        ax.set_xticklabels(["Non-Fraud", "Fraud"])
        st.pyplot(fig)

        if y_true is not None:
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], "r--")
            ax.set_title("ROC Curve")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
