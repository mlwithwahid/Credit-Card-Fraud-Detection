import os
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# ---------------------- Streamlit Config ----------------------
st.set_page_config(page_title="Credit Card Fraud Detection (TF)", layout="wide")

st.title("💳 Credit Card Fraud Detection App (TensorFlow)")
st.write("Upload a dataset or use the demo example to explore fraud detection.")

MODEL_PATH = "tf_model.keras"   # ✅ use .keras extension
SCALER_PATH = "scaler.pkl"

# ---------------------- Demo Dataset ----------------------
def demo_dataset():
    """Synthetic demo dataset with same schema as Kaggle creditcard.csv"""
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]

    data = {
        "Time": [0, 10, 20, 30, 40, 50],
        "Amount": [50.0, 200.0, 500.0, 1200.0, 300.0, 750.0],
        "Class": [0, 0, 1, 1, 0, 1],
    }

    # Fill V1–V28 with some toy values
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(6)

    df = pd.DataFrame(data, columns=cols)
    return df

# ---------------------- Load or Train Model ----------------------
model = None
scaler = None
feature_cols = [c for c in demo_dataset().columns if c != "Class"]

st.subheader("🤖 Model Status")

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        st.success("✅ TensorFlow Model and Scaler loaded successfully.")
    else:
        st.warning("⚠️ TensorFlow Model not found. Training fallback model on demo dataset...")

        df_demo = demo_dataset()
        X, y = df_demo.drop("Class", axis=1), df_demo["Class"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation="relu", input_shape=(X_scaled.shape[1],)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(X_scaled, y, epochs=20, verbose=0)

        model.save(MODEL_PATH)  # ✅ correctly saves with .keras
        joblib.dump(scaler, SCALER_PATH)

        st.success("✅ Fallback TensorFlow model trained & saved.")
except Exception as e:
    st.error(f"⚠️ Error while loading/training model: {str(e)}")
    model = None
    scaler = None

# ---------------------- File Upload ----------------------
st.subheader("📂 Upload Dataset")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Dataset uploaded successfully. Shape: {df.shape}")
        st.write(df.head())
    except Exception as e:
        st.error(f"⚠️ Error reading CSV file: {str(e)}")
else:
    st.info("ℹ️ No file uploaded. Using demo dataset.")
    df = demo_dataset()
    st.write(df.head())

# ---------------------- Prediction Section ----------------------
st.subheader("🔎 Fraud Prediction")

if model is not None and scaler is not None and df is not None:
    try:
        if "Class" in df.columns:
            X_new = df.drop("Class", axis=1)
        else:
            X_new = df.copy()

        # ✅ ensure all required columns are present (align to training schema)
        X_new = X_new.reindex(columns=feature_cols, fill_value=0)

        X_scaled = scaler.transform(X_new)
        predictions = model.predict(X_scaled)
        predictions = (predictions > 0.5).astype(int).flatten()

        df_results = df.copy()
        df_results["Prediction"] = predictions

        st.write("### 📝 Predictions")
        st.dataframe(df_results.head(10))

        fraud_count = int(sum(predictions))
        st.success(f"🚨 Fraudulent Transactions Detected: {fraud_count}")

    except Exception as e:
        st.error(f"⚠️ Prediction error: {str(e)}")
else:
    st.warning("⚠️ Model/Scaler not available. Please check setup.")
