import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------- Streamlit Config ----------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection App")
st.write("Upload a dataset or use the demo example to explore fraud detection.")

MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.pkl"

# ---------------------- Demo Dataset ----------------------
def demo_dataset():
    data = {
        "V1": [0.1, -1.2, 1.5, -0.3, 0.7, -2.1],
        "V2": [1.3, -0.4, 0.7, 2.1, -1.5, 0.9],
        "V3": [-0.2, 0.8, -1.5, 0.6, 1.1, -0.7],
        "Amount": [50.0, 200.0, 500.0, 1200.0, 300.0, 750.0],
        "Class": [0, 0, 1, 1, 0, 1],
    }
    return pd.DataFrame(data)

# ---------------------- Load or Train Model ----------------------
model = None
scaler = None

st.subheader("🤖 Model Status")

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        st.success("✅ Model and Scaler loaded successfully.")
    else:
        st.warning("⚠️ Model not found. Training fallback model on demo dataset...")
        df_demo = demo_dataset()
        X, y = df_demo.drop("Class", axis=1), df_demo["Class"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_scaled, y)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

        st.success("✅ Fallback model trained & saved.")
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
        st.success("✅ Dataset uploaded successfully.")
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

        X_scaled = scaler.transform(X_new)
        predictions = model.predict(X_scaled)

        df_results = df.copy()
        df_results["Prediction"] = predictions

        st.write("### 📝 Predictions")
        st.dataframe(df_results.head(10))

        fraud_count = sum(predictions)
        st.success(f"🚨 Fraudulent Transactions Detected: {fraud_count}")
    except Exception as e:
        st.error(f"⚠️ Prediction error: {str(e)}")
else:
    st.warning("⚠️ Model/Scaler not available. Please check setup.")
