import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# =============================
# Paths
# =============================
MODEL_PATH = "tf_model.keras"
SCALER_PATH = "scaler.pkl"

# =============================
# Load or train TensorFlow model
# =============================
def load_or_train_model(df):
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            return model, scaler

        # Balance dataset
        legit = df[df.Class == 0]
        fraud = df[df.Class == 1]
        legit_downsampled = legit.sample(len(fraud), random_state=42)
        balanced = pd.concat([legit_downsampled, fraud])

        X = balanced.drop("Class", axis=1)
        y = balanced["Class"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        return model, scaler

    except Exception as e:
        st.error(f"⚠️ Model training error: {e}")
        return None, None

# =============================
# Predict fraud
# =============================
def predict_fraud(model, scaler, df, threshold=0.5):
    try:
        X = df.drop("Class", axis=1, errors="ignore")
        X_scaled = scaler.transform(X)
        probs = model.predict(X_scaled, verbose=0)
        preds = (probs > threshold).astype(int)
        return preds, probs
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
        return None, None

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection (TensorFlow)")

uploaded_file = st.file_uploader("📂 Upload your credit card transaction CSV", type=["csv"])

# Load dataset (example if not uploaded)
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    try:
        df = pd.read_csv("data/creditcard.csv")
        st.info("ℹ️ Using example dataset (data/creditcard.csv)")
    except:
        st.warning("⚠️ Example dataset not found. Please upload your own CSV.")
        df = None

if df is not None:
    st.write("### Dataset Preview", df.head())

    # Train or load model
    model, scaler = load_or_train_model(df)

    if model is not None and scaler is not None:
        st.sidebar.header("⚙️ Settings")
        threshold = st.sidebar.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5, 0.01)

        if st.button("🔍 Run Fraud Detection"):
            preds, probs = predict_fraud(model, scaler, df, threshold)

            if preds is not None:
                fraud_count = np.sum(preds)
                st.metric("🚨 Fraudulent Transactions Detected", fraud_count)

                df_result = df.copy()
                df_result["Fraud_Prob"] = probs
                df_result["Prediction"] = preds

                st.write("### Detection Results", df_result.head(20))

                st.download_button("📥 Download Results", df_result.to_csv(index=False), "fraud_results.csv")
