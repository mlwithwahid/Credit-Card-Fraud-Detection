import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="💳 Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("Detect fraudulent transactions using a TensorFlow model.")

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "tf_model.h5"
SCALER_PATH = "scaler.pkl"
DATA_PATH = "data/creditcard.csv"

# -----------------------------
# Helper functions
# -----------------------------
def load_dataset(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None

def preprocess_data(df):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y

def build_model(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(32, activation="relu", input_shape=(input_dim,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# -----------------------------
# Upload or Example Dataset
# -----------------------------
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = load_dataset(uploaded_file)
elif os.path.exists(DATA_PATH):
    st.sidebar.success("✅ Using example dataset (creditcard.csv).")
    df = load_dataset(DATA_PATH)
else:
    st.warning("⚠️ No dataset found. Please upload a CSV file.")
    df = None

# -----------------------------
# Show Dataset
# -----------------------------
if df is not None:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Fraud vs Non-Fraud count
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Class", ax=ax)
    ax.set_title("Fraud vs Non-Fraud Distribution")
    st.pyplot(fig)

    # -----------------------------
    # Train or Load Model
    # -----------------------------
    X, y = preprocess_data(df)

    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            st.sidebar.success("✅ Loading saved TensorFlow model...")
            model = keras.models.load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
        else:
            st.sidebar.warning("⚠️ No saved model found. Training a new one...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = build_model(X_train_scaled.shape[1])
            model.fit(X_train_scaled, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0)

            # Save model + scaler
            model.save(MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)

            # Evaluate
            y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
            st.text("📈 Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

        # -----------------------------
        # Fraud Detection Section
        # -----------------------------
        st.subheader("🔍 Fraud Detection")

        threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)

        if st.button("Run Fraud Detection"):
            try:
                X_scaled = scaler.transform(X)
                preds = (model.predict(X_scaled) > threshold).astype(int)

                fraud_count = np.sum(preds)
                st.success(f"🚨 Fraudulent Transactions Detected: {fraud_count}")

                df_results = df.copy()
                df_results["Prediction"] = preds

                st.dataframe(df_results.head(20))

                # Download
                csv = df_results.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions", csv, "fraud_predictions.csv", "text/csv")

            except Exception as e:
                st.error(f"⚠️ Prediction error: {e}")

    except Exception as e:
        st.error(f"⚠️ Model error: {e}")
