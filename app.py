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

MODEL_PATH = "tf_model.keras"   # uses Keras native format
SCALER_PATH = "scaler.pkl"

# ---------------------- Demo Dataset (30 features like Kaggle) ----------------------
def demo_dataset():
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    n = 12
    rng = np.random.RandomState(42)
    data = {
        "Time": np.arange(n) * 10,
        "Amount": rng.lognormal(mean=4.5, sigma=1.0, size=n).round(2),
        "Class": np.r_[np.zeros(n-3, dtype=int), np.ones(3, dtype=int)],  # a few frauds
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, size=n)
    df = pd.DataFrame(data, columns=cols)
    return df

# ---------------------- Helpers ----------------------
def get_trained_feature_names(scaler, fallback_cols):
    """
    Prefer the scaler's feature list (what it was fit on).
    Fall back to provided list if attribute not available.
    """
    names = getattr(scaler, "feature_names_in_", None)
    if names is not None:
        return list(names)
    return list(fallback_cols)

def align_to_features(df_X, trained_cols):
    """
    Reindex df_X to EXACTLY the columns used during training.
    Missing columns are filled with 0. Extra columns are dropped.
    """
    return df_X.reindex(columns=trained_cols, fill_value=0.0)

def model_input_dim(model):
    # Works for Dense models: (None, n_features)
    try:
        return int(model.input_shape[-1])
    except Exception:
        return None

# ---------------------- Load or Train Model ----------------------
model = None
scaler = None

st.subheader("🤖 Model Status")

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        st.success("✅ TensorFlow model and scaler loaded.")
    else:
        st.warning("⚠️ No saved model/scaler found. Training a SMALL fallback demo model (for UI only).")

        df_demo = demo_dataset()
        X_demo = df_demo.drop("Class", axis=1)
        y_demo = df_demo["Class"].astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_demo)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation="relu", input_shape=(X_scaled.shape[1],)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(X_scaled, y_demo, epochs=15, verbose=0)

        # Save so future runs are stable
        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        st.success("✅ Fallback model & scaler trained and saved.")
except Exception as e:
    st.error(f"⚠️ Error while loading/training model: {e}")
    model = None
    scaler = None

# If loaded, show what the scaler expects (very important for matching)
trained_cols_preview = []
if scaler is not None:
    trained_cols_preview = get_trained_feature_names(
        scaler, demo_dataset().drop("Class", axis=1).columns
    )
    with st.expander("🔎 Columns the scaler was trained on"):
        st.code(", ".join(trained_cols_preview))

# ---------------------- File Upload ----------------------
st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload your CSV file (columns like Kaggle: Time, V1..V28, Amount, Class optional)", type=["csv"])

df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Dataset uploaded. Shape: {df.shape}")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"⚠️ Error reading CSV file: {e}")
else:
    st.info("ℹ️ No file uploaded. Using a small demo dataset (for testing only).")
    df = demo_dataset()
    st.dataframe(df.head())

# ---------------------- Prediction Section ----------------------
st.subheader("🔎 Fraud Prediction")
if model is not None and scaler is not None and df is not None:
    try:
        # Split off features/labels (keep Class if present for preview/metrics)
        if "Class" in df.columns:
            X_new = df.drop("Class", axis=1)
        else:
            X_new = df.copy()

        # Get the exact trained feature list from the scaler
        trained_cols = get_trained_feature_names(
            scaler, demo_dataset().drop("Class", axis=1).columns
        )

        # Validate scaler vs model input size
        in_dim = model_input_dim(model)
        if in_dim is not None and len(trained_cols) != in_dim:
            st.error(
                f"❌ Artifacts mismatch: scaler has {len(trained_cols)} features "
                f"but model expects {in_dim}. Replace 'tf_model.keras' and 'scaler.pkl' "
                "so they are from the SAME training run (same columns & order)."
            )
        else:
            # Align uploaded data to the EXACT training columns
            X_aligned = align_to_features(X_new, trained_cols)

            # Transform & predict
            X_scaled = scaler.transform(X_aligned)
            probs = model.predict(X_scaled, verbose=0).reshape(-1)
            preds = (probs > 0.5).astype(int)

            # Show results
            out = df.copy()
            out["prob_fraud"] = np.round(probs, 6)
            out["Prediction"] = preds

            st.write("### 📝 Predictions")
            st.dataframe(out.head(20))

            fraud_count = int(preds.sum())
            st.success(f"🚨 Fraudulent Transactions Detected: {fraud_count}")

            st.download_button(
                "📥 Download predictions CSV",
                out.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv",
            )

            # Helpful warning if everything is one class
            if len(np.unique(preds)) == 1:
                st.warning("⚠️ All predictions are the same. Consider lowering the threshold or retraining with class weights.")

    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
else:
    st.warning("⚠️ Model/Scaler not available. Please check that both files exist and are readable.")
