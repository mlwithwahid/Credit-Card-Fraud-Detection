# app.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- Config ----------------
st.set_page_config(page_title="Credit Card Fraud (fixed predict)", layout="wide")
st.title("💳 Credit Card Fraud Detection — Robust Predict")

MODEL_PATH = "tf_model.keras"   # or tf_model.h5
SCALER_PATH = "scaler.pkl"
FEATURES_META = "train_features.json"

DEFAULT_THRESHOLD = 0.30

# ---------------- util helpers ----------------
def load_features_meta():
    if os.path.exists(FEATURES_META):
        with open(FEATURES_META, "r") as f:
            return json.load(f).get("features")
    return None

def align_features(df: pd.DataFrame, expected_cols):
    # Reindex to expected columns; fill missing with 0.0
    return df.reindex(columns=expected_cols, fill_value=0.0)

def prepare_df_for_model(df: pd.DataFrame, drop_time=True):
    """Return X_df, y (y may be None). Drops Time if requested and Class if present."""
    df = df.copy()
    y = df["Class"] if "Class" in df.columns else None
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
    if drop_time and "Time" in df.columns:
        df = df.drop(columns=["Time"])
    return df, y

# ---------------- load model + artifacts ----------------
@st.cache_resource
def load_model_and_predict_fn():
    """
    Loads model and scaler; creates a tf.function predict wrapper (reduce_retracing).
    Returns (model, scaler, predict_fn)
    """
    model = None
    scaler = None
    predict_fn = None
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            # create a single tf.function that calls the model; reduce_retracing helps avoid warnings
            @tf.function(reduce_retracing=True)
            def _predict(x):
                return model(x, training=False)
            predict_fn = _predict
    except Exception as e:
        st.warning(f"Could not load model/scaler: {e}")
    return model, scaler, predict_fn

model, scaler, predict_fn = load_model_and_predict_fn()

if model is None or scaler is None or predict_fn is None:
    st.warning("Model or scaler not found. The app will allow uploads but predictions might fallback or train.")
else:
    st.sidebar.success("✅ Model & scaler loaded")

# ---------------- UI controls ----------------
threshold = st.sidebar.slider("Decision threshold (fraud if ≥ threshold)", 0.01, 0.95, DEFAULT_THRESHOLD, 0.01)
drop_time_flag = st.sidebar.checkbox("Drop 'Time' column before predict", value=True)

# ---------------- Upload / Example ----------------
st.header("Predict from CSV")
uploaded = st.file_uploader("Upload CSV (Kaggle schema: Time,V1..V28,Amount,Class optional)", type=["csv"])

if uploaded is None:
    # try example location
    if os.path.exists("data/creditcard.csv"):
        df = pd.read_csv("data/creditcard.csv")
        st.info("Using data/creditcard.csv as example")
    else:
        st.info("No file uploaded and no example found. Upload your CSV to predict.")
        df = None
else:
    try:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded file: {df.shape[0]} rows, {df.shape[1]} cols")
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
        df = None

# ---------------- Prediction logic (batched, single call) ----------------
def safe_predict_batch(model, scaler, predict_fn, df_in, threshold, drop_time=True):
    """
    Aligns columns, scales, and predicts in a single batch call using predict_fn.
    Returns a DataFrame with prob and pred, plus messages for special cases.
    """
    messages = []
    try:
        X_raw, y_true = prepare_df_for_model(df_in, drop_time=drop_time)
        trained_features = load_features_meta()
        if trained_features is None:
            # fallback: use columns present in X_raw (preserve order); but better to save meta during training
            trained_features = list(X_raw.columns)
            messages.append("No feature metadata (train_features.json) found — aligning to uploaded columns.")
        # align
        X_aligned = align_features(X_raw, trained_features)
        # Ensure numeric and consistent dtype
        X_num = X_aligned.astype(np.float32).to_numpy()
        # Scale
        X_scaled = scaler.transform(X_num)
        # Use predict_fn on a single tensor (single call) to avoid retracing warnings
        tf_x = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
        probs_tf = predict_fn(tf_x)  # returns tensor
        probs = np.asarray(probs_tf.numpy()).reshape(-1)
        preds = (probs >= threshold).astype(int)
        out = df_in.copy()
        out["prob_fraud"] = np.round(probs, 6)
        out["pred"] = preds
        # check uniform predictions
        if len(np.unique(preds)) == 1:
            messages.append("All predictions are the same. Try lowering the threshold or retrain with class weighting.")
        # metrics if labels present
        metrics = None
        if y_true is not None:
            y_arr = y_true.to_numpy().astype(int)
            unique_y = np.unique(y_arr)
            unique_p = np.unique(preds)
            if len(unique_y) < 2 or len(unique_p) < 2:
                # avoid classification_report warning; compute simple accuracy + confusion with labels fixed
                acc = np.mean(y_arr == preds)
                cm = confusion_matrix(y_arr, preds, labels=[0,1]).tolist()
                metrics = {"accuracy": float(acc), "confusion_matrix": cm}
            else:
                rep = classification_report(y_arr, preds, output_dict=True, zero_division=0)
                cm = confusion_matrix(y_arr, preds, labels=[0,1]).tolist()
                metrics = {"report": rep, "confusion_matrix": cm}
        return out, messages, metrics
    except Exception as e:
        return None, [f"Prediction pipeline error: {e}"], None

# ---------------- UI: run predictions ----------------
if df is not None:
    st.subheader("Preview")
    st.dataframe(df.head(5))

    if model is None or scaler is None or predict_fn is None:
        st.warning("Model/scaler not loaded — predictions unavailable. Use Quick Train tab or upload saved artifacts.")
    else:
        if st.button("Run batch prediction"):
            out_df, messages, metrics = safe_predict_batch(model, scaler, predict_fn, df, threshold, drop_time=drop_time_flag)
            if messages:
                for m in messages:
                    st.warning(m)
            if out_df is not None:
                st.success(f"Predicted {out_df.shape[0]} rows. Fraud predicted: {int(out_df['pred'].sum())}")
                st.dataframe(out_df.head(20))
                st.download_button("Download predictions CSV", out_df.to_csv(index=False).encode(), "preds.csv", "text/csv")
                # show prob distribution
                hist = pd.DataFrame({"prob": out_df["prob_fraud"]})
                counts = hist["prob"].plot.hist(bins=30).get_figure()
                # Instead of complex plotting, just show value counts for simple inspection
                st.write("Probability summary (min, mean, max):", hist["prob"].min(), hist["prob"].mean(), hist["prob"].max())
            else:
                st.error("Prediction failed. See warnings above.")
            if metrics is not None:
                st.markdown("### Metrics")
                if "accuracy" in metrics:
                    st.write("Accuracy:", metrics["accuracy"])
                    st.write("Confusion matrix:", metrics["confusion_matrix"])
                else:
                    st.write("Classification report (fraud=1):")
                    st.json(metrics["report"]["1"])
                    st.write("Confusion matrix:", metrics["confusion_matrix"])

# ---------------- Manual single-row predict (no loops) ----------------
st.subheader("Manual single-row predict")
if model is not None and scaler is not None and predict_fn is not None:
    trained_features = load_features_meta()
    if trained_features is None:
        st.info("No feature metadata found; please train & save model to store feature order. Using simple fallback fields.")
        fields = ["V1","V2","V3","Amount"]
    else:
        fields = trained_features
    cols = st.columns(3)
    vals = []
    for i, f in enumerate(fields):
        default = 100.0 if f.lower()=="amount" else 0.0
        with cols[i%3]:
            vals.append(st.number_input(f, value=float(default), key=f"manual_{f}"))
    if st.button("Predict single row"):
        try:
            X_one = pd.DataFrame([vals], columns=fields)
            X_one_aligned = align_features(X_one, load_features_meta() or fields)
            Xone_scaled = scaler.transform(X_one_aligned.astype(np.float32))
            tf_x = tf.convert_to_tensor(Xone_scaled, dtype=tf.float32)
            prob = float(predict_fn(tf_x).numpy().reshape(-1)[0])
            pred = int(prob >= threshold)
            label = "🚨 Fraud" if pred==1 else "✅ Legit"
            st.metric("Prediction", label, delta=f"prob={prob:.4f}, threshold={threshold:.2f}")
        except Exception as e:
            st.error(f"Single-row prediction error: {e}")
else:
    st.info("Model/scaler not ready for single-row predictions.")

# ---------------- Advice when model predicts single class ----------------
st.markdown("---")
st.header("Tips if model predicts a single class (all 0 or all 1)")
st.write(
    "- Lower the decision threshold in the sidebar to increase recall for fraud.\n"
    "- Ensure you loaded the same `scaler.pkl` used during training. Scaling mismatch often makes outputs extreme.\n"
    "- Retrain with class weighting or undersampling (see Train/Quick-Train tab if implemented).\n"
    "- Make sure uploaded CSV columns match training columns/order. Save feature order during training into train_features.json."
)
