# app.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------- Config ----------------------
st.set_page_config(page_title="Credit Card Fraud Detection (TensorFlow)", layout="wide")
st.title("💳 Credit Card Fraud Detection (TensorFlow)")
st.caption("Upload Kaggle-format data (Time, V1..V28, Amount, [Class]) or use the demo/train tools below.")

MODEL_PATH = "tf_model.keras"        # use .keras or .h5
SCALER_PATH = "scaler.pkl"
FEATURES_META_PATH = "train_features.json"  # stores feature order
DEFAULT_THRESHOLD = 0.30  # lower than 0.5 to help recall on imbalanced sets

# ---------------------- Demo Dataset ----------------------
def demo_dataset():
    # Tiny demo with both classes (so fallback always works)
    data = {
        "Time":   [0,   10,   20,   30,   40,   50,   60,   70],
        "V1":     [0.1, -1.2, 1.5,  -0.3, 0.7,  -2.1, 0.2,  1.0],
        "V2":     [1.3, -0.4, 0.7,  2.1,  -1.5, 0.9,  0.1, -0.8],
        "V3":     [-0.2,0.8, -1.5,  0.6,   1.1, -0.7, 0.4, -0.5],
        "V4":     [0.0]*8,
        "V5":     [0.0]*8,
        "V6":     [0.0]*8,
        "V7":     [0.0]*8,
        "V8":     [0.0]*8,
        "V9":     [0.0]*8,
        "V10":    [0.0]*8,
        "V11":    [0.0]*8,
        "V12":    [0.0]*8,
        "V13":    [0.0]*8,
        "V14":    [0.0]*8,
        "V15":    [0.0]*8,
        "V16":    [0.0]*8,
        "V17":    [0.0]*8,
        "V18":    [0.0]*8,
        "V19":    [0.0]*8,
        "V20":    [0.0]*8,
        "V21":    [0.0]*8,
        "V22":    [0.0]*8,
        "V23":    [0.0]*8,
        "V24":    [0.0]*8,
        "V25":    [0.0]*8,
        "V26":    [0.0]*8,
        "V27":    [0.0]*8,
        "V28":    [0.0]*8,
        "Amount": [50, 200, 500, 1200, 300,  750,  20,  1800],
        "Class":  [0,   0,   1,    1,    0,    1,    0,    1],
    }
    return pd.DataFrame(data)

# Expected Kaggle feature set (we usually drop Time)
BASE_FEATURES = ["V" + str(i) for i in range(1, 29)] + ["Amount"]

# ---------------------- Utilities ----------------------
def save_features_meta(feature_names):
    with open(FEATURES_META_PATH, "w") as f:
        json.dump({"features": list(feature_names)}, f)

def load_features_meta():
    if os.path.exists(FEATURES_META_PATH):
        with open(FEATURES_META_PATH, "r") as f:
            return json.load(f).get("features")
    return None

def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def prepare_Xy(df: pd.DataFrame, drop_time: bool = True):
    df = df.copy()
    y = df["Class"] if "Class" in df.columns else None
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
    if drop_time and "Time" in df.columns:
        df = df.drop(columns=["Time"])
    return df, y

def align_features(df: pd.DataFrame, expected_cols):
    # Reindex to the expected feature order; fill missing with 0; drop extras
    return df.reindex(columns=expected_cols, fill_value=0.0)

def compute_class_weights_if_possible(y):
    try:
        classes = np.unique(y)
        if len(classes) == 2:
            weights = compute_class_weight("balanced", classes=classes, y=y)
            return {int(classes[0]): float(weights[0]), int(classes[1]): float(weights[1])}
    except Exception:
        pass
    return None

# ---------------------- Load or Train ----------------------
@st.cache_resource
def load_or_train_model_fallback():
    """
    Load model+scaler+features if present, otherwise train on demo dataset and save.
    """
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            feat = load_features_meta()
            return model, scaler, feat
    except Exception as e:
        st.warning(f"Could not load saved model/scaler: {e}")

    # Fallback train on demo
    st.info("Training a small fallback model on the built-in demo dataset.")
    demo = demo_dataset()
    X_df, y = prepare_Xy(demo, drop_time=True)
    trained_features = list(X_df.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    model = build_model(X_scaled.shape[1])
    cw = compute_class_weights_if_possible(y.values if hasattr(y, "values") else y)
    model.fit(X_scaled, y, epochs=30, verbose=0, class_weight=cw)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    save_features_meta(trained_features)

    return model, scaler, trained_features

def train_and_save_from_df(df: pd.DataFrame, drop_time=True, epochs=15):
    """
    Trains a model from a full Kaggle-like dataset (must include Class with both 0 and 1).
    Uses stratified split, StandardScaler, and class weights. Saves model, scaler, and feature names.
    """
    # Ensure labels
    if "Class" not in df.columns:
        raise ValueError("No 'Class' column found; cannot train without labels.")

    # Prepare
    X_df, y = prepare_Xy(df, drop_time=drop_time)

    # Ensure we have both classes; if not, augment with demo
    if len(np.unique(y)) < 2:
        demo = demo_dataset()
        Xd, yd = prepare_Xy(demo, drop_time=drop_time)
        # Align columns union
        all_cols = sorted(set(X_df.columns) | set(Xd.columns))
        X_df = X_df.reindex(columns=all_cols, fill_value=0.0)
        Xd = Xd.reindex(columns=all_cols, fill_value=0.0)
        X_df = pd.concat([X_df, Xd], axis=0, ignore_index=True)
        y = pd.concat([y, demo["Class"]], axis=0, ignore_index=True)

    # (Optional) simple undersampling to balance
    if "Class" in df.columns:
        # Reconstruct for balancing on original df
        full = df.copy()
        if drop_time and "Time" in full.columns:
            full = full.drop(columns=["Time"])
        legit = full[full["Class"] == 0]
        fraud = full[full["Class"] == 1]
        if len(fraud) > 0 and len(legit) > 0:
            n = min(len(fraud), len(legit))
            full_bal = pd.concat([legit.sample(n, random_state=42), fraud.sample(n, random_state=42)])
            X_df, y = prepare_Xy(full_bal, drop_time=False if drop_time is False else True)

    trained_features = list(X_df.columns)

    X_tr, X_va, y_tr, y_va = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)

    model = build_model(X_tr_sc.shape[1])
    cw = compute_class_weights_if_possible(y_tr.values if hasattr(y_tr, "values") else y_tr)

    hist = model.fit(
        X_tr_sc, y_tr,
        validation_data=(X_va_sc, y_va),
        epochs=epochs, verbose=0,
        class_weight=cw
    )

    # Save artifacts
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    save_features_meta(trained_features)

    # Evaluate quick metrics
    y_pred = (model.predict(X_va_sc, verbose=0).reshape(-1) >= 0.5).astype(int)
    report = classification_report(y_va, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_va, y_pred).tolist()
    return hist, report, cm, trained_features

# ---------------------- Load model/scaler or fallback ----------------------
model, scaler, trained_features = load_or_train_model_fallback()

if model is None or scaler is None:
    st.error("❌ Model/scaler unavailable. Check TensorFlow or requirements.")
    st.stop()

# ---------------------- Sidebar settings ----------------------
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Decision threshold (fraud if ≥ threshold)", 0.05, 0.95, DEFAULT_THRESHOLD, 0.01)
drop_time_flag = st.sidebar.checkbox("Drop 'Time' column for predictions", value=True)
st.sidebar.caption("Tip: Most Kaggle baselines drop 'Time'.")

# ---------------------- Tabs ----------------------
tab1, tab2, tab3 = st.tabs(["📂 Predict on CSV", "⌨️ Predict Single Row", "🧪 Train / Retrain"])

# ===== Predict on CSV =====
with tab1:
    st.subheader("Upload a CSV to Predict")
    uploaded = st.file_uploader("Choose CSV (Kaggle schema preferred)", type=["csv"], key="pred_csv")

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            st.success(f"Loaded file: {df_up.shape[0]} rows, {df_up.shape[1]} columns")
            st.dataframe(df_up.head())

            # Prepare input features
            X_raw, y_true = prepare_Xy(df_up, drop_time=drop_time_flag)

            # Align to trained feature order (or fallback to BASE_FEATURES)
            expected = trained_features or BASE_FEATURES
            X_aligned = align_features(X_raw, expected)

            # Scale & predict
            X_scaled = scaler.transform(X_aligned)
            probs = model.predict(X_scaled, verbose=0).reshape(-1)
            preds = (probs >= threshold).astype(int)

            out = df_up.copy()
            out["prob_fraud"] = np.round(probs, 6)
            out["pred"] = preds
            st.markdown("### 📝 Predictions")
            st.dataframe(out.head(20))

            # Summary
            unique_preds = np.unique(preds)
            st.info(f"Prediction summary — counts: {dict(zip(*np.unique(preds, return_counts=True)))}")
            if len(unique_preds) == 1:
                st.warning(
                    "⚠️ All predictions are the same. Try lowering the threshold, "
                    "check scaling, or retrain the model with class weighting."
                )

            # Metrics if labels present
            if y_true is not None:
                try:
                    rep = classification_report(y_true, preds, output_dict=True, zero_division=0)
                    cm = confusion_matrix(y_true, preds).tolist()
                    c1 = st.columns(3)
                    c1[0].metric("Precision (Fraud=1)", f"{rep['1']['precision']:.3f}")
                    c1[1].metric("Recall (Fraud=1)", f"{rep['1']['recall']:.3f}")
                    c1[2].metric("F1 (Fraud=1)", f"{rep['1']['f1-score']:.3f}")
                    st.write("Confusion Matrix [[TN, FP], [FN, TP]]:", cm)
                except Exception as e:
                    st.info(f"Metrics skipped: {e}")

            # Download
            st.download_button(
                "⬇️ Download predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"❌ Could not process file: {e}")
    else:
        st.info("Upload a CSV above to run predictions.")

# ===== Predict Single Row =====
with tab2:
    st.subheader("Manual Single-Row Prediction")

    # Use trained feature names; fallback to BASE_FEATURES
    cols_for_input = trained_features or BASE_FEATURES
    col_objs = st.columns(3)
    values = []
    for i, col in enumerate(cols_for_input):
        default_val = 0.0 if col.lower() != "amount" else 100.0
        with col_objs[i % 3]:
            values.append(st.number_input(col, value=float(default_val), key=f"in_{col}"))
    if st.button("Predict Row"):
        try:
            X_one = pd.DataFrame([values], columns=cols_for_input)
            X_scaled_one = scaler.transform(X_one)
            p = float(model.predict(X_scaled_one, verbose=0).reshape(-1)[0])
            label = "🚨 Fraud" if p >= threshold else "✅ Legit"
            st.subheader(f"{label} — probability={p:.4f} (threshold={threshold:.2f})")
            st.progress(min(1.0, max(0.0, p)))
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ===== Train / Retrain =====
with tab3:
    st.subheader("Quick Train / Retrain Model")
    st.write("Upload full Kaggle dataset (`creditcard.csv`) to train a better model with class weights.")

    train_file = st.file_uploader("Upload labeled dataset with 'Class' column", type=["csv"], key="train_csv")
    drop_time_train = st.checkbox("Drop 'Time' during training", value=True)
    epochs = st.slider("Epochs", 5, 50, 20, 1)

    if st.button("⚙️ Train & Save"):
        if train_file is None:
            st.warning("Please upload a labeled CSV first.")
        else:
            try:
                dfx = pd.read_csv(train_file)
                hist, report, cm, feats = train_and_save_from_df(dfx, drop_time=drop_time_train, epochs=epochs)
                st.success("✅ Model, scaler, and feature metadata saved. Reload the app to use the new model.")
                st.write(f"Validation Accuracy (last epoch): {hist.history['val_accuracy'][-1]:.3f}")
                st.write("Confusion Matrix (val) [[TN, FP], [FN, TP]]:", cm)
                st.write("Fraud (1) metrics:", {k: round(v, 3) for k, v in report['1'].items() if isinstance(v, (int, float))})
            except Exception as e:
                st.error(f"Training failed: {e}")

st.markdown("---")
st.caption(
    "Tips: Lower the threshold to increase recall. Ensure uploaded CSV columns match the training features. "
    "This app stores feature order in train_features.json and aligns incoming data accordingly."
)
