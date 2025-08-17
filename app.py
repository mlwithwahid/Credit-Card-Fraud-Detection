import os
import io
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

# ============== Streamlit Config ==============
st.set_page_config(page_title="Credit Card Fraud Detection (TF)", page_icon="💳", layout="wide")
st.title("💳 Credit Card Fraud Detection App (TensorFlow)")
st.write("Upload a dataset or use the demo example to explore fraud detection.")

# ============== Paths / Constants ==============
MODEL_PATHS = ["tf_model.keras", "tf_model.h5"]  # we'll accept either
SCALER_PATH = "scaler.pkl"
FEATURES_META_PATH = "train_features.json"  # stores training feature names in order

DEFAULT_THRESHOLD = 0.5

# ============== Demo Dataset (tiny fallback) ==============
def demo_dataset():
    # Small synthetic sample with both classes
    data = {
        "Time":   [0, 10, 20, 30, 40, 50],
        "V1":     [0.1, -1.2, 1.5, -0.3, 0.7, -2.1],
        "V2":     [1.3, -0.4, 0.7, 2.1, -1.5, 0.9],
        "V3":     [-0.2, 0.8, -1.5, 0.6, 1.1, -0.7],
        "Amount": [50.0, 200.0, 500.0, 1200.0, 300.0, 750.0],
        "Class":  [0, 0, 1, 1, 0, 1],
    }
    return pd.DataFrame(data)

# ============== Utilities ==============
def find_existing_model_path():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            return p
    return None

def save_features_meta(feature_names):
    with open(FEATURES_META_PATH, "w") as f:
        json.dump({"features": feature_names}, f)

def load_features_meta():
    if os.path.exists(FEATURES_META_PATH):
        with open(FEATURES_META_PATH, "r") as f:
            return json.load(f).get("features", None)
    return None

def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def align_features(df: pd.DataFrame, expected_cols, fill_value=0.0):
    """
    Reindex df to expected_cols order; extra cols dropped, missing cols filled.
    """
    X = df.reindex(columns=expected_cols, fill_value=fill_value)
    return X

def prepare_features(df: pd.DataFrame, drop_time: bool):
    """
    Prepares X (features DataFrame) and y (if present).
    - Drops 'Class' from X if present.
    - Optionally drops 'Time' (common).
    """
    df = df.copy()
    y = df["Class"] if "Class" in df.columns else None
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
    if drop_time and "Time" in df.columns:
        df = df.drop(columns=["Time"])
    return df, y

def safe_class_weights(y: np.ndarray):
    """
    Returns class weights if both classes present; else returns None.
    """
    try:
        classes = np.unique(y)
        if len(classes) == 2:
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
            return {int(classes[0]): weights[0], int(classes[1]): weights[1]}
    except Exception:
        pass
    return None

# ============== Load / Train Model ==============
@st.cache_resource
def load_or_train_model(prefer_drop_time: bool = True):
    """
    Loads TF model + scaler + trained feature names, or trains a fallback model on demo data.
    Returns: (model, scaler, trained_feature_names)
    """
    model_path = find_existing_model_path()
    scaler = None
    trained_features = None

    # Try load
    if model_path and os.path.exists(SCALER_PATH):
        try:
            model = tf.keras.models.load_model(model_path)
            scaler = joblib.load(SCALER_PATH)
            trained_features = load_features_meta()
            if trained_features is None:
                # If no meta file, infer from scaler if possible
                trained_features = getattr(scaler, "feature_names_in_", None)
            return model, scaler, list(trained_features) if trained_features is not None else None
        except Exception as e:
            st.warning(f"⚠️ Could not load existing model/scaler: {e}")

    # Fallback train on demo
    st.info("ℹ️ Training a small fallback model on the built-in demo dataset (for demo only).")
    demo = demo_dataset()
    X_df, y = prepare_features(demo, drop_time=prefer_drop_time)
    trained_features = list(X_df.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    model = build_model(X_scaled.shape[1])
    cw = safe_class_weights(y.values if hasattr(y, "values") else y)
    model.fit(X_scaled, y, epochs=25, verbose=0, class_weight=cw)

    # Save in .keras format
    model.save(MODEL_PATHS[0])  # tf_model.keras
    joblib.dump(scaler, SCALER_PATH)
    save_features_meta(trained_features)

    return model, scaler, trained_features

model, scaler, trained_features = load_or_train_model(prefer_drop_time=True)

if model is not None and scaler is not None:
    st.sidebar.success("✅ Model & scaler ready")
else:
    st.sidebar.error("❌ Model/scaler unavailable (unexpected). The app will show UI but won’t predict.")

# ============== Sidebar Controls ==============
st.sidebar.header("Prediction Settings")
threshold = st.sidebar.slider("Decision threshold (fraud if ≥ threshold)", 0.05, 0.95, DEFAULT_THRESHOLD, 0.01)
drop_time_flag = st.sidebar.checkbox("Drop 'Time' column during preprocessing", value=True)
st.sidebar.caption("Tip: Most public baselines drop 'Time'.")

# ============== Tabs ==============
tab_upload, tab_manual, tab_examples = st.tabs(["📂 Upload CSV", "⌨️ Manual Input", "📊 Example / Metrics"])

# ---------- Upload CSV ----------
with tab_upload:
    st.subheader("Upload CSV (same schema as Kaggle: Time, V1..V28, Amount, Class optional)")
    uploaded = st.file_uploader("Choose file", type=["csv"])
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            st.success(f"✅ Loaded file with shape {df_up.shape}")
            st.dataframe(df_up.head())

            # Prepare features
            X_raw, y_up = prepare_features(df_up, drop_time=drop_time_flag)

            # Align to training columns
            if trained_features is None:
                st.error("⚠️ Missing feature metadata. Retrain or include train_features.json.")
            else:
                X_aligned = align_features(X_raw, trained_features, fill_value=0.0)
                # Scale
                X_scaled = scaler.transform(X_aligned)
                # Predict probabilities
                probs = model.predict(X_scaled, verbose=0).reshape(-1)
                preds = (probs >= threshold).astype(int)

                out = df_up.copy()
                out["prob_fraud"] = np.round(probs, 6)
                out["pred"] = preds

                st.markdown("### 📝 Predictions")
                st.dataframe(out.head(20))

                st.download_button(
                    "⬇️ Download predictions CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

                # If true labels provided, show quick metrics
                if y_up is not None:
                    try:
                        report = classification_report(y_up, preds, output_dict=True, zero_division=0)
                        cm = confusion_matrix(y_up, preds)
                        st.markdown("### 📈 Quick Metrics (against provided Class)")
                        colA, colB, colC = st.columns(3)
                        colA.metric("Precision (Fraud=1)", f"{report['1']['precision']:.3f}")
                        colB.metric("Recall (Fraud=1)", f"{report['1']['recall']:.3f}")
                        colC.metric("F1 (Fraud=1)", f"{report['1']['f1-score']:.3f}")
                        st.write("Confusion Matrix [[TN, FP], [FN, TP]]:")
                        st.write(cm.tolist())
                    except Exception as e:
                        st.info(f"Metrics skipped: {e}")

                # Warn if all predictions same
                if len(np.unique(preds)) == 1:
                    st.warning(
                        "⚠️ All predictions are the same. "
                        "Try adjusting the threshold, verify scaling, or ensure your data has variability."
                    )

        except Exception as e:
            st.error(f"⚠️ Could not process file: {e}")
    else:
        st.info("Upload a CSV to get predictions here.")

# ---------- Manual Input ----------
with tab_manual:
    st.subheader("Enter a single transaction")
    # Build a minimal schema for manual entry based on trained_features
    if trained_features is None:
        st.info("Model features are unknown. Using demo features.")
        demo_cols = list(demo_dataset().drop(columns=["Class", "Time"]).columns)
        cols_for_input = demo_cols
    else:
        cols_for_input = trained_features

    cols = st.columns(3)
    values = []
    for i, col in enumerate(cols_for_input):
        default_val = 0.0
        # nice defaults if Amount is present
        if col.lower() == "amount":
            default_val = 100.0
        with cols[i % 3]:
            values.append(st.number_input(col, value=float(default_val)))

    if st.button("Predict this transaction"):
        try:
            X_one = pd.DataFrame([values], columns=cols_for_input)
            X_scaled_one = scaler.transform(X_one)
            p = float(model.predict(X_scaled_one, verbose=0).reshape(-1)[0])
            label = "🚨 Fraud" if p >= threshold else "✅ Legit"
            st.subheader(f"{label}  —  probability={p:.4f} (threshold={threshold:.2f})")
            st.progress(min(1.0, max(0.0, p)))
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------- Example / Metrics ----------
with tab_examples:
    st.subheader("Built-in Example & (Optional) Quick Train")
    st.write(
        "If you upload a full Kaggle `creditcard.csv` here, you can also **quick-train** "
        "a small TF model (with class weights) and save it for the app."
    )

    colL, colR = st.columns(2)
    with colL:
        eg_file = st.file_uploader("Optional: upload creditcard.csv here to quick-train", type=["csv"], key="train_csv")
    with colR:
        do_drop_time = st.checkbox("Drop 'Time' during training", value=True)

    if st.button("⚙️ Quick-Train & Save Model (small NN)"):
        if eg_file is None:
            st.warning("Please upload a CSV first.")
        else:
            try:
                dfx = pd.read_csv(eg_file)
                X_df, y = prepare_features(dfx, drop_time=do_drop_time)
                if y is None:
                    st.error("No `Class` column found; cannot train without labels.")
                else:
                    # Ensure both classes exist
                    classes_present = np.unique(y)
                    if len(classes_present) < 2:
                        st.error("Uploaded data contains only one class. Need both 0 and 1 to train.")
                    else:
                        # Train/valid split
                        X_tr, X_va, y_tr, y_va = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)
                        scaler2 = StandardScaler()
                        X_tr_sc = scaler2.fit_transform(X_tr)
                        X_va_sc = scaler2.transform(X_va)

                        model2 = build_model(X_tr_sc.shape[1])
                        cw = safe_class_weights(y_tr.values if hasattr(y_tr, "values") else y_tr)

                        history = model2.fit(
                            X_tr_sc, y_tr,
                            validation_data=(X_va_sc, y_va),
                            epochs=20,
                            verbose=0,
                            class_weight=cw
                        )

                        # Save artifacts
                        model2.save(MODEL_PATHS[0])  # tf_model.keras
                        joblib.dump(scaler2, SCALER_PATH)
                        save_features_meta(list(X_df.columns))

                        st.success("✅ Model, scaler, and feature metadata saved. App will use them on next run.")
                        st.write("Validation accuracy (last epoch):", float(history.history["val_accuracy"][-1]))
            except Exception as e:
                st.error(f"Training failed: {e}")

    st.markdown("---")
    st.caption(
        "Notes:\n"
        "- This app aligns incoming columns to the **training feature order** to avoid name/order mismatches.\n"
        "- It uses **StandardScaler** and saves it alongside the model.\n"
        "- A **class-weighted** small NN helps avoid the “all zeros” issue on imbalanced data.\n"
        "- Use the **threshold slider** to tune recall vs precision."
    )
