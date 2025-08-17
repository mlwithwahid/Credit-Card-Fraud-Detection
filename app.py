import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("model.h5")

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

st.title("💳 Credit Card Fraud Detection 🚀")
st.markdown("Enter transaction details below or load an example transaction. The model will predict whether it is **Legit ✅** or **Fraudulent ❌**.")

# Feature names
feature_names = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18","V19",
    "V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

# Example default values (non-fraud transaction)
example_values = [
    0.0, -1.3598, -0.07278, 2.5363, 1.3781, -0.3383, 0.4624, 0.2396, 0.0987, 0.3638,
    0.0908, -0.5516, -0.6178, -0.9914, -0.3112, 1.4681, -0.4704, 0.2079, 0.0258, 0.403,
    0.2514, -0.0183, 0.2778, -0.1105, 0.0669, 0.1285, -0.1891, 0.1336, -0.0210, 149.62
]

# Session state to hold feature values
if "features" not in st.session_state:
    st.session_state.features = [0.0] * len(feature_names)

# Buttons for convenience
col1, col2 = st.columns([1,1])
with col1:
    if st.button("✨ Use Example Transaction"):
        st.session_state.features = example_values.copy()
with col2:
    if st.button("🔄 Reset All"):
        st.session_state.features = [0.0] * len(feature_names)

# Input grid (3 columns)
features = []
cols = st.columns(3)
for i, name in enumerate(feature_names):
    col = cols[i % 3]
    value = col.number_input(name, value=float(st.session_state.features[i]), format="%.4f")
    features.append(value)
    st.session_state.features[i] = value  # keep state synced

# Predict
if st.button("🔍 Predict"):
    input_data = np.array([features])  # shape (1,30)
    prediction = model.predict(input_data)
    prob = prediction[0][0]
    result = "Fraudulent ❌" if prob > 0.5 else "Legit ✅"
    confidence = prob if prob > 0.5 else 1 - prob

    if prob > 0.5:
        st.error(f"🚨 Prediction: **{result}**\n\nConfidence: {confidence*100:.2f}%")
    else:
        st.success(f"✅ Prediction: **{result}**\n\nConfidence: {confidence*100:.2f}%")

st.markdown("---")
st.info("ℹ️ This is a demo ML model trained on the Credit Card Fraud Detection dataset. Predictions may not be 100% accurate and should not be used for real financial decisions.")
