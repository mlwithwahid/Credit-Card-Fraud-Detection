import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("model.h5")

st.title("Credit Card Fraud Detection 🚀")

st.header("Enter Input Features:")

# Create 30 input fields dynamically
features = []
for i in range(30):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(value)

# Convert inputs to numpy array
input_data = np.array([features])  # shape (1,30)

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0][0]}")
