import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("model.h5")

st.title("Deep Learning Model Deployment 🚀")

# Example input fields (adjust according to your model features)
st.header("Enter Input Features:")
feature1 = st.number_input("Feature 1", min_value=0.0, max_value=100.0, value=10.0)
feature2 = st.number_input("Feature 2", min_value=0.0, max_value=100.0, value=20.0)

# Convert inputs to numpy array
input_data = np.array([[feature1, feature2]])

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction}")
