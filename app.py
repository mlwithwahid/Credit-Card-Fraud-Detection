import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("💳 Credit Card Fraud Detection")

# Input fields
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
# add inputs for all required features...

if st.button("Predict"):
    features = np.array([[feature1, feature2]])  # add more features
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("⚠️ Fraud Detected!")
    else:
        st.success("✅ Legitimate Transaction")
