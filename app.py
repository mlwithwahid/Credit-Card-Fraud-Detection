import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Feature names
FEATURES = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18","V19",
    "V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

# Title
st.title("💳 Credit Card Fraud Detection App")
st.write("Upload transaction data or try sample transactions below.")

# Tabs for navigation
tab1, tab2 = st.tabs(["🔼 Upload Transaction", "📊 Example Transactions"])

with tab1:
    st.subheader("Upload Your Transaction File (CSV)")
    uploaded_file = st.file_uploader("Upload a CSV with same 30 features", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            # Check feature columns
            missing_cols = [col for col in FEATURES if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                st.success("✅ File looks good!")

                for idx, row in df.iterrows():
                    input_array = np.array(row[FEATURES]).reshape(1, -1)
                    prediction = model.predict(input_array, verbose=0)[0][0]

                    # Show probability bar
                    st.write(f"### Transaction {idx+1}")
                    fig, ax = plt.subplots()
                    ax.bar(["Legit", "Fraud"], [1-prediction, prediction], color=["green","red"])
                    ax.set_ylim([0,1])
                    ax.set_ylabel("Probability")
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")

with tab2:
    st.subheader("Try Example Transactions")

    try:
        if os.path.exists("data/creditcard.csv"):
            df = pd.read_csv("data/creditcard.csv")

            # Safely sample (min of available vs requested)
            legit_samples = df[df["Class"] == 0].sample(
                n=min(2, len(df[df["Class"] == 0])), random_state=42
            )
            fraud_samples = df[df["Class"] == 1].sample(
                n=min(2, len(df[df["Class"] == 1])), random_state=42
            )

            samples = pd.concat([legit_samples, fraud_samples])

            for idx, row in samples.iterrows():
                input_array = np.array(row[FEATURES]).reshape(1, -1)
                prediction = model.predict(input_array, verbose=0)[0][0]

                st.write(f"### Example Transaction (Class: {row['Class']})")
                fig, ax = plt.subplots()
                ax.bar(["Legit", "Fraud"], [1-prediction, prediction], color=["green","red"])
                ax.set_ylim([0,1])
                ax.set_ylabel("Probability")
                st.pyplot(fig)
        else:
            st.warning("⚠️ Example dataset not found. Upload data/creditcard.csv to use this section.")
    except Exception as e:
        st.error(f"⚠️ Error loading examples: {e}")
