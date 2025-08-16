📌 Project Title

Credit Card Fraud Detection using Deep Learning

📖 Overview

This project focuses on detecting fraudulent credit card transactions from a highly imbalanced dataset. The goal is to maximize fraud detection (recall) without significantly compromising precision, as missing frauds has higher business risk.

🔑 Key Features

Dataset: 56,962 transactions (imbalanced, only ~0.17% fraud).

Techniques: Data balancing, Neural Networks (TensorFlow/Keras), Model Evaluation.

Results:

ROC-AUC: 0.975

Recall (Fraud): 88%

Accuracy: ~100%

Visualizations:

Confusion Matrix

ROC Curve

Training vs Validation Accuracy & Loss

🛠️ Tech Stack

Python, Pandas, NumPy

TensorFlow/Keras

Scikit-learn, Matplotlib, Seaborn

📊 Results

Successfully reduced false negatives (missed frauds).

Achieved industry-level fraud detection performance.

Balanced business trade-off between precision and recall.

🚀 How to Run
git clone <repo_link>
cd credit-card-fraud-detection
pip install -r requirements.txt
python fraud_detection.py

📷 Sample Outputs

Confusion Matrix Plot

ROC Curve Plot

Accuracy vs Validation Accuracy Graph

📌 Future Work

Deploy model as a Streamlit Web App.

Experiment with ensemble models (XGBoost, Random Forest) for comparison.

Integrate explainability with SHAP/LIME for feature importance.