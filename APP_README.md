💳 Credit Card Fraud Detection (TensorFlow + Streamlit)

A simple interactive web app built with TensorFlow and Streamlit that demonstrates credit card fraud detection.
It allows users to upload a dataset, view example transactions, and get fraud predictions in real-time.

🚀 Features

Upload your own dataset (.csv) or use the built-in demo dataset.
TensorFlow Neural Network trained on scaled features.
Handles imbalanced data using class weights.
Dynamic fraud prediction with adjustable decision threshold.
View predictions in tabular form with fraud counts highlighted.
Download predictions as CSV for further analysis.

🛠️ Tech Stack

Python 3.9+
TensorFlow – Model training & inference
Scikit-learn – Scaling & preprocessing
Streamlit – Interactive web UI
Joblib – Model/scaler persistence
Pandas & NumPy for data handling

📂 Project Structure
.
├── app.py               # Main Streamlit app
├── tf_model.keras       # Saved TensorFlow model (after training)
├── scaler.pkl           # Saved StandardScaler
├── train_features.json  # Stores training feature metadata
├── requirements.txt     # Dependencies
└── README.md            # Project documentation

📦 Installation & Setup

Clone this repository:
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection

Install dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py

📊 Usage

Upload a CSV file containing features:
Time, V1, V2, ... V28, Amount, (Class optional)
The app scales features and predicts fraud probability.
Predictions appear in a table with fraud count summary.
If the dataset has a Class column, evaluation metrics (precision, recall, F1) are also shown.
You can download predictions as CSV.

🧪 Demo Dataset

If no dataset is uploaded, the app uses a small synthetic demo dataset with columns:
Time, V1 ... V28, Amount, Class
This helps showcase the app without requiring Kaggle’s full dataset.

🌐 Deployment

The app can be deployed easily on:
Streamlit Cloud (free & simple)
Heroku, Render, or AWS/GCP with minor adjustments

Example deployed app:
👉 Try Live Demo


📌 Next Steps

Improve model architecture (deeper network, dropout, etc.)
Handle severe class imbalance with SMOTE or anomaly detection.
Add interactive visualizations (fraud vs. nonfraud distribution).
Experiment with XGBoost / RandomForest baselines for comparison.


👨‍💻 Author

Shaikh Abdul Wahid
B.Tech Student | Aspiring Data Scientist / ML Engineer

✨ If you found this useful, give it a ⭐ on GitHub!