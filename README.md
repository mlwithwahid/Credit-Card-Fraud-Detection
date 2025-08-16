# Credit-Card-Fraud-Detection
This project focuses on detecting fraudulent credit card transactions using Deep Learning techniques. Fraud detection is a  imbalanced classification problem, where fraudulent transactions are rare compared to genuine ones. The goal of this project is to maximize recall for fraud cases while maintaining a strong overall accuracy, ROC-AUC score.

📊 Project Overview
Dataset: Highly imbalanced credit card transaction dataset (with anonymized features).
Objective: Build a model that accurately detects fraud while minimizing false negatives (missing fraud cases).

Techniques Used:
Data preprocessing (scaling, handling imbalance)
Exploratory Data Analysis (EDA)
Model training with ML and DL approaches
Evaluation using metrics beyond accuracy (recall, precision, F1-score, ROC-AUC)

🧠 Models Implemented
Deep Neural Network (TensorFlow/Keras)

📈 Results
Final selected model achieved:
Accuracy: ~99.7%
Recall (Fraud class): ~88%
ROC-AUC Score: ~0.975
Confusion Matrix: Low false negatives compared to other models
This balance ensures fraud cases are detected with high sensitivity while minimizing false alarms.

🚀 Tech Stack
Python
Scikit-learn
TensorFlow/Keras
Pandas, NumPy, Matplotlib, Seaborn

🔮 Key Learnings

Importance of handling imbalanced datasets.
Why recall matters more than raw accuracy in fraud detection.
Trade-offs between precision and recall in real-world applications.
Practical model evaluation beyond accuracy.

📌 Future Improvements
Deploy the model using Flask/FastAPI or Streamlit for real-time fraud detection.
Experiment with SMOTE/undersampling to handle imbalance.
Try ensemble deep learning models for further improvement.

🤝 Connect
If you liked this project or have suggestions, feel free to connect with me on LinkedIn 🚀
LinkedIn : www.linkedin.com/in/shaikh-abdul-wahid-78a13a2b5
