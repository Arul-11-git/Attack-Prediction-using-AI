🔐 CVE Severity Prediction (Machine Learning Project)

This project predicts the severity of CVE vulnerabilities using machine learning.
The model reads a CVE description and classifies it as:

High / Critical

Low / Medium

It uses real CVE datasets collected from multiple years, including the official NVD 2024 feed.

📘 What This Project Does

Cleans and standardizes CVE data

Merges multi-year CVE datasets

Preprocesses text using NLP

Converts descriptions into TF-IDF vectors

Trains a Logistic Regression classifier

Predicts if a vulnerability is critical or not

Saves the model for future predictions

📂 Files Used

cleaned_cve_dataset.csv — cleaned historical CVEs

parsed_nvd_2024.csv — structured CVE 2024 data

combined_cve_dataset.csv — final merged dataset

🛠️ Technologies

Python

Scikit-learn

Pandas

NLTK

Imbalanced-Learn (SMOTE)

Joblib

🚀 How to Run the Model
Install dependencies:
pip install pandas numpy scikit-learn nltk imbalanced-learn joblib

Train the model:
python train_model.py

Predict severity:
python predict.py

📊 Example Output
Input: "A remote code execution vulnerability in..."
Prediction: High / Critical

🌟 Future Improvements

Add deep-learning models (BERT)

Build a simple UI with Streamlit

Support multi-class severity prediction
