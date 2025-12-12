import pandas as pd
import numpy as np
import re
import nltk
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')

lemm = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'cve-\d+-\d+', ' ', text)
    text = re.sub(r'\d+(\.\d+)*', ' ', text)
    text = re.sub(r'[^a-z ]', ' ', text)
    text = ' '.join([lemm.lemmatize(w) for w in text.split() if w not in stop_words])
    return text

# ------------------------------------------------------
# 1. Load the FINAL CLEANED COMBINED dataset
# ------------------------------------------------------
df = pd.read_csv("combined_cve_dataset.csv")

# Rename to match training fields
df.rename(columns={'description': 'text', 'severity': 'Severity'}, inplace=True)

# Keep only needed columns
df = df[['Severity', 'text']].dropna()

# ------------------------------------------------------
# 2. Clean text
# ------------------------------------------------------
df['text'] = df['text'].apply(clean_text)

# Drop very short descriptions
df = df[df['text'].str.split().str.len() > 5]

# ------------------------------------------------------
# 3. Prepare labels
# ------------------------------------------------------
df['Severity'] = df['Severity'].str.lower().str.strip()
df['label'] = df['Severity'].apply(lambda x: 1 if x in ['high', 'critical'] else 0)

# ------------------------------------------------------
# 4. Vectorize text
# ------------------------------------------------------
tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1,3),
    stop_words='english'
)

X = tfidf.fit_transform(df['text'])
y = df['label']

# ------------------------------------------------------
# 5. Train-test split
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------
# 6. Handle class imbalance
# ------------------------------------------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# ------------------------------------------------------
# 7. Train Logistic Regression
# ------------------------------------------------------
clf = LogisticRegression(max_iter=5000, class_weight='balanced')
clf.fit(X_train_res, y_train_res)

# ------------------------------------------------------
# 8. Evaluate
# ------------------------------------------------------
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------
# 9. Save model + TF-IDF
# ------------------------------------------------------
joblib.dump(clf, "lg_combined_model.pkl")
joblib.dump(tfidf, "lg_combined_tfidf.pkl")

print("\n✅ Model training complete using combined dataset!")
print("Saved as lg_combined_model.pkl and lg_combined_tfidf.pkl")
