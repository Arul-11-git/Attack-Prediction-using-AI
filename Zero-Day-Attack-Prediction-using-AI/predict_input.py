import joblib

# Load model and vectorizer
clf = joblib.load('lg_combined_model.pkl')
vectorizer = joblib.load('lg_combined_tfidf.pkl')

# Get description from user
description = input("\nEnter CVE description: ")

# Vectorize input
X_new = vectorizer.transform([description])

# Predict
prediction = clf.predict(X_new)

# Output
print("PREDICTION:", "CRITICAL" if prediction[0] == 1 else "Not Critical")
