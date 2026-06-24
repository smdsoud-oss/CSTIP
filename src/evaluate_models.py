import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from preprocess import clean_text

# Load Dataset
df = pd.read_csv("data/support_tickets_100k.csv")

# Clean Text
df["ticket_text"] = df["ticket_text"].apply(clean_text)

X = df["ticket_text"]

# Load Vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# ==========================
# CATEGORY MODEL EVALUATION
# ==========================

y_category = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_category,
    test_size=0.2,
    random_state=42
)

category_model = joblib.load("models/category_model.pkl")

X_test_vec = vectorizer.transform(X_test)

y_pred = category_model.predict(X_test_vec)

print("\n==============================")
print("CATEGORY MODEL RESULTS")
print("==============================")

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================
# PRIORITY MODEL EVALUATION
# ==========================

y_priority = df["priority"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_priority,
    test_size=0.2,
    random_state=42
)

priority_model = joblib.load("models/priority_model.pkl")

X_test_vec = vectorizer.transform(X_test)

y_pred = priority_model.predict(X_test_vec)

print("\n==============================")
print("PRIORITY MODEL RESULTS")
print("==============================")

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))