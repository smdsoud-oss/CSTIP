import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

from preprocess import clean_text

# Load Dataset
df = pd.read_csv("data/support_tickets_100k.csv")

df["ticket_text"] = df["ticket_text"].apply(clean_text)

X = df["ticket_text"]
y = df["priority"]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2)
)

X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\nPriority Accuracy:")
print(f"{accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, pred))

joblib.dump(model, "models/priority_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(encoder, "models/priority_encoder.pkl")

print("\nXGBoost Priority Model Saved")