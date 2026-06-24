import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from preprocess import clean_text

# Load Dataset
df = pd.read_csv("data/support_tickets_100k.csv")

# Clean Ticket Text
df["ticket_text"] = df["ticket_text"].apply(clean_text)

# Features and Labels
X = df["ticket_text"]
y = df["category"]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X_tfidf = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nCategory Model Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save Model
joblib.dump(model, "models/category_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nCategory model saved successfully!")