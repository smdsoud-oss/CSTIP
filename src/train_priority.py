import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from preprocess import clean_text

# Load Dataset
df = pd.read_csv("data/support_tickets_100k.csv")

# Clean Ticket Text
df["ticket_text"] = df["ticket_text"].apply(clean_text)

# Features and Labels
X = df["ticket_text"]
y = df["priority"]

# Load TF-IDF Vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

X_tfidf = vectorizer.transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
     stratify=y
)

# Train Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=3,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nPriority Model Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save Model
joblib.dump(model, "models/priority_model.pkl")

print("\nPriority model saved successfully!")