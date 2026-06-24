import joblib
from preprocess import clean_text

# Load Models
category_model = joblib.load("models/category_model.pkl")
priority_model = joblib.load("models/priority_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

while True:
    ticket = input("\nEnter Ticket: ")

    cleaned_ticket = clean_text(ticket)

    ticket_vector = vectorizer.transform([cleaned_ticket])

    category = category_model.predict(ticket_vector)[0]

    priority = priority_model.predict(ticket_vector)[0]

    print("\nPrediction Result")
    print("------------------")
    print("Category :", category)
    print("Priority :", priority)

    again = input("\nCheck another ticket? (y/n): ")

    if again.lower() != "y":
        break