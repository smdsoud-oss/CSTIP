# Customer Support Ticket Intelligence Platform

## Project Overview

The Customer Support Ticket Intelligence Platform is a Machine Learning-based application that automatically analyzes customer support tickets.

The system predicts:

* Ticket Category
* Ticket Priority
* Ticket Summary

It also provides dashboard analytics and batch ticket processing using Streamlit.

---

## Features

### Single Ticket Prediction

* Enter a customer support ticket.
* Predict ticket category.
* Predict ticket priority.
* Generate ticket summary.

### Batch Prediction

* Upload CSV file containing support tickets.
* Predict category and priority for multiple tickets.
* Generate summaries automatically.

### Dashboard Analytics

* Total ticket count
* Category distribution
* Priority distribution
* Interactive charts

### Search Functionality

* Search tickets using keywords.

### Download Results

* Export prediction results as CSV.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* TF-IDF Vectorization
* Logistic Regression
* Random Forest
* XGBoost
* NLTK
* Plotly
* Streamlit

---

## Dataset Information

Dataset Size: 100000 Customer Support Tickets

Target Variables:

* Category
* Priority

Sample Categories:

* Technical Support
* Billing
* Account Access
* Complaint
* Cancellation
* General Inquiry

Priority Levels:

* High
* Medium
* Low

---

## Data Preprocessing

The following preprocessing steps were performed:

* Text cleaning
* Lowercase conversion
* Removal of special characters
* Stopword removal
* Missing value checking
* Duplicate value checking

---

## Exploratory Data Analysis (EDA)

EDA operations performed:

* Dataset overview
* Missing value analysis
* Duplicate value analysis
* Category distribution analysis
* Priority distribution analysis
* Ticket length analysis
* Visualization using bar charts and histograms

---

## Model Building

### Category Prediction Model

* TF-IDF Vectorization
* Logistic Regression

Accuracy:

* 100%

### Priority Prediction Model

* Random Forest
* XGBoost

Accuracy:

* Approximately 65%

---

## Project Structure

Customer-Support-Ticket-Intelligence/

├── app.py

├── requirements.txt

├── README.md

├── data/

├── models/

│ ├── category_model.pkl

│ ├── priority_model.pkl

│ └── tfidf_vectorizer.pkl

└── src/

├── preprocess.py

├── summarize_ticket.py

├── train_category.py

├── train_priority.py

├── train_priority_xgboost.py

└── evaluate_models.py

---
#Project Workflow

Support Ticket
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Category Prediction Model
      ↓
Priority Prediction Model
      ↓
Ticket Summary Generation
      ↓
Dashboard Visualization

train - test split

Model Evaluation

Dataset Size: 6000 Tickets

Training Data: 80% (4800 Tickets)

Testing Data: 20% (1200 Tickets)

Random State: 42

Stratified Sampling used for Priority Prediction.






## How to Run

1. Clone the repository

2. Install dependencies

pip install -r requirements.txt

3. Run Streamlit application

streamlit run app.py

---

## Future Improvements

* Improve priority prediction accuracy
* Hyperparameter tuning
* Advanced NLP techniques
* Deep Learning models
* Real-time ticket monitoring

---

## Author

Mohammed Soud SN 
Aarib afnan peersa

B.Tech – Artificial Intelligence and Data Science
