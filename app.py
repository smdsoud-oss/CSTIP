import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

from src.preprocess import clean_text
from src.summarize_ticket import summarize_ticket

# ==========================================

# LOAD MODELS

# ==========================================

category_model = joblib.load("models/category_model.pkl")
priority_model = joblib.load("models/priority_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# ==========================================

# PAGE TITLE

# ==========================================

st.set_page_config(
page_title="Customer Support Ticket Intelligence",
layout="wide"
)

st.title("📊 Customer Support Ticket Intelligence Platform")

# ==========================================

# SINGLE TICKET PREDICTION

# ==========================================

# SINGLE TICKET PREDICTION

st.header("Single Ticket Prediction")

ticket = st.text_area("Enter Support Ticket")

if st.button("Analyze Ticket"):

    cleaned_ticket = clean_text(ticket)

    ticket_vector = vectorizer.transform(
        [cleaned_ticket]
    )

    category = category_model.predict(
        ticket_vector
    )[0]

    priority = priority_model.predict(
        ticket_vector
    )[0]

    summary = summarize_ticket(ticket)

    st.success("Analysis Complete")

    st.subheader("Ticket Summary")
    st.write(summary)

    st.subheader("Predicted Category")
    st.write(category)

    st.subheader("Predicted Priority")
    st.write(priority)

# ==========================================

# BATCH PREDICTION

# ==========================================

st.header("Batch Prediction")

uploaded_file = st.file_uploader(
"Upload CSV File",
type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "ticket_text" not in df.columns:

        st.error(
            "CSV must contain a column named 'ticket_text'"
        )

    else:

        df["cleaned_ticket"] = df["ticket_text"].apply(
            clean_text
        )

        X = vectorizer.transform(
            df["cleaned_ticket"]
        )

        df["Predicted_Category"] = (
            category_model.predict(X)
        )

        df["Predicted_Priority"] = (
            priority_model.predict(X)
        )

        st.subheader("Predicted Priority Counts")
        st.write(df["Predicted_Priority"].value_counts())

    # ==================================
    # SUMMARIZATION
    # ==================================

    df["Summary"] = (
        df["ticket_text"]
        .apply(summarize_ticket)
    )

    st.success(
        "Batch Prediction Complete"
    )

    # ==================================
    # RESULTS TABLE
    # ==================================

    st.subheader(
        "Prediction Results"
    )

    st.dataframe(
        df[
            [
                "ticket_text",
                "Predicted_Category",
                "Predicted_Priority",
                "Summary"
            ]
        ]
    )

    # ==================================
    # ANALYTICS DASHBOARD
    # ==================================

    st.header(
        "📊 Analytics Dashboard"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Tickets",
            len(df)
        )

    with col2:

        high_priority = len(
            df[
                df["Predicted_Priority"]
                == "High"
            ]
        )

        st.metric(
            "High Priority",
            high_priority
        )

    with col3:

        top_category = (
            df["Predicted_Category"]
            .value_counts()
            .idxmax()
        )

        st.metric(
            "Top Category",
            top_category
        )

    st.divider()

    # ==================================
    # CATEGORY PERCENTAGE CARDS
    # ==================================

    st.subheader(
        "📈 Category Percentage Breakdown"
    )

    category_percentages = (
        df["Predicted_Category"]
        .value_counts(normalize=True)
        * 100
    )

    cols = st.columns(
        len(category_percentages)
    )

    for i, (
        category,
        percentage
    ) in enumerate(
        category_percentages.items()
    ):

        with cols[i]:

            st.metric(
                category,
                f"{percentage:.1f}%"
            )

    st.divider()

    # ==================================
    # CHARTS
    # ==================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Category Distribution"
        )

        category_count = (
            df["Predicted_Category"]
            .value_counts()
            .reset_index()
        )

        category_count.columns = [
            "Category",
            "Count"
        ]

        fig1 = px.bar(
            category_count,
            x="Category",
            y="Count",
            title="Ticket Categories"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Priority Distribution"
        )

        priority_count = (
            df["Predicted_Priority"]
            .value_counts()
            .reset_index()
        )

        priority_count.columns = [
            "Priority",
            "Count"
        ]

        fig2 = px.pie(
            priority_count,
            names="Priority",
            values="Count",
            title="Priority Levels"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # ==================================
    # SOURCE CHANNEL DISTRIBUTION
    # ==================================

    if "source_channel" in df.columns:

        st.subheader(
            "📡 Ticket Source Distribution"
        )

        source_count = (
            df["source_channel"]
            .value_counts()
            .reset_index()
        )

        source_count.columns = [
            "Source",
            "Count"
        ]

        fig3 = px.pie(
            source_count,
            names="Source",
            values="Count",
            title="Ticket Source Channels"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    st.divider()

    # ==================================
    # SEARCH TICKETS
    # ==================================

    st.subheader(
        "🔍 Search Tickets"
    )

    search_text = st.text_input(
        "Search by keyword"
    )

    if search_text:

        filtered_df = df[
            df["ticket_text"]
            .str.contains(
                search_text,
                case=False,
                na=False
            )
        ]

        st.dataframe(
            filtered_df[
                [
                    "ticket_text",
                    "Predicted_Category",
                    "Predicted_Priority",
                    "Summary"
                ]
            ]
        )

    st.divider()

    # ==================================
    # DOWNLOAD RESULTS
    # ==================================

    csv = (
        df.to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="📥 Download Results CSV",
        data=csv,
        file_name="ticket_predictions.csv",
        mime="text/csv"
    )

