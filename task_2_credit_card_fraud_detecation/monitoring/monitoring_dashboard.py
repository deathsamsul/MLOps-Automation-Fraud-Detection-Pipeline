import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
import pandas as pd
import plotly.express as px
import requests

from task_2_credit_card_fraud_detecation.utils.utility import load_predictions_from_csv





# Run command
# streamlit run task_2_credit_card_fraud_detecation/monitoring/monitoring_dashboard.py


st.set_page_config(page_title="Fraud Monitoring", layout="wide")

st.title(" Fraud Detection Monitoring Dashboard")
st.write("Monitor model performance, data drift, and update labels for continuous improvement.")


# Load data from CSV
df = load_predictions_from_csv()

st.sidebar.header("Filters")

if not df.empty:

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Date filter
    min_date = df["timestamp"].dt.date.min()
    max_date = df["timestamp"].dt.date.max()

    date_range = st.sidebar.date_input("Date range",[min_date, max_date])

    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df["timestamp"].dt.date >= start_date) &(df["timestamp"].dt.date <= end_date)]


    # kpi metrics
    col1, col2, col3, col4 = st.columns(4)

    total = len(df)
    fraud_pred = df[df["prediction"] == 1].shape[0]
    labeled = df["actual_label"].notna().sum()

    correct = df[df["prediction"] == df["actual_label"]].shape[0] if labeled > 0 else 0
    accuracy = correct / labeled if labeled > 0 else 0

    col1.metric("Total Predictions", total)
    col2.metric("Predicted Fraud", fraud_pred)
    col3.metric("Labeled Samples", labeled)
    col4.metric("Accuracy (labeled)", f"{accuracy:.2%}" if labeled > 0 else "N/A")


    # chart of predictions over time
    st.subheader(" Predictions Over Time")

    df["date"] = df["timestamp"].dt.date

    time_series = df.groupby("date").size().reset_index(name="count")

    fig = px.line(time_series,x="date",y="count",title="Daily Prediction Volume")
    st.plotly_chart(fig, use_container_width=True)


    # fraud probability distribution
    st.subheader(" Fraud Probability Distribution")

    fig2 = px.histogram(df,x="fraud_probability", nbins=50, title="Histogram of Fraud Probabilities")

    st.plotly_chart(fig2, use_container_width=True)

    # table of recent predictions
    st.subheader(" Recent Predictions & Label Updates")

    st.dataframe(df.tail(10)
        #df[ ["transaction_id","timestamp","fraud_probability","prediction","actual_label" ]].tail(100)
        )


    # update true label form 

    st.subheader(" Update True Label")

    with st.form("label_update"):

        txn_id = st.text_input("Transaction ID")

        true_label = st.selectbox("Actual Label",[0, 1])

        submitted = st.form_submit_button("Update")

        if submitted:

            if txn_id.strip() == "":
                st.warning("Please enter a Transaction ID")

            else:
                try:

                    response = requests.post("http://127.0.0.1:8000/update_label",json={"transaction_id": txn_id,
                                                                                        "actual_label": true_label})

                    if response.status_code == 200:
                        st.success(f" Label for {txn_id} updated to {true_label}")

                        # reload dashboard
                        st.rerun()

                    else:

                        st.error(response.json()["detail"])

                except Exception as e:

                    st.error(f"API Connection Error: {e}")

else:

    st.info("No predictions in CSV yet.")