"""
Streamlit dashboard: model health at a glance.

Reads the same logs.db and drift_report.json the API and drift detector
write to, so it always reflects the latest state without needing its own
connection to the model.
"""
import json
import subprocess
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

import db

st.set_page_config(page_title="ML Model Health Monitor", layout="wide")

st.title("Live ML Model Health Monitor")
st.caption("Housing price predictor served via FastAPI, monitored for data drift with the KS test.")

if st.button("Run drift check now"):
    subprocess.run([sys.executable, "drift_detector.py"])

try:
    logs = db.fetch_all()
except Exception:
    logs = pd.DataFrame()

try:
    with open("drift_report.json") as f:
        drift_report = json.load(f)
except FileNotFoundError:
    drift_report = None

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total predictions served", len(logs))
if not logs.empty:
    col2.metric("Avg latency (ms)", f"{logs['latency_ms'].mean():.2f}")
    col3.metric("p95 latency (ms)", f"{logs['latency_ms'].quantile(0.95):.2f}")
else:
    col2.metric("Avg latency (ms)", "—")
    col3.metric("p95 latency (ms)", "—")

if drift_report:
    status = "DRIFT DETECTED" if drift_report["drift_detected"] else "HEALTHY"
    col4.metric("Model status", status)
else:
    col4.metric("Model status", "Not checked yet")

st.divider()

if drift_report is None:
    st.info("No drift report yet. Click **Run drift check now** after sending some traffic.")
elif drift_report["drift_detected"]:
    st.error(
        f"Data drift detected in: **{', '.join(drift_report['drifted_columns'])}**. "
        "Incoming data no longer matches the training distribution — consider retraining."
    )
else:
    st.success("Incoming data matches the training distribution. Model is healthy.")

if drift_report and drift_report.get("column_details"):
    st.subheader("Feature-level drift detail")
    detail_df = pd.DataFrame(drift_report["column_details"]).T
    detail_df.index.name = "feature"
    st.dataframe(detail_df, use_container_width=True)

if not logs.empty:
    st.subheader("Requests over time")
    logs["timestamp"] = pd.to_datetime(logs["timestamp"])
    logs["cumulative"] = range(1, len(logs) + 1)
    fig1 = px.line(logs, x="timestamp", y="cumulative", title="Cumulative predictions served")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Incoming feature distributions (most recent 200)")
    try:
        reference = pd.read_csv("reference_data.csv")
        recent = db.fetch_recent(200)
        feature = st.selectbox(
            "Feature", ["size_sqft", "bedrooms", "age_years", "location_score", "distance_to_city_km"]
        )
        combined = pd.concat(
            [
                reference[[feature]].assign(source="training reference"),
                recent[[feature]].assign(source="recent live traffic"),
            ]
        )
        fig2 = px.histogram(
            combined,
            x=feature,
            color="source",
            barmode="overlay",
            opacity=0.6,
            title=f"{feature}: training vs. live traffic",
        )
        st.plotly_chart(fig2, use_container_width=True)
    except FileNotFoundError:
        st.warning("reference_data.csv not found — run train_model.py first.")
else:
    st.warning("No predictions logged yet. Run `simulate_traffic.py` to generate traffic.")
