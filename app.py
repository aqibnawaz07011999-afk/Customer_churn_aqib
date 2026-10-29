
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide"
)

st.title("Customer Churn Prediction Dashboard")
st.caption("IBM Telco Customer Churn | ML-based decision support")

# Model and schema file paths
model_path = ROOT / "models" / "best_pipeline.joblib"
schema_path = ROOT / "models" / "schema.json"

# Check model file
if not model_path.exists():
    st.error("Model file not found: models/best_pipeline.joblib")
    st.stop()

# Check schema JSON file
if not schema_path.exists():
    st.error("Schema file not found: models/schema.json")
    st.stop()

# Load trained model
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error("Error loading the trained model.")
    st.exception(e)
    st.stop()

# Load schema from JSON instead of joblib
try:
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
except Exception as e:
    st.error("Error loading schema.json")
    st.exception(e)
    st.stop()

# Customer input form
with st.form("prediction"):
    cols = st.columns(3)
    values = {}

    for i, c in enumerate(schema["columns"]):
        with cols[i % 3]:
            if c in schema["categorical_values"]:
                values[c] = st.selectbox(
                    c,
                    schema["categorical_values"][c]
                )
            else:
                values[c] = st.number_input(
                    c,
                    value=float(schema["numeric_defaults"][c])
                )

    submitted = st.form_submit_button(
        "Predict churn risk",
        type="primary"
    )

# Prediction
if submitted:
    try:
        input_df = pd.DataFrame(
            [values],
            columns=schema["columns"]
        )

        score = float(
            model.predict_proba(input_df)[0, 1]
        )

        st.metric(
            "Estimated churn probability",
            f"{score:.1%}"
        )

        st.progress(score)

        st.write(
            "Predicted class:",
            "Likely to churn" if score >= 0.5 else "Likely to stay"
        )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

# Model comparison
st.divider()

metrics = ROOT / "outputs" / "model_comparison.csv"

if metrics.exists():
    st.subheader("Cross-validation model comparison")
    st.dataframe(
        pd.read_csv(metrics),
        use_container_width=True
    )

# Feature importance
imp = ROOT / "outputs" / "feature_importance.csv"

if imp.exists():
    st.subheader("Top influential original features")

    importance_df = pd.read_csv(imp)

    if {"feature", "mean_auc_drop"}.issubset(importance_df.columns):
        st.bar_chart(
            importance_df.head(10)
            .set_index("feature")["mean_auc_drop"]
        )
