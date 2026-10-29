# Customer Churn Prediction — ML Project

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Required columns include `Churn`, `TotalCharges`, `MonthlyCharges`, `tenure` and standard IBM Telco fields. The project intentionally does not fabricate a dataset or metrics.

## Run

```bash
python -m src.eda --data data/Telco-Customer-Churn.csv
python -m src.train --data data/Telco-Customer-Churn.csv
streamlit run app.py
```

Results: `outputs/model_comparison.csv`, `outputs/tuning.json`, `outputs/final_metrics.json`, `outputs/feature_importance.csv`, eight charts in `outputs/eda/`, and serialized best pipeline in `models/`.

## Method

- Stratified 80/20 split; untouched test set until the end.
- Five-fold stratified CV, four baseline models.
- Class imbalance handled with balanced class weights or XGBoost positive-class scaling.
- Leakage-safe train-fold imputation, scaling, and encoding via sklearn Pipeline.
- RandomizedSearchCV for Logistic Regression and Random Forest.
- Best model selected using **training CV ROC-AUC**, not test performance.
- Single final test evaluation and original-feature permutation importance.

## Limitations

The app is a local Streamlit dashboard until you deploy it. This project does not include telephony, a US number, a hosted API, or a Voice AI agent; those belong to the separate CareCloud assignment mentioned in the email. No performance results are claimed until actual training is run on the dataset.

## Report

See `REPORT_TEMPLATE.md`. Replace placeholders with the actual generated metrics and plots after execution.
