# Technical Report: Customer Churn Prediction

## 1. Executive summary
The goal is to predict customer churn using the IBM Telco Customer Churn dataset. Four supervised classification algorithms are trained and compared using a consistent stratified cross-validation protocol. **Fill in results after running the project.**

## 2. Data and target
Target: `Churn` (Yes=1, No=0). Exclude `customerID` because it is an identifier. Convert `TotalCharges` to numeric and impute missing values within CV folds. Record dataset size, missing-value counts and churn distribution from actual data.

## 3. Exploratory analysis
Eight plots cover churn frequency, gender, contract, internet service, payment method, tenure, monthly charges and total charges. Interpret each generated plot, avoiding causal claims.

## 4. Preprocessing and leakage prevention
A stratified 80/20 split is created before preprocessing. Numeric fields use median imputation and standardization; categorical fields use most-frequent imputation and one-hot encoding. All transforms are inside CV pipelines. Class imbalance is handled through class weighting and XGBoost positive-class scaling.

## 5. Model training and tuning
Baseline models: Logistic Regression, Random Forest, SVM and XGBoost. Evaluate using stratified five-fold CV. Tune Logistic Regression and Random Forest with RandomizedSearchCV. Summarize the actual search spaces and optimal parameters in `outputs/tuning.json`.

## 6. Results and model selection
Insert actual table from `outputs/model_comparison.csv`. Select the model with the highest training cross-validation ROC-AUC. Report held-out accuracy, precision, recall, F1, ROC-AUC and confusion matrix from `outputs/final_metrics.json`. Compare train and CV ROC-AUC as an overfitting diagnostic. Do not use held-out test results to reselect models.

## 7. Explainability
Use permutation importance measured as decrease in held-out ROC-AUC. Discuss the top five features from `outputs/feature_importance.csv`. Note that importance reflects predictive contribution, not causality.

## 8. Deployment and limitations
Streamlit app loads the fitted pipeline and accepts customer features. Explain dataset representativeness, temporal drift, class imbalance, operational threshold choice and the limitations of evaluating on a single random split. The dashboard is not a production API.

## References
- IBM Telco Customer Churn dataset (public dataset mirrored on Kaggle): https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- scikit-learn Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- scikit-learn model selection: https://scikit-learn.org/stable/modules/grid_search.html
- Streamlit documentation: https://docs.streamlit.io/
