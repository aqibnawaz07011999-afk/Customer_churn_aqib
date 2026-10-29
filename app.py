from pathlib import Path
import joblib, pandas as pd, streamlit as st
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Customer Churn Predictor',layout='wide')
st.title('Customer Churn Prediction Dashboard')
st.caption('IBM Telco Customer Churn | ML-based decision support')
model_path=ROOT/'models'/'best_pipeline.joblib'
if not model_path.exists():
    st.warning('Train the model first: python -m src.train --data data/Telco-Customer-Churn.csv')
    st.stop()
model=joblib.load(model_path); schema=joblib.load(ROOT/'models'/'schema.joblib')
with st.form('prediction'):
    cols=st.columns(3); values={}
    for i,c in enumerate(schema['columns']):
        with cols[i%3]:
            if c in schema['categorical_values']:
                values[c]=st.selectbox(c,schema['categorical_values'][c])
            else:
                values[c]=st.number_input(c,value=schema['numeric_defaults'][c])
    submitted=st.form_submit_button('Predict churn risk',type='primary')
if submitted:
    score=float(model.predict_proba(pd.DataFrame([values]))[0,1])
    st.metric('Estimated churn probability',f'{score:.1%}')
    st.progress(score)
    st.write('Predicted class:', 'Likely to churn' if score>=.5 else 'Likely to stay')
st.divider()
metrics=ROOT/'outputs'/'model_comparison.csv'
if metrics.exists():
    st.subheader('Cross-validation model comparison')
    st.dataframe(pd.read_csv(metrics),use_container_width=True)
imp=ROOT/'outputs'/'feature_importance.csv'
if imp.exists():
    st.subheader('Top influential original features')
    st.bar_chart(pd.read_csv(imp).head(10).set_index('feature')['mean_auc_drop'])
