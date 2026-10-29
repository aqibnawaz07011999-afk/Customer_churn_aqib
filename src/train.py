"""Reproducible churn model training. Run: python -m src.train --data data/Telco-Customer-Churn.csv"""
import argparse, json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def load_data(path):
    df=pd.read_csv(path)
    if 'Churn' not in df: raise ValueError('Dataset must contain Churn target')
    df=df.drop(columns=['customerID'],errors='ignore').drop_duplicates()
    df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
    df=df.dropna(subset=['Churn'])
    y=df.pop('Churn').astype(str).str.strip().str.lower().map({'yes':1,'no':0})
    if y.isna().any(): raise ValueError('Churn must be Yes/No')
    return df,y

def make_preprocessor(X):
    numeric=X.select_dtypes(include=np.number).columns.tolist()
    categorical=[c for c in X if c not in numeric]
    return ColumnTransformer([
        ('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),numeric),
        ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore'))]),categorical)
    ])

def main(data_path):
    (ROOT/'outputs').mkdir(exist_ok=True); (ROOT/'models').mkdir(exist_ok=True)
    X,y=load_data(data_path)
    Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    prep=make_preprocessor(Xtrain)
    ratio=(ytrain==0).sum()/max((ytrain==1).sum(),1)
    candidates={
        'Logistic Regression':LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42),
        'Random Forest':RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=-1),
        'SVM':SVC(probability=True,class_weight='balanced',random_state=42),
        'XGBoost':XGBClassifier(n_estimators=200,eval_metric='logloss',scale_pos_weight=ratio,random_state=42,n_jobs=2)
    }
    records=[]; fitted={}
    for name,est in candidates.items():
        pipe=Pipeline([('preprocess',prep),('model',est)])
        t=time.time()
        scores=cross_validate(pipe,Xtrain,ytrain,cv=cv,scoring={'auc':'roc_auc','f1':'f1','recall':'recall'},n_jobs=1)
        pipe.fit(Xtrain,ytrain)
        fitted[name]=pipe
        records.append({'model':name,'cv_auc':float(scores['test_auc'].mean()),'cv_f1':float(scores['test_f1'].mean()),'cv_recall':float(scores['test_recall'].mean()),'train_auc':float(roc_auc_score(ytrain,pipe.predict_proba(Xtrain)[:,1])),'seconds':round(time.time()-t,1)})
        print(name,records[-1],flush=True)
    pd.DataFrame(records).to_csv(ROOT/'outputs'/'model_comparison.csv',index=False)
    # Tune two models on TRAIN ONLY. The test set is never used for model selection.
    spaces={
      'Logistic Regression':{'model__C':[.01,.1,1,10],'model__solver':['lbfgs']},
      'Random Forest':{'model__n_estimators':[100,200,300],'model__max_depth':[None,5,10,20],'model__min_samples_leaf':[1,2,4]}
    }
    tuned=[]
    for name,space in spaces.items():
        search=RandomizedSearchCV(Pipeline([('preprocess',prep),('model',candidates[name])]),space,n_iter=min(10,int(np.prod([len(v) for v in space.values()]))),scoring='roc_auc',cv=cv,random_state=42,n_jobs=-1,refit=True)
        search.fit(Xtrain,ytrain)
        tuned.append({'model':name,'cv_auc':float(search.best_score_),'params':search.best_params_})
        fitted[name+' (tuned)']=search.best_estimator_
    (ROOT/'outputs'/'tuning.json').write_text(json.dumps(tuned,indent=2))
    ranking={r['model']:r['cv_auc'] for r in records}
    ranking.update({t['model']+' (tuned)':t['cv_auc'] for t in tuned})
    best_name=max(ranking,key=ranking.get); best=fitted[best_name]
    # One final untouched test evaluation, only after all model choices.
    pred=best.predict(Xtest); proba=best.predict_proba(Xtest)[:,1]
    result={'selected_model':best_name,'selection_cv_auc':ranking[best_name],'test_accuracy':accuracy_score(ytest,pred),'test_precision':precision_score(ytest,pred,zero_division=0),'test_recall':recall_score(ytest,pred,zero_division=0),'test_f1':f1_score(ytest,pred,zero_division=0),'test_roc_auc':roc_auc_score(ytest,proba),'confusion_matrix':confusion_matrix(ytest,pred).tolist(),'classification_report':classification_report(ytest,pred,output_dict=True)}
    (ROOT/'outputs'/'final_metrics.json').write_text(json.dumps(result,indent=2))
    joblib.dump(best,ROOT/'models'/'best_pipeline.joblib')
    joblib.dump({'columns':X.columns.tolist(),'categorical_values':{c:sorted(X[c].dropna().astype(str).unique().tolist()) for c in X.select_dtypes(exclude=np.number)},'numeric_defaults':{c:float(X[c].median()) for c in X.select_dtypes(include=np.number)}},ROOT/'models'/'schema.joblib')
    imp=permutation_importance(best,Xtest,ytest,scoring='roc_auc',n_repeats=5,random_state=42,n_jobs=1)
    importance=pd.DataFrame({'feature':Xtest.columns,'mean_auc_drop':imp.importances_mean}).sort_values('mean_auc_drop',ascending=False)
    importance.to_csv(ROOT/'outputs'/'feature_importance.csv',index=False)
    print('FINAL',json.dumps({k:v for k,v in result.items() if k!='classification_report'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default=str(ROOT/'data'/'Telco-Customer-Churn.csv')); args=p.parse_args(); main(args.data)
