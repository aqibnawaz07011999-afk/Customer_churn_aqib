"""Generate 8 EDA visualizations from original data."""
from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
ROOT=Path(__file__).resolve().parents[1]
def main(path):
    df=pd.read_csv(path); df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
    out=ROOT/'outputs'/'eda';out.mkdir(parents=True,exist_ok=True)
    charts=[('Churn','count'),('gender','count'),('Contract','count'),('InternetService','count'),('PaymentMethod','count'),('tenure','hist'),('MonthlyCharges','hist'),('TotalCharges','box')]
    for i,(col,kind) in enumerate(charts,1):
        plt.figure(figsize=(9,5))
        if kind=='count': sns.countplot(data=df,x=col,hue='Churn' if col!='Churn' else None)
        elif kind=='hist': sns.histplot(data=df,x=col,hue='Churn',bins=25)
        else: sns.boxplot(data=df,x='Churn',y=col)
        plt.xticks(rotation=25,ha='right');plt.tight_layout();plt.savefig(out/f'{i:02d}_{col}.png',dpi=140);plt.close()
    df.describe(include='all').to_csv(ROOT/'outputs'/'dataset_summary.csv')
    print('Saved 8 plots in',out)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(ROOT/'data'/'Telco-Customer-Churn.csv'))
    main(p.parse_args().data)
