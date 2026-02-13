import pandas as pd
import Model_comparison_smote as mc
import numpy as np
from xgboost import XGBClassifier
import joblib

param_grid_xgb = {
    "model__max_depth": [3, 4, 5],
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.8, 0.9],
    "model__colsample_bytree": [0.8, 0.9],
    "model__scale_pos_weight": [1, 3, 5] 
}

df = pd.read_csv('Telco_Churn.csv')
df = df[df['TotalCharges'] != ' ']
df['TotalCharges'] = df['TotalCharges'].astype(dtype='float64', errors='ignore')
df = df.drop(columns='customerID')
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
features = df.drop(columns='Churn')
target = df['Churn']


columns = mc.get_columns_by_dtype(features)
best_model = mc.best_model_search(
    features=features, 
    target=target,
    model=XGBClassifier(), 
    search_params=param_grid_xgb,
    cols=columns
)
joblib.dump(best_model, 'best_model.pkl')


