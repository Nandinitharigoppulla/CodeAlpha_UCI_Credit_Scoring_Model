import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# ------------------------------
# 1. Load data
# ------------------------------
df = pd.read_csv('dataset/UCI_Credit_Card.csv')
df.drop('ID', axis=1, inplace=True)

# Target
X = df.drop('default.payment.next.month', axis=1)
y = df['default.payment.next.month']

# ------------------------------
# 2. Feature engineering
# ------------------------------
# Payment status columns (PAY_0 to PAY_6)
pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

X['avg_bill'] = X[bill_cols].mean(axis=1)
X['avg_pay'] = X[pay_amt_cols].mean(axis=1)
X['pay_ratio'] = X['avg_pay'] / (X['avg_bill'] + 1e-6)
X['delinquent_count'] = (X[pay_cols] > 0).sum(axis=1)
X['total_bill'] = X[bill_cols].sum(axis=1)
X['total_pay'] = X[pay_amt_cols].sum(axis=1)

# ------------------------------
# 3. Define column types
# ------------------------------
categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
numeric_features = ['LIMIT_BAL', 'AGE', 'avg_bill', 'avg_pay', 'pay_ratio',
                    'delinquent_count', 'total_bill', 'total_pay']
ordinal_features = pay_cols   # keep as is

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ]), categorical_features),
    ('ord', 'passthrough', ordinal_features)
])

# ------------------------------
# 4. Train / test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------
# 5. Build pipeline with GridSearch
# ------------------------------
xgb = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [5, 7],
    'model__learning_rate': [0.05, 0.1],
    'model__subsample': [0.8, 1.0]
}

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb)
])

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best params:", grid.best_params_)

# ------------------------------
# 6. Evaluate
# ------------------------------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc:.4f}")

# ------------------------------
# 7. Save pipeline
# ------------------------------
joblib.dump(best_model, 'models/credit_pipeline.pkl')
print("Pipeline saved as 'models/credit_pipeline.pkl'")