
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import auc, roc_curve

from sklearn.model_selection import train_test_split
FILE_PATH = "../data/heart_failure_clinical_records_dataset.csv"
df = pd.read_csv(FILE_PATH, sep=',')
print(df.head())

print(df.info())
#['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes','ejection_fraction', 'high_blood_pressure',
# #'platelets','serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
# #'DEATH_EVENT']
print(df.columns)
ax = sns.boxplot(data=df, x='DEATH_EVENT', y='age') # ~40-100

ax = sns.boxplot(data=df, x='DEATH_EVENT', y='anaemia') #0-1
df.anaemia.describe()

ax = sns.boxplot(data=df, x='DEATH_EVENT', y='creatinine_phosphokinase') #0-8000, log?

df.diabetes.describe() #0-1
df.loc[:, ['DEATH_EVENT','diabetes']].groupby("DEATH_EVENT").sum().plot(kind="bar")

df.ejection_fraction.describe()

ax = sns.boxplot(data=df, x='DEATH_EVENT', y='ejection_fraction') #0-80

df.high_blood_pressure.describe() #0-1

df.loc[:, ['DEATH_EVENT','high_blood_pressure']].groupby("high_blood_pressure").sum().plot(kind="bar")



ax = sns.boxplot(data=df, x='DEATH_EVENT', y='ejection_fraction')

df.platelets.describe()

ax = sns.boxplot(data=df, x='DEATH_EVENT', y='platelets') #0-800000, outliers

df.serum_creatinine.describe()
sns.reset_orig()
ax = sns.boxplot(data=df, x='DEATH_EVENT', y='serum_creatinine') #1.3 - 9.4, outliers

df.serum_sodium.describe()

ax = sns.boxplot(data=df, x='DEATH_EVENT', y='serum_sodium') #115-145


df.loc[:, ['DEATH_EVENT','sex']].groupby( "sex").sum().plot(kind="bar")

plt.clf()
df.loc[:, ['DEATH_EVENT','smoking']].groupby( "smoking").sum().plot(kind="bar")

df.time.describe() #followup period 4-285

plt.show()

df.columns = "DEATH_EVENT"

# for Random forest
X_train, X_test, y_train, y_test = train_test_split(df.loc[:, ~df.columns.isin(["DEATH_EVENT"])], df.loc[:, "DEATH_EVENT"], test_size=0.3, random_state=123 )
print(X_train.shape)
print(X_test.shape)

rf_model = RandomForestClassifier(n_estimators = 100, max_depth=3, random_state=123)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
rf_auc = auc(fpr, tpr)
#0.73

#for XGBoost CV
X, y = df.loc[:, ~df.columns.isin(["DEATH_EVENT"])], df.loc[:, "DEATH_EVENT"]
heart_dmatrix = xgb.DMatrix(data=X, label=y)
params={"objective": "binary:logistic", "max_depth": 3}

cv_results = xgb.cv(dtrain=heart_dmatrix, params=params,
                  nfold=5, num_boost_round=5,
                  metrics="auc", as_pandas=True, seed=123)

print(cv_results)

xgb_auc = cv_results["test-auc-mean"].iloc[-1]
#0.88




