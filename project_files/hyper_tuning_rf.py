
import argparse
import os
import numpy as np
from sklearn.metrics import auc, roc_curve
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

path="https://www.kaggle.com/andrewmvd/heart-failure-clinical-data/download"
#this
ds = TabularDatasetFactory.from_delimited_files(path, validate=True, include_path=False, infer_column_types=True,set_column_types=None, separator=',', header=True, partition_format=None)

df = ds.to_pandas()
X, y = df.loc[:, ~df.columns.isin(["DEATH_EVENT"])], df.loc[:, "DEATH_EVENT"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
run = Run.get_context()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=50, help="Number of trees")
    parser.add_argument('--max_depth', type=int, default=3, help="Maximum depth of the trees to used")
    parser.add_argument('--min_samples_split', type=int, default=2, help="Minimum samples")
    

    args = parser.parse_args()
    params = {"n_estimators": np.int(args.n_estimators),
              "max_depth": np.int(args.reg_lambda),
              "min_samples_split": np.int(args.min_samples_split)
              }

    run.log("n_estimators:", np.int(args.n_estimators))
    run.log("max_depth:", np.int(args.max_depth))
    run.log("min_samples_split:", np.int(args.min_samples_split))
    

    rf_model = RandomForestClassifier(n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        metrics="auc",
                        random_state=123)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    rf_auc = auc(fpr, tpr)

    run.log("AUC", np.float(xgb_auc))
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=xgb_model, filename='outputs/model.pkl')

if __name__ == '__main__':
    main()