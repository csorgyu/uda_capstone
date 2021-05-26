
import xgboost as xgb
import argparse
import os
import numpy as np
from sklearn.metrics import auc, roc_curve
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
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
    #params = {"objective": "binary:logistic", "max_depth": 3}


    parser.add_argument('--num_boost_round', type=int, default=5, help="Number of boosting rounds")
    parser.add_argument('--max_depth', type=int, default=3, help="Maximum depth of the trees to be boosted")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate, xgb's eta")
    parser.add_argument('--gamma', type=float, default=0.1, help="Minimum loss reduction")
    parser.add_argument('--reg_lambda', type=float, default=0.1, help="L2 regularization term on weights")
    parser.add_argument('--scale_pos_weight', type=float, default=1.0, help="Balancing of positive and negative weights")

    args = parser.parse_args()
    params = {"scale_pos_weight": np.float(args.scale_pos_weight),
              "reg_lambda": np.float(args.reg_lambda),
              "gamma": np.float(args.gamma),
              "learning_rate": np.float(args.learning_rate),
              "max_depth": np.int(args.max_depth),
              "num_boost_round": np.int(args.num_boost_round)}

    run.log("scale_pos_weigth:", np.float(args.scale_pos_weight))
    run.log("Max depth:", np.int(args.max_depth))
    run.log("Learning rate:", np.int(args.learning_rate))
    run.log("Boosting rounds:", np.int(args.num_boost_round))
    run.log("Gamma (minimum loss reduction):", np.int(args.learning_rate))
    run.log("Lambda (L2 regularization):", np.int(args.reg_lambda))

    xgb_model = xgb.XGBClassifier(objective="binary:logistic",
                        params=params,
                        nfold=5,
                        metrics="auc",
                        as_pandas=True,
                        seed=123)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    xgb_auc = auc(fpr, tpr)

    run.log("AUC", np.float(xgb_auc))
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=xgb_model, filename='outputs/model.pkl')

if __name__ == '__main__':
    main()