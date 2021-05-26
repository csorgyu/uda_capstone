
import xgboost as xgb
import argparse
import os
import numpy as np
from sklearn.metrics import auc, roc_curve
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.core.dataset import Dataset





def main():
    parser = argparse.ArgumentParser()
    #params = {"objective": "binary:logistic", "max_depth": 3}
    run = Run.get_context()

    parser.add_argument('--num_boost_round', type=int, default=5, help="Number of boosting rounds")
    parser.add_argument('--max_depth', type=int, default=3, help="Maximum depth of the trees to be boosted")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate, xgb's eta")
    parser.add_argument('--gamma', type=float, default=0.1, help="Minimum loss reduction")
    parser.add_argument('--reg_lambda', type=float, default=0.1, help="L2 regularization term on weights")
    parser.add_argument('--scale_pos_weight', type=float, default=1.0, help="Balancing of positive and negative weights")

    args = parser.parse_args()

    workspace = run.experiment.workspace

    key = "heart-failure" #"Heart failure"
    description_text = "Heart failure dataset for udacity capstone"

    if key in workspace.datasets.keys():
        found = True
        dataset = Dataset.get_by_name(workspace, name='heart-failure')
        
    df = dataset.to_pandas_dataframe()
    X, y = df.loc[:, ~df.columns.isin(["DEATH_EVENT"])], df.loc[:, "DEATH_EVENT"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


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
                        n_estimators = np.int(args.num_boost_round),
                        max_depth = np.int(args.max_depth),
                        learning_rate=np.float(args.learning_rate),
                        gamma= np.float(args.gamma),
                        reg_lambda=np.float(args.reg_lambda),
                        scale_pos_weight=np.float(args.scale_pos_weight),
                        random_state=123)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    xgb_auc = auc(fpr, tpr)

    run.log("AUC", np.float(xgb_auc))
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=xgb_model, filename='outputs/model.pkl')

if __name__ == '__main__':
    main()