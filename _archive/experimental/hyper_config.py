from azureml.widgets import RunDetails
from azureml.train.sklearn import SKLearn
from azureml.train.hyperdrive import choice
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.policy import BanditPolicy
from azureml.train.hyperdrive.policy import TruncationSelectionPolicy
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.parameter_expressions import uniform
from azureml.core import ScriptRunConfig
import os


# Specify parameter sampler
# https://xgboost.readthedocs.io/en/latest/parameter.html
ps = {
    '--num_boost_round': choice(5, 10, 20, 50),
    '--max_depth': choice(3, 5, 8),
    '--learning_rate': choice(0.001, 0.005, 0.01, 0.05),
    '--gamma': choice(0,1,2),
    '--reg_lambda': choice(0.1, 1, 2, 5),
    '--scale_pos_weight': choice(1, 2)
}
samp = RandomParameterSampling(parameter_space=ps)

# Specify a Policy
policy = TruncationSelectionPolicy(truncation_percentage=50)  # BanditPolicy(slack_factor=0.1)

if "training" not in os.listdir():
    os.mkdir("./training")


src = ScriptRunConfig(source_directory=script_folder,
                      script="hyper_tuning.py",
                      arguments=['--num_boost_round', 3.0, '--max_depth', 3, '--learning_rate', 0.001, '--gamma', 0, '--reg_lambda', 1, '--scale_pos_weight',1],
                      compute_target=compute_target,
                      environment=tf_env)
# Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.
hyperdrive_config = HyperDriveConfig(run_config=src,
                                     hyperparameter_sampling=samp,
                                     primary_metric_name="AUC",
                                     primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                     max_total_runs=30,
                                     max_concurrent_runs=4)