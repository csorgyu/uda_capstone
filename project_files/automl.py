
import logging
import os
import csv

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import pkg_resources

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.core.dataset import Dataset




from azureml.pipeline.steps import AutoMLStep


from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)


ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')

# Choose a name for the run history container in the workspace.
# NOTE: update these to match your existing experiment name
experiment_name = 'bankmarketing-experiment-csorgyu'
project_folder = './real-time-project'

experiment = Experiment(ws, experiment_name)
experiment

##################


# NOTE: update the cluster name to match the existing cluster
# Choose a name for your CPU cluster
amlcompute_cluster_name = "auto-ml"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=amlcompute_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',# for GPU, use "STANDARD_NC6"
                                                           #vm_priority = 'lowpriority', # optional
                                                           min_nodes=4,
                                                           max_nodes=4)
    compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True, min_node_count = 1, timeout_in_minutes = 10)
# For a more detailed view of current AmlCompute status, use get_status().

############################

found = False
key = "heart-failure" #"BankMarketing Dataset"
description_text = "Heart failure dataset for udacity capstone"

if key in ws.datasets.keys():
        found = True
        dataset = ws.datasets[key]

if not found:
        # Create AML Dataset and register it into Workspace
        example_data = "https://www.kaggle.com/andrewmvd/heart-failure-clinical-data/download"
        dataset = Dataset.Tabular.from_delimited_files(example_data)
        #Register Dataset in Workspace
        dataset = dataset.register(workspace=ws,
                                   name=key,
                                   description=description_text)


df = dataset.to_pandas_dataframe()
df.describe()

##############################################
dataset.take(5).to_pandas_dataframe()


########################################################
project_folder='./real'
automl_settings = {
    "experiment_timeout_minutes": 15,
    "max_concurrent_iterations": 4,
    "primary_metric" : 'AUC_weighted'
}
automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="DEATH_EVENT",
                             path = project_folder,
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             model_explainability = True,
                             enable_onnx_compatible_models=True,
                             **automl_settings
                            )

#################

automl_run1 = experiment.submit(automl_config, show_output=True)

####
rns = automl_experiment.get_runs()
next(rns)
#####
from azureml.widgets import RunDetails
widget1 = RunDetails(automl_run1)
widget1.show()
###

best_run = automl_run1.get_best_child()
best_run.get_metrics()
