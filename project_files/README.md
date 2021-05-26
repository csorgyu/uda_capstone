# Heart failure prediction with Azure ML Workspace toolkit
 
The goal of the project is creating a classification model with the 2 primary toolkits Azure ML Workspace offers for a selected dataset.
For my own work I chose the [Heart failure clinical data|https://www.kaggle.com/andrewmvd/heart-failure-clinical-data].

There are 2 major modeling tasks and with deployment tasks included to be delivered
- [ ] Delivering a model with the Auto ML functionality
- [x] Delivering a model created as my own work and optimized with the hyperdrive feature of Azure ML 

The best model needs to be deployed and tested for consumption in both of the cases.

As a preparation for the project I have created a baseline model on my own computer, to manage my expectations about the prediction excercise.
As the dataset is not huge (300 observations in total), this was easily managable on a single worsktation. From this perspective the dataset does not set challenges in dataset sizing, sampling, management perspective. From the other side it can be easily attached to the git repository as it is the case in this project.
More on the dataset can be read in Dataset overview

During baselining I was experimenting with Random Forest and XGBoost models. Random Forest is a good, well interpretable, robust model for non-linear problems, scaling does not impact the model much.
XGBoost is currently a widely used model, can de used with linear base models and tree based too, I was opting for the second. It iterativerly corrects weak predictors through boosting rounds.
I created a hyperdrive configuration for both of these models, and eventually the AutoML best model was an ensemble built on these type of models.


## Project Set Up and Installation
The project assumes the usage of Azure ML Workspace. I was using the lab environment provisioned by udacity, but it does not have any specific settings, that assumes that environment - in fact I was provided different environment in all of the cases, where the dataset and the notebooks, python scripts were usable.

Azre ML workspace provisioning can be done multiple ways, manually from the portal, with ARM template deloyment and from terraform, and it includes several components
* A storage device that serves as a backend for storing data, model reszlts and execution logs
* Azure ML Workspace PaaS service itself
* A keyvault, where secrets and certificates can be stored
* Azure Container Registry for model deployment as a service
* Application Insights for monitoring

The workspace itself uses spark cluster under the hood for compute and inference clusters.

The workspace can be accessed through various API versions, I was personally using the python SDK.

To reproduce my work, the dataset from the ../data folder needs to be registered as a dataset and 4 files need to be uploaded to the the worspace, attached to a compute instance in the following structure:
root
- *hyperparameter_tuning.ipynb*
- *automl.ipynb*
- **SCRIPT**
  - *hyper_tuning_rf.py*
  - *hyper_tuning_xgb.py*

I registered the dataset manually as a tabular dataset, I used the default storage as a datastore. Other datastores can be registered of various types.

## Dataset

### Dataset overview ###
I downloaded the data from the kaggle link provided above and registered it to the workspace. This could be done through code, however that call is slow, so I wanted to focus lab time on AutoML runs and hyperdrive runs.

The dataset itself contains 13 columns, that describe clinical conditions of 300 observed patients and whether they died or not.
The conditions include:
* age: the age of the patient 
* anaemia: decrease of the red blood cells in the hemoglobin (encoded by 0 and 1)
* creatinine_phosphokinase: level of CPK enzyme in the blood (mcg/L, ranges between 23 and 7861)
* diabetes: if the patient has diabetes (encoded by 0 and 1)
* ejection_fraction: the percentage of blood leaving the heart at each contraction (percentage, ranges between 14 and 80) 
* high_blood_pressure: if patient has hypertension (encoded by 0 and 1 )
* platelets: Platelets in the blood (kiloplatelets/mL, ranges between 25,100 and 850,000)
* serum_creatinine: Level of serum creatinine in the blood (mg/dL, ranges between 0.5 and 9.4)
* serum_sodium: Level of serum sodium in the blood (mEq/L, ranges between 113 and 148)
* sex: woman or man (binary, encoded by 0 and 1)
* smoking: If the patient smokes or not (boolean, encoded by 0 and 1)
* time: follow up periods (days, ranges from 4 to 285)

As we can see, the predictors are all numeric, however the ranges, shown above and the distributions shwn in the 2 examples below suggest, that the predictors are not on the same scale and do not show similar distribution at all, so the problem is really a non-linear one by nature.


![image](https://user-images.githubusercontent.com/81808810/119728522-df315200-be73-11eb-84e2-3912c3951113.png)

**Age distribution example**
The age distribution shows, that the patients, who died were slightly older on average compared to those who survived.



![image](https://user-images.githubusercontent.com/81808810/119729924-7b0f8d80-be75-11eb-96e7-743e25c66bfc.png)

**Distribution of the CPK enzyme levels example**
The distribution of the CPK enzyme is really skewed in both of the cases, mean, Q3 quartile are very similar. The patients who survived show slightly more outliers, but the skewness of the CPK enzyme levels for those who died, is bigger.

#### Datasets on portal
![image](https://user-images.githubusercontent.com/81808810/119352400-ef91d300-bca1-11eb-81a3-4a1eaf3ae9e6.png)


### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
#### Hyperdrive access
*TODO*: Explain how you are accessing the data in your workspace.
![image](https://user-images.githubusercontent.com/81808810/119352531-16500980-bca2-11eb-9f67-101d43b88b0d.png)
Example shows hyperdrive script based access

#### Auto ML code access

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### AutoML Run
![image](https://user-images.githubusercontent.com/81808810/119652654-aae37480-be26-11eb-8e3a-bd7c3f3f0910.png)


### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
![image](https://user-images.githubusercontent.com/81808810/119652785-d49c9b80-be26-11eb-801b-8d3024ed13c1.png)

Explanations
![image](https://user-images.githubusercontent.com/81808810/119652882-f39b2d80-be26-11eb-99e4-f169fa528a9d.png)



*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

Widget
![image](https://user-images.githubusercontent.com/81808810/119653166-4c6ac600-be27-11eb-8e51-14127ecc9c87.png)

Metrics
![image](https://user-images.githubusercontent.com/81808810/119653970-3c071b00-be28-11eb-98da-242b71847de5.png)


## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

![image](https://user-images.githubusercontent.com/81808810/119644203-e8430480-be1c-11eb-9950-7d348d5c316d.png)
Multiple runs


### Model01: The Random Forest

#### Training progress
##### Portal view
![image](https://user-images.githubusercontent.com/81808810/119352067-827e3d80-bca1-11eb-8580-3fe832ac8ca0.png)

##### Widget view
![image](https://user-images.githubusercontent.com/81808810/119352190-a93c7400-bca1-11eb-8022-167a105efb70.png)

### Model02: XGB with random parameter search
#### Portal view
![image](https://user-images.githubusercontent.com/81808810/119645118-f9404580-be1d-11eb-9ad6-6f98b5d25bf2.png)

#### Widget view
![image](https://user-images.githubusercontent.com/81808810/119645070-e88fcf80-be1d-11eb-8aec-13b93e652475.png)



#### Best metrics
![image](https://user-images.githubusercontent.com/81808810/119352846-7777dd00-bca2-11eb-8a88-32f869c3bd42.png)

and xgb:
![image](https://user-images.githubusercontent.com/81808810/119645311-31e01f00-be1e-11eb-925b-4d5e8c8f86e5.png)

#### Registered model
![image](https://user-images.githubusercontent.com/81808810/119354246-24068e80-bca4-11eb-86f5-49aee49ae47e.png)

and xgb

![image](https://user-images.githubusercontent.com/81808810/119645724-a0bd7800-be1e-11eb-99f6-a12101b0b78e.png)


#### Deployed model
![image](https://user-images.githubusercontent.com/81808810/119354680-a55e2100-bca4-11eb-92bd-09e340f57ef1.png)





#### Using model
![image](https://user-images.githubusercontent.com/81808810/119356896-3afab000-bca7-11eb-8885-4c1318306f0c.png)

#### Logs
![image](https://user-images.githubusercontent.com/81808810/119357692-2539ba80-bca8-11eb-9cc8-a27c8ae0100b.png)


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.


## Deleting compute for Hyperdrive
![image](https://user-images.githubusercontent.com/81808810/119647826-04e13b80-be21-11eb-87e5-bef8a916b538.png)


## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
