import sagemaker
# from sagemaker.estimator import Estimator
# import boto3
from sagemaker.xgboost import XGBoost
role = "arn:aws:iam::607007849765:role/fraud-detection-anna"
region = "ca-central-1"
s3_input = "s3://fraud-detection-anna/creditcard_2023.csv"
output_path = "s3://fraud-detection-anna/fraud-output/"


xgb_estimator = XGBoost(
    entry_point="train.py",
    source_dir=".",
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="1.7-1",  # compatible version
    output_path=output_path,
    hyperparameters={
        "max_depth": 5,
        "n_estimators": 100,
        "learning_rate": 0.1,
        "objective": "binary:logistic"
    }
)
print("Submitting training job...")
xgb_estimator.fit({"train": s3_input}, wait=True, logs=True)
print("Training job finished")