import sagemaker
from sagemaker.xgboost import XGBoost
import boto3

# === Configuration ===
role = "arn:aws:iam::607007849765:role/fraud-sagemaker-role"  # Must be a role, not a user
region = "ca-central-1"
s3_input = "s3://fraud-detection-anna/creditcard_2023.csv"
output_path = "s3://fraud-detection-anna/fraud-output/"

# === Create explicit SageMaker session ===
boto_session = boto3.Session(region_name=region)
sagemaker_session = sagemaker.Session(
    boto_session=boto_session,
    default_bucket="fraud-detection-anna"
)

# === Define XGBoost estimator ===
xgb_estimator = XGBoost(
    entry_point="train.py",      # Your training script
    source_dir=".",              # Include all local files (utils/)
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="1.7-1",   # Compatible XGBoost version
    output_path=output_path,
    hyperparameters={
        "max_depth": 5,
        "n_estimators": 100,
        "learning_rate": 0.1,
        "objective": "binary:logistic"
    },
    sagemaker_session=sagemaker_session
)

# === Submit training job ===
print("Submitting training job...")
xgb_estimator.fit({"train": s3_input}, wait=True, logs=True)
print("Training job finished")

# === Confirm model artifact location ===
print(f"Model saved to: {xgb_estimator.model_data}")
