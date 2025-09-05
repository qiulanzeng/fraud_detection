# fraud_detection

Data is from https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023?resource=download


Folder structure
fraud-detection/
│
├── .github/
│   └── workflows/
│       └── sagemaker-ci-cd.yml   # GitHub Actions workflow
│
├── train.py                      # SageMaker entry point for training
├── inference.py                  # Optional inference logic
├── requirements.txt              # Dependencies
│
├── config/
│   └── config.json               # Hyperparameters
│
├── utils/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_utils.py
│   └── evaluation.py
│
├── scripts/
│   └── sagemaker_train.py        # Script called by GitHub Actions
│
└── notebooks/
    └── experiment.ipynb          # For local experimentation



End-to-End ML Pipeline Using SageMaker

# 1 Local Setup: Install AWS CLI and Configure IAM
Confirm it is working:
    - aws --version

## 1 Create IAM User with Admin Access:
- Go to IAM Console
- Create user [username]
- select Attac policies directly
- Attach AdministratorAccess

## 2 Generate Access Keys:

## 3 Congigure AWS CLI
- In Anaconda prompt or terminal, enter:
    - aws configure
    - Enter:
        - Access key ID
        - Secrete access key
        - Region
        - Output format: json

# 2 Set up Python Environment
- pip install requirements.txt

# 3 Prepare Data
- Go to AWS -> S3 -> Create a bucket
- Upload your training/testing data (CSV, Parquet)

# 4 Create Training Script train.py (SageMaker entry point)

# 5 Use SageMaker
- Go to IAM Console -> roles -> create role
    - Select Trusted Entity: AWS service
    - use case: SageMaker -> Next
- IAM Console -> roles -> click the role created -> Trust relationships -> 
- Attach Permissions Policies
    - AmazonSageMakerFullAccess
    - AmazonS3FullAccess
- Name the Role
    - [Role Name] (eg. SageMakerExecutionRole-FraudDetection) -> Create role
- Copy the Role ARN

# 6 - Add scripts/sagemaker_train.py (Train the model on SageMaker) and copy Role ARN in the file.

# 7. CI/CD using GitHub Actions

- Add sagemaker-ci-cd.yml
- commit to github main

# 7 Monitor Training Job
- Go to AWS -> SageMaker -> Training -> Training Jobs
- View logs, metrics, and status of the job

# 8 Deploy the Model
predictor = sklearn_estimator.deploy(
    instance_type='ml.m5.large',
    initial_instance_count=1
)

# 8. Use the Endpoint for Inference
import pandas as pd

test_data = pd.read_csv("test.csv").drop("target", axis=1)
result = predictor.predict(test_data.values)
print(result)

# 9. Clean up Resources
predictor.delete_endpoint()
