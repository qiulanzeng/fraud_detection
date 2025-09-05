import os
import joblib

def save_model(model, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))

def load_model(model_path):
    return joblib.load(model_path)