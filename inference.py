# inference.py

import pandas as pd
import joblib
import os

# Optional: import preprocessing utilities
from utils.data_preprocessing import preprocess_new_data

class FraudPredictor:
    """
    Class to load a trained model and make predictions on new data.
    """

    def __init__(self, model_path: str, preprocessor_path: str = None):
        """
        Args:
            model_path (str): Path to the saved trained model (joblib or pickle)
            preprocessor_path (str, optional): Path to saved preprocessor for new data
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)
        
        if preprocessor_path:
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
            self.preprocessor = joblib.load(preprocessor_path)
        else:
            self.preprocessor = None

    def predict(self, df: pd.DataFrame):
        """
        Predict fraud probabilities and labels for new data.
        
        Args:
            df (pd.DataFrame): New data, same features as training (before target)
        
        Returns:
            pd.DataFrame: Input data with added 'pred_label' and 'pred_proba' columns
        """
        # Preprocess if preprocessor is provided
        if self.preprocessor:
            X = self.preprocessor.transform(df)
        else:
            X = df.values  # assume df is already numeric and clean

        # Make predictions
        pred_proba = self.model.predict_proba(X)[:, 1]  # probability of fraud
        pred_label = self.model.predict(X)

        result = df.copy()
        result['pred_label'] = pred_label
        result['pred_proba'] = pred_proba

        return result


# Example usage
if __name__ == "__main__":
    # Paths
    MODEL_PATH = "models/xgb_fraud_model.joblib"
    PREPROCESSOR_PATH = "models/preprocessor.joblib"  # optional

    # Load predictor
    predictor = FraudPredictor(model_path=MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH)

    # Load new data
    new_data = pd.read_csv("data/new_transactions.csv")

    # Predict
    predictions = predictor.predict(new_data)
    print(predictions.head())
