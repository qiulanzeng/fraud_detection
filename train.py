import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from utils.data_preprocessing import load_and_preprocess
from utils.model_utils import save_model
from experiments.sagemaker_experiment import log_metrics, log_plot
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    input_path = os.environ.get("SM_CHANNEL_TRAIN", "data/creditcard_2023.csv")
    output_path = os.environ.get("SM_MODEL_DIR", "model/")
    run_name = os.environ.get("SM_RUN_NAME", "fraud-xgboost-run")
    
    X, y = load_and_preprocess(input_path)
    X = X.drop(columns=['id'])
    # Train/Test Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train XGBoost
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc:.4f}")

    # Log metrics to SageMaker Experiments
    metrics = {
        "roc_auc": roc_auc,
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth
    }
    log_metrics(run_name, metrics)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title("Confusion Matrix")
    log_plot(run_name, fig, "confusion_matrix.png")

    # Feature Importance Plot
    fig2, ax2 = plt.subplots(figsize=(12,6))
    xgb.plot_importance(model, max_num_features=20, importance_type='gain', ax=ax2)
    plt.title("Top 20 Feature Importances")
    log_plot(run_name, fig2, "feature_importance.png")

    # Save Model
    save_model(model, output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    main()