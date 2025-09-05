# utils/evaluation.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_classification(y_true, y_pred, y_proba=None, plot_confusion=True):
    """
    Evaluate classification model performance.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        y_proba (array-like, optional): Predicted probabilities for positive class
        plot_confusion (bool, optional): Whether to plot confusion matrix
        
    Returns:
        metrics_dict (dict): Dictionary of metrics
    """
    metrics_dict = {}

    # Basic metrics
    metrics_dict['accuracy'] = accuracy_score(y_true, y_pred)
    metrics_dict['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics_dict['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics_dict['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # ROC AUC
    if y_proba is not None:
        metrics_dict['roc_auc'] = roc_auc_score(y_true, y_proba)
    else:
        metrics_dict['roc_auc'] = None

    # Print classification report
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Confusion matrix
    if plot_confusion:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    return metrics_dict


def top_feature_analysis(model, feature_names, top_n=10):
    """
    Display top feature importances from XGBoost or tree-based model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        top_n (int): Number of top features to show
    """
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feat_imp, palette='viridis')
    plt.title(f'Top {top_n} Feature Importances')
    plt.show()

    return feat_imp
