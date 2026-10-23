"""
Model evaluation, comparison, and cross-validation utilities for AgriPredict AI.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from utils.metrics import compute_classification_scorecard, compute_regression_scorecard


def compare_classification_models(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Constructs a comparative evaluation table across multiple classification algorithms.
    """
    rows = []
    for model_name, metrics in results_dict.items():
        rows.append({
            "Algorithm": model_name,
            "Accuracy": metrics.get("accuracy", 0.0),
            "F1 (Weighted)": metrics.get("f1_weighted", 0.0),
            "Precision (Weighted)": metrics.get("precision_weighted", 0.0),
            "Recall (Weighted)": metrics.get("recall_weighted", 0.0),
            "F1 (Macro)": metrics.get("f1_macro", 0.0)
        })
    df = pd.DataFrame(rows)
    return df.sort_values(by="F1 (Weighted)", ascending=False)


def compare_regression_models(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Constructs a comparative evaluation table across multiple regression algorithms.
    """
    rows = []
    for model_name, metrics in results_dict.items():
        rows.append({
            "Algorithm": model_name,
            "R² Score": metrics.get("r2_score", 0.0),
            "RMSE": metrics.get("rmse", 0.0),
            "MAE": metrics.get("mae", 0.0)
        })
    df = pd.DataFrame(rows)
    return df.sort_values(by="R² Score", ascending=False)
