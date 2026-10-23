"""
Soil nutrient-based crop recommendation machine learning pipeline for AgriPredict AI.
Implements multi-class classification, feature importance analysis, model persistence,
and real-time crop suitability inference.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from config.settings import SAVED_MODELS_DIR
from utils.metrics import compute_classification_scorecard


FEATURE_COLUMNS = ["N", "P", "K", "ph", "EC", "S", "Cu", "Fe", "Mn", "Zn", "B"]
TARGET_COLUMN = "label"
MODEL_FILE = SAVED_MODELS_DIR / "crop_recommendation_pipeline.joblib"


def get_feature_ranges(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Computes valid min, max, and median ranges for each feature based on training data.
    """
    ranges = {}
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            ranges[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "default": float(df[col].median())
            }
    return ranges


def train_and_evaluate_crop_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Trains and compares Random Forest, Decision Tree, and Gradient Boosting classifiers
    using an 80/20 stratified split. Persists the top-performing pipeline.
    """
    if df.empty or TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain target column '{TARGET_COLUMN}'.")

    # Clean missing/invalid values
    data = df.dropna(subset=[TARGET_COLUMN] + [c for c in FEATURE_COLUMNS if c in df.columns]).copy()

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN].str.strip().str.lower()

    classes = np.unique(y)

    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    algorithms = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    eval_results = {}
    trained_pipelines = {}

    for name, clf in algorithms.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        scorecard = compute_classification_scorecard(y_test, y_pred)
        scorecard["classes"] = list(classes)

        # Feature importances
        if hasattr(clf, "feature_importances_"):
            scorecard["feature_importances"] = {
                feat: round(float(imp), 4)
                for feat, imp in zip(FEATURE_COLUMNS, clf.feature_importances_)
            }

        eval_results[name] = scorecard
        trained_pipelines[name] = pipeline

    # Select best model based on weighted F1
    best_model_name = max(eval_results.keys(), key=lambda k: eval_results[k]["f1_weighted"])
    best_pipeline = trained_pipelines[best_model_name]

    # Persist best model
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "classes": list(classes),
        "feature_columns": FEATURE_COLUMNS,
        "scorecard": eval_results[best_model_name]
    }, MODEL_FILE)

    return {
        "best_model_name": best_model_name,
        "results": eval_results,
        "classes": list(classes),
        "feature_columns": FEATURE_COLUMNS
    }


def load_trained_crop_model(df_fallback: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
    """
    Loads saved model artifact from disk, or trains from df_fallback if artifact is absent.
    """
    if MODEL_FILE.exists():
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            pass

    if df_fallback is not None and not df_fallback.empty:
        train_and_evaluate_crop_models(df_fallback)
        if MODEL_FILE.exists():
            return joblib.load(MODEL_FILE)

    return None


def predict_crop_suitability(
    model_artifact: Dict[str, Any],
    input_features: Dict[str, float]
) -> Dict[str, Any]:
    """
    Generates crop recommendation and probability distribution for a given soil sample.
    """
    pipeline = model_artifact["pipeline"]
    feature_cols = model_artifact.get("feature_columns", FEATURE_COLUMNS)

    # Format input row
    row = [[input_features.get(col, 0.0) for col in feature_cols]]
    df_input = pd.DataFrame(row, columns=feature_cols)

    pred = pipeline.predict(df_input)[0]
    result = {"predicted_crop": str(pred).title()}

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(df_input)[0]
        classes = pipeline.named_steps["classifier"].classes_
        prob_dict = {str(c).title(): round(float(p) * 100, 2) for c, p in zip(classes, probs)}
        # Rank by probability
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
        result["probabilities"] = sorted_probs
        result["confidence"] = sorted_probs[str(pred).title()]

    return result
