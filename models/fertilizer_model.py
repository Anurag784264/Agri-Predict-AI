"""
Fertilizer recommendation machine learning pipeline for AgriPredict AI.
Handles mixed numerical and categorical features using ColumnTransformer,
compares multi-class classification models, and provides real-time fertilizer prediction.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from config.settings import SAVED_MODELS_DIR
from utils.metrics import compute_classification_scorecard


NUMERICAL_FEATURES = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
CATEGORICAL_FEATURES = ["Soil Type", "Crop Type"]
TARGET_COLUMN = "Fertilizer Name"
MODEL_FILE = SAVED_MODELS_DIR / "fertilizer_pipeline.joblib"


def get_fertilizer_categories(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Extracts distinct categories for Soil Type and Crop Type dropdowns.
    """
    categories = {}
    if "Soil Type" in df.columns:
        categories["soil_types"] = sorted(df["Soil Type"].dropna().unique().tolist())
    if "Crop Type" in df.columns:
        categories["crop_types"] = sorted(df["Crop Type"].dropna().unique().tolist())
    return categories


def train_and_evaluate_fertilizer_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Trains and compares Random Forest, Decision Tree, and Gradient Boosting models
    with full preprocessing pipeline (OneHotEncoder + StandardScaler).
    """
    if df.empty or TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain target '{TARGET_COLUMN}'.")

    # Clean column names
    data = df.copy()
    data.columns = data.columns.str.strip()

    req_cols = [TARGET_COLUMN] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    data = data.dropna(subset=[c for c in req_cols if c in data.columns])

    X = data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET_COLUMN].astype(str).str.strip()

    classes = np.unique(y)

    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )

    algorithms = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    eval_results = {}
    trained_pipelines = {}

    for name, clf in algorithms.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        scorecard = compute_classification_scorecard(y_test, y_pred)
        scorecard["classes"] = list(classes)

        eval_results[name] = scorecard
        trained_pipelines[name] = pipeline

    best_model_name = max(eval_results.keys(), key=lambda k: eval_results[k]["f1_weighted"])
    best_pipeline = trained_pipelines[best_model_name]

    # Persist model artifact
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "classes": list(classes),
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "scorecard": eval_results[best_model_name]
    }, MODEL_FILE)

    return {
        "best_model_name": best_model_name,
        "results": eval_results,
        "classes": list(classes)
    }


def load_trained_fertilizer_model(df_fallback: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
    """
    Loads saved model artifact from disk, or trains from df_fallback if artifact is absent.
    """
    if MODEL_FILE.exists():
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            pass

    if df_fallback is not None and not df_fallback.empty:
        train_and_evaluate_fertilizer_models(df_fallback)
        if MODEL_FILE.exists():
            return joblib.load(MODEL_FILE)

    return None


def predict_fertilizer(
    model_artifact: Dict[str, Any],
    input_features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes inference for user-provided environmental and soil conditions.
    """
    pipeline = model_artifact["pipeline"]
    num_cols = model_artifact.get("numerical_features", NUMERICAL_FEATURES)
    cat_cols = model_artifact.get("categorical_features", CATEGORICAL_FEATURES)

    row_dict = {}
    for col in num_cols:
        row_dict[col] = [float(input_features.get(col, 0.0))]
    for col in cat_cols:
        row_dict[col] = [str(input_features.get(col, ""))]

    df_input = pd.DataFrame(row_dict)
    pred = pipeline.predict(df_input)[0]

    result = {"predicted_fertilizer": str(pred)}

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(df_input)[0]
        classes = pipeline.named_steps["classifier"].classes_
        prob_dict = {str(c): round(float(p) * 100, 2) for c, p in zip(classes, probs)}
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
        result["probabilities"] = sorted_probs
        result["confidence"] = sorted_probs.get(str(pred), 0.0)

    return result
