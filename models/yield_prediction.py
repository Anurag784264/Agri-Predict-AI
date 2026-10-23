"""
Crop yield regression modeling pipeline for AgriPredict AI.
Predicts estimated crop yield (Tonnes per Hectare) based on geographic,
temporal, seasonal, and cultivation area parameters.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from config.settings import SAVED_MODELS_DIR
from utils.metrics import compute_regression_scorecard


MODEL_FILE = SAVED_MODELS_DIR / "yield_regression_pipeline.joblib"


def prepare_yield_dataset(df: pd.DataFrame, sample_size: int = 25000) -> pd.DataFrame:
    """
    Cleans and samples crop production data for regression modeling:
    - Filters Area > 0 and Production > 0.
    - Yield = Production / Area.
    - Drops extreme erroneous yields (> 100 Tonnes/Ha) to preserve regression stability.
    - Samples up to sample_size rows for fast, interactive response.
    """
    if df.empty:
        return pd.DataFrame()

    data = df.copy()
    data.columns = data.columns.str.strip()

    req_cols = ["State_Name", "Crop", "Season", "Area", "Production"]
    for c in req_cols:
        if c not in data.columns:
            return pd.DataFrame()

    data = data.dropna(subset=req_cols)
    data["Area"] = pd.to_numeric(data["Area"], errors="coerce")
    data["Production"] = pd.to_numeric(data["Production"], errors="coerce")

    # Filter positive values
    data = data[(data["Area"] > 0) & (data["Production"] > 0)].copy()
    data["Yield"] = data["Production"] / data["Area"]

    # Filter realistic yield range (0.01 to 100 tonnes/ha)
    data = data[(data["Yield"] >= 0.01) & (data["Yield"] <= 100.0)]

    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)

    return data


def train_and_evaluate_yield_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Trains and compares Linear Regression, Decision Tree, and Random Forest regressors.
    """
    data = prepare_yield_dataset(df)
    if data.empty or len(data) < 100:
        raise ValueError("Insufficient valid data samples for yield regression.")

    cat_cols = ["State_Name", "Crop", "Season"]
    num_cols = ["Area"]
    target = "Yield"

    X = data[cat_cols + num_cols]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )

    algorithms = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=12, n_jobs=-1, random_state=42)
    }

    eval_results = {}
    trained_pipelines = {}

    for name, reg in algorithms.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", reg)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        scorecard = compute_regression_scorecard(y_test, y_pred)
        eval_results[name] = scorecard
        trained_pipelines[name] = pipeline

    best_model_name = max(eval_results.keys(), key=lambda k: eval_results[k]["r2_score"])
    best_pipeline = trained_pipelines[best_model_name]

    # Save model artifact
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "scorecard": eval_results[best_model_name],
        "cat_cols": cat_cols,
        "num_cols": num_cols
    }, MODEL_FILE)

    return {
        "best_model_name": best_model_name,
        "results": eval_results,
        "sample_size": len(data)
    }


def load_trained_yield_model(df_fallback: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
    """
    Loads saved yield model from disk or fits from fallback DataFrame.
    """
    if MODEL_FILE.exists():
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            pass

    if df_fallback is not None and not df_fallback.empty:
        train_and_evaluate_yield_models(df_fallback)
        if MODEL_FILE.exists():
            return joblib.load(MODEL_FILE)

    return None


def predict_yield(
    model_artifact: Dict[str, Any],
    state: str,
    crop: str,
    season: str,
    area_ha: float
) -> Dict[str, float]:
    """
    Estimates crop yield per hectare and total anticipated harvest production.
    """
    pipeline = model_artifact["pipeline"]
    input_df = pd.DataFrame([{
        "State_Name": state,
        "Crop": crop,
        "Season": season,
        "Area": float(area_ha)
    }])

    pred_yield = float(pipeline.predict(input_df)[0])
    pred_yield = max(0.01, round(pred_yield, 3))
    pred_production = round(pred_yield * area_ha, 2)

    return {
        "predicted_yield_tonnes_per_ha": pred_yield,
        "predicted_total_production_tonnes": pred_production
    }
