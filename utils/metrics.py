"""
Performance metrics and KPI computation utilities for AgriPredict AI.
Calculates platform-wide agricultural indicators and ML evaluation scorecards.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)


def format_compact_number(val: float | int) -> str:
    """
    Formats large numbers into compact human-readable strings (e.g. 1.2M, 45.6K).
    """
    if pd.isnull(val):
        return "N/A"
    abs_val = abs(val)
    if abs_val >= 1e9:
        return f"{val / 1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"{val / 1e6:.2f}M"
    elif abs_val >= 1e3:
        return f"{val / 1e3:.2f}K"
    elif isinstance(val, int) or val.is_integer():
        return f"{int(val):,}"
    else:
        return f"{val:.2f}"


def format_rupees(val: float | int) -> str:
    """
    Formats monetary values in Indian Rupees (₹).
    """
    if pd.isnull(val):
        return "N/A"
    return f"₹ {val:,.2f}"


def compute_platform_kpis(
    df_crop: pd.DataFrame,
    df_fert: pd.DataFrame,
    df_market: pd.DataFrame,
    df_soil: pd.DataFrame,
    weather_city_count: int = 8
) -> Dict[str, Any]:
    """
    Computes genuine platform KPIs strictly derived from real loaded datasets.
    Does not fabricate any figures.
    """
    # Total records across loaded datasets
    total_records = len(df_crop) + len(df_fert) + len(df_market) + len(df_soil)

    # Crop production metrics
    crop_count = int(df_crop["Crop"].nunique()) if not df_crop.empty and "Crop" in df_crop.columns else 0
    state_count = int(df_crop["State_Name"].nunique()) if not df_crop.empty and "State_Name" in df_crop.columns else 0
    district_count = int(df_crop["District_Name"].nunique()) if not df_crop.empty and "District_Name" in df_crop.columns else 0

    total_production = float(df_crop["Production"].sum()) if not df_crop.empty and "Production" in df_crop.columns else 0.0
    total_area = float(df_crop["Area"].sum()) if not df_crop.empty and "Area" in df_crop.columns else 0.0
    avg_yield = round(total_production / total_area, 2) if total_area > 0 else 0.0

    # Market metrics
    commodity_count = int(df_market["crop"].nunique()) if not df_market.empty and "crop" in df_market.columns else 0
    avg_modal_price = float(df_market["modal_rs_per_quintal"].mean()) if not df_market.empty and "modal_rs_per_quintal" in df_market.columns else 0.0

    # Fertilizer & Soil metrics
    fert_count = int(df_fert["Fertilizer Name"].nunique()) if not df_fert.empty and "Fertilizer Name" in df_fert.columns else 0
    soil_crops = int(df_soil["label"].nunique()) if not df_soil.empty and "label" in df_soil.columns else 0

    return {
        "total_records": total_records,
        "total_records_formatted": format_compact_number(total_records),
        "crop_count": crop_count,
        "state_count": state_count,
        "district_count": district_count,
        "total_production_tonnes": total_production,
        "total_production_formatted": f"{format_compact_number(total_production)} Tonnes",
        "total_area_hectares": total_area,
        "total_area_formatted": f"{format_compact_number(total_area)} Ha",
        "avg_yield": avg_yield,
        "commodity_count": commodity_count,
        "avg_modal_price_formatted": format_rupees(avg_modal_price),
        "fert_count": fert_count,
        "soil_crops_count": soil_crops,
        "weather_cities": weather_city_count
    }


def compute_classification_scorecard(y_true, y_pred) -> Dict[str, Any]:
    """
    Computes rigorous classification performance metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_macro": round(f1_m, 4),
        "f1_weighted": round(f1_w, 4),
        "confusion_matrix": cm
    }


def compute_regression_scorecard(y_true, y_pred) -> Dict[str, Any]:
    """
    Computes regression evaluation metrics: MAE, RMSE, and R-squared.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4)
    }
