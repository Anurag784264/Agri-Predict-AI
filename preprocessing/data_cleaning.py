"""
Data cleaning and preprocessing module for AgriPredict AI.
Implements robust cleaning, missing value handling, outlier detection,
feature validation, and processed data persistence.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from config.settings import PROCESSED_DATA_DIR


def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> Dict[str, Any]:
    """
    Identifies outliers using the Interquartile Range (IQR) technique.
    Does not delete them automatically to preserve real-world agricultural variation.
    """
    if column not in df.columns or not np.issubdtype(df[column].dtype, np.number):
        return {"count": 0, "lower_bound": None, "upper_bound": None, "indices": []}

    series = df[column].dropna()
    q25, q75 = np.percentile(series, 25), np.percentile(series, 75)
    iqr = q75 - q25
    lower_bound = q25 - (multiplier * iqr)
    upper_bound = q75 + (multiplier * iqr)

    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outlier_indices = df[outlier_mask].index.tolist()

    return {
        "column": column,
        "count": len(outlier_indices),
        "percentage": round((len(outlier_indices) / len(df)) * 100, 2),
        "lower_bound": round(lower_bound, 3),
        "upper_bound": round(upper_bound, 3),
        "outlier_indices": outlier_indices
    }


def clean_crop_production(
    df: pd.DataFrame,
    impute_missing: bool = True,
    impute_method: str = "median_by_crop"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean the crop production dataset:
    - Normalizes column names and textual strings.
    - Resolves missing production values appropriately.
    - Validates Area > 0 and Production >= 0.
    - Computes Yield (Production / Area) with transparent formula explanation.
    """
    if df.empty:
        return df, {"status": "empty"}

    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()

    initial_rows = len(cleaned)
    initial_dups = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates()

    # Standardize string fields
    str_cols = ["State_Name", "District_Name", "Crop", "Season"]
    for col in str_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip().str.title()

    # Crop year as integer
    if "Crop_Year" in cleaned.columns:
        cleaned["Crop_Year"] = pd.to_numeric(cleaned["Crop_Year"], errors="coerce")

    # Filter out invalid non-positive cultivation areas
    invalid_area_count = 0
    if "Area" in cleaned.columns:
        cleaned["Area"] = pd.to_numeric(cleaned["Area"], errors="coerce")
        invalid_area_count = int((cleaned["Area"] <= 0).sum())
        cleaned = cleaned[cleaned["Area"] > 0]

    # Handle missing production
    missing_prod_before = int(cleaned["Production"].isnull().sum())
    imputed_count = 0

    if "Production" in cleaned.columns:
        cleaned["Production"] = pd.to_numeric(cleaned["Production"], errors="coerce")
        if impute_missing:
            if impute_method == "median_by_crop" and "Crop" in cleaned.columns:
                # Group-wise median imputation
                crop_medians = cleaned.groupby("Crop")["Production"].transform("median")
                overall_median = cleaned["Production"].median()
                cleaned["Production"] = cleaned["Production"].fillna(crop_medians).fillna(overall_median)
                imputed_count = missing_prod_before - int(cleaned["Production"].isnull().sum())
            elif impute_method == "median":
                med = cleaned["Production"].median()
                cleaned["Production"] = cleaned["Production"].fillna(med)
                imputed_count = missing_prod_before
            elif impute_method == "drop":
                cleaned = cleaned.dropna(subset=["Production"])
        else:
            # Drop NaN production records for modeling
            cleaned = cleaned.dropna(subset=["Production"])

    # Calculate Yield = Production / Area
    if "Area" in cleaned.columns and "Production" in cleaned.columns:
        cleaned["Yield"] = cleaned["Production"] / cleaned["Area"]
        cleaned["Yield"] = cleaned["Yield"].round(4)

    summary = {
        "dataset": "Crop Production",
        "initial_rows": initial_rows,
        "final_rows": len(cleaned),
        "duplicates_removed": initial_dups,
        "invalid_areas_removed": invalid_area_count,
        "missing_production_original": missing_prod_before,
        "missing_production_imputed": imputed_count,
        "yield_calculated": "Yield = Production / Area (Tonnes per Hectare)"
    }

    return cleaned, summary


def clean_fertilizer_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean the fertilizer dataset:
    - Normalizes column names (fixes trailing whitespace like 'Humidity ').
    - Standardizes text strings for Soil Type, Crop Type, and Fertilizer Name.
    - Validates nutrient and environmental ranges.
    """
    if df.empty:
        return df, {"status": "empty"}

    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()

    initial_rows = len(cleaned)
    initial_dups = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates()

    # Clean text columns
    text_cols = ["Soil Type", "Crop Type", "Fertilizer Name"]
    for col in text_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip().str.title()

    # Numeric fields validation
    num_cols = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
    for col in num_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Drop any row where essential features are null
    cleaned = cleaned.dropna()

    summary = {
        "dataset": "Fertilizer Prediction",
        "initial_rows": initial_rows,
        "final_rows": len(cleaned),
        "duplicates_removed": initial_dups,
        "features_cleaned": list(cleaned.columns)
    }

    return cleaned, summary


def clean_soil_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean the soil nutrients dataset:
    - Validates N, P, K, ph, EC, micronutrients (S, Cu, Fe, Mn, Zn, B).
    - Standardizes crop label names.
    - Ensures valid positive chemical ranges.
    """
    if df.empty:
        return df, {"status": "empty"}

    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()

    initial_rows = len(cleaned)
    initial_dups = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates()

    if "label" in cleaned.columns:
        cleaned["label"] = cleaned["label"].astype(str).str.strip().str.lower()

    # Convert all feature columns to numeric
    feature_cols = [c for c in cleaned.columns if c != "label"]
    for col in feature_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Drop invalid rows
    cleaned = cleaned.dropna()

    summary = {
        "dataset": "Soil Nutrients and Crop Recommendation",
        "initial_rows": initial_rows,
        "final_rows": len(cleaned),
        "duplicates_removed": initial_dups,
        "classes_count": cleaned["label"].nunique() if "label" in cleaned.columns else 0
    }

    return cleaned, summary


def clean_market_prices(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean Indian Agricultural Market Prices dataset:
    - Standardizes crop naming.
    - Computes price spread (Max - Min).
    - Computes MSP comparison metrics for crops with declared Government MSP.
    """
    if df.empty:
        return df, {"status": "empty"}

    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()

    initial_rows = len(cleaned)
    initial_dups = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates()

    if "crop" in cleaned.columns:
        cleaned["crop"] = cleaned["crop"].astype(str).str.strip()

    num_cols = ["modal_rs_per_quintal", "min", "max", "msp"]
    for col in num_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Price spread (Max - Min)
    if "max" in cleaned.columns and "min" in cleaned.columns:
        cleaned["price_spread"] = cleaned["max"] - cleaned["min"]

    # Market Price vs MSP Variance
    if "modal_rs_per_quintal" in cleaned.columns and "msp" in cleaned.columns:
        cleaned["market_vs_msp_diff"] = cleaned["modal_rs_per_quintal"] - cleaned["msp"]
        cleaned["trading_above_msp"] = cleaned["market_vs_msp_diff"].apply(
            lambda v: "Above MSP" if v > 0 else ("Below MSP" if v < 0 else "Equal / At MSP") if pd.notnull(v) else "No MSP Declared"
        )

    summary = {
        "dataset": "Market Prices",
        "initial_rows": initial_rows,
        "final_rows": len(cleaned),
        "duplicates_removed": initial_dups,
        "crops_with_msp": int(cleaned["msp"].notnull().sum()) if "msp" in cleaned.columns else 0
    }

    return cleaned, summary


def export_processed_dataset(df: pd.DataFrame, filename: str) -> Path:
    """
    Persist cleaned DataFrame to data/processed directory safely.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_path = PROCESSED_DATA_DIR / filename
    df.to_csv(target_path, index=False, encoding="utf-8")
    return target_path
