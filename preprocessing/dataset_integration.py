"""
Multi-dataset integration module for AgriPredict AI.
Handles validated merges across heterogeneous agricultural datasets,
canonical entity alignment (crops, geographic entities), and join diagnostics.
"""

from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np

# Canonical mapping for City weather to State and District in Crop Production
CITY_TO_GEO = {
    "Bengaluru": {"state": "Karnataka", "districts": ["Bangalore Urban", "Bangalore", "Bengaluru Urban"]},
    "Bombay": {"state": "Maharashtra", "districts": ["Mumbai", "Mumbai Suburban"]},
    "Delhi": {"state": "Delhi", "districts": ["Delhi"]},
    "Hyderabad": {"state": "Telangana", "districts": ["Hyderabad", "Rangareddi"]},
    "Jaipur": {"state": "Rajasthan", "districts": ["Jaipur"]},
    "Kanpur": {"state": "Uttar Pradesh", "districts": ["Kanpur Nagar", "Kanpur Dehat"]},
    "Nagpur": {"state": "Maharashtra", "districts": ["Nagpur"]},
    "Pune": {"state": "Maharashtra", "districts": ["Pune"]}
}

# Canonical crop name harmonization dictionary
CANONICAL_CROPS = {
    "wheat": "Wheat",
    "rice": "Rice",
    "rice (paddy)": "Rice",
    "paddy": "Rice",
    "maize": "Maize",
    "maize (corn)": "Maize",
    "potato": "Potato",
    "sugarcane": "Sugarcane",
    "cotton": "Cotton",
    "cotton(lint)": "Cotton",
    "barley": "Barley",
    "jowar": "Jowar",
    "jowar (sorghum)": "Jowar",
    "bajra": "Bajra",
    "bajra (pearl millet)": "Bajra",
    "onion": "Onion",
    "tomato": "Tomato",
    "garlic": "Garlic",
    "ginger": "Ginger",
    "turmeric": "Turmeric",
    "ragi": "Ragi",
    "grapes": "Grapes",
    "mango": "Mango",
    "pomegranate": "Pomegranate"
}


def standardize_crop_name(name: Any) -> str:
    """
    Map variations of crop names to their canonical representation.
    """
    if pd.isnull(name):
        return ""
    clean = str(name).strip().lower()
    return CANONICAL_CROPS.get(clean, str(name).strip().title())


def perform_safe_join(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_key: str,
    right_key: str,
    how: str = "inner",
    left_name: str = "Left Dataset",
    right_name: str = "Right Dataset"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes a validated join between two DataFrames with integrity checking
    to prevent unintended row multiplication and data corruption.
    """
    if df_left.empty or df_right.empty:
        return pd.DataFrame(), {
            "status": "error",
            "message": "One or both DataFrames are empty. Join cannot proceed."
        }

    if left_key not in df_left.columns:
        return pd.DataFrame(), {
            "status": "error",
            "message": f"Join key '{left_key}' not found in {left_name}."
        }

    if right_key not in df_right.columns:
        return pd.DataFrame(), {
            "status": "error",
            "message": f"Join key '{right_key}' not found in {right_name}."
        }

    # Inspect key cardinality
    left_keys_series = df_left[left_key].dropna()
    right_keys_series = df_right[right_key].dropna()

    left_unique_count = left_keys_series.nunique()
    right_unique_count = right_keys_series.nunique()
    left_has_dups = len(left_keys_series) > left_unique_count
    right_has_dups = len(right_keys_series) > right_unique_count

    # Determine cardinality warning
    is_many_to_many = left_has_dups and right_has_dups
    cardinality = "Many-to-Many" if is_many_to_many else ("Many-to-One" if left_has_dups else ("One-to-Many" if right_has_dups else "One-to-One"))

    # Overlapping unique keys
    overlap_keys = set(left_keys_series.unique()).intersection(set(right_keys_series.unique()))
    overlap_count = len(overlap_keys)

    # Perform the merge
    merged = pd.merge(
        df_left,
        df_right,
        left_on=left_key,
        right_on=right_key,
        how=how,
        suffixes=(f"_{left_name[:4].lower()}", f"_{right_name[:4].lower()}")
    )

    diagnostics = {
        "status": "success",
        "left_name": left_name,
        "right_name": right_name,
        "left_rows": len(df_left),
        "right_rows": len(df_right),
        "merged_rows": len(merged),
        "join_type": how,
        "left_key": left_key,
        "right_key": right_key,
        "overlap_keys_count": overlap_count,
        "left_unique_keys": left_unique_count,
        "right_unique_keys": right_unique_count,
        "cardinality": cardinality,
        "many_to_many_warning": is_many_to_many,
        "sample_matching_keys": list(overlap_keys)[:10]
    }

    return merged, diagnostics


def integrate_crop_and_market_prices(
    df_crop: pd.DataFrame,
    df_market: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Integrates National Crop Production statistics with Agricultural Market Prices
    via canonical crop name alignment.
    Computes national totals (Production, Area, Mean Yield) aggregated by crop
    and aligns with market modal prices, price range, and MSP status.
    """
    if df_crop.empty or df_market.empty:
        return pd.DataFrame(), {"status": "empty"}

    # Aggregate Crop Production by Crop
    crop_clean = df_crop.copy()
    crop_clean["canonical_crop"] = crop_clean["Crop"].apply(standardize_crop_name)

    crop_agg = crop_clean.groupby("canonical_crop").agg({
        "Production": "sum",
        "Area": "sum",
        "Yield": "mean"
    }).reset_index()
    crop_agg["Yield"] = crop_agg["Yield"].round(2)

    # Prepare Market Prices
    market_clean = df_market.copy()
    market_clean["canonical_crop"] = market_clean["crop"].apply(standardize_crop_name)

    # Safe merge on canonical_crop
    merged, diag = perform_safe_join(
        crop_agg,
        market_clean,
        left_key="canonical_crop",
        right_key="canonical_crop",
        how="inner",
        left_name="National Production Aggregates",
        right_name="Market Intelligence"
    )

    return merged, diag


def integrate_weather_and_crop_production(
    df_crop: pd.DataFrame,
    df_weather_monthly: pd.DataFrame,
    city_name: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Integrates district-level crop production with city weather climatology
    strictly for verified geographical overlap and overlapping years (2009-2015).
    """
    city_cap = city_name.capitalize()
    if city_cap not in CITY_TO_GEO:
        return pd.DataFrame(), {
            "status": "error",
            "message": f"No verified geographic mapping defined for city: {city_name}"
        }

    geo = CITY_TO_GEO[city_cap]
    target_state = geo["state"]
    target_districts = [d.title() for d in geo["districts"]]

    # Filter crop production for the matching state & district
    crop_sub = df_crop[
        (df_crop["State_Name"].str.title() == target_state) &
        (df_crop["District_Name"].str.title().isin(target_districts))
    ].copy()

    if crop_sub.empty:
        # Fallback to state match if specific district is not in crop dataset
        crop_sub = df_crop[df_crop["State_Name"].str.title() == target_state].copy()

    if crop_sub.empty:
        return pd.DataFrame(), {
            "status": "error",
            "message": f"No crop production records found for {target_state} / {target_districts}."
        }

    # Weather annual summary by year
    weather_annual = df_weather_monthly.groupby("year").agg({
        "tempC": "mean",
        "precipMM": "sum",
        "humidity": "mean"
    }).reset_index()
    weather_annual.columns = ["Crop_Year", "mean_annual_tempC", "total_annual_rainfall_mm", "mean_annual_humidity"]
    weather_annual["mean_annual_tempC"] = weather_annual["mean_annual_tempC"].round(2)
    weather_annual["mean_annual_humidity"] = weather_annual["mean_annual_humidity"].round(2)
    weather_annual["total_annual_rainfall_mm"] = weather_annual["total_annual_rainfall_mm"].round(1)

    # Safe join on Crop_Year
    merged, diag = perform_safe_join(
        crop_sub,
        weather_annual,
        left_key="Crop_Year",
        right_key="Crop_Year",
        how="inner",
        left_name=f"Crop Production ({target_state})",
        right_name=f"Weather Climatology ({city_cap})"
    )

    diag["target_state"] = target_state
    diag["target_districts"] = target_districts
    diag["city"] = city_cap

    return merged, diag
