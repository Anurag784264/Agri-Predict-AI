"""
Fertilizer utilization and formulation analytics for AgriPredict AI.
Analyzes relationships between soil types, crop classifications,
environmental parameters, and recommended fertilizer grades.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def get_fertilizer_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes distribution counts and percentages of recommended fertilizer grades.
    """
    if df.empty or "Fertilizer Name" not in df.columns:
        return pd.DataFrame()

    counts = df["Fertilizer Name"].value_counts().reset_index()
    counts.columns = ["Fertilizer", "Count"]
    counts["Percentage"] = ((counts["Count"] / len(df)) * 100).round(1)
    return counts


def get_nutrient_by_fertilizer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates average soil Nitrogen, Phosphorus, and Potassium levels under which
    each fertilizer grade is typically recommended.
    """
    if df.empty or "Fertilizer Name" not in df.columns:
        return pd.DataFrame()

    nutrients = ["Nitrogen", "Phosphorous", "Potassium"]
    available_nutrients = [n for n in nutrients if n in df.columns]

    grouped = df.groupby("Fertilizer Name")[available_nutrients].mean().round(1).reset_index()
    return grouped


def get_fertilizer_by_soil_and_crop(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes cross-tabulation matrices between Fertilizer and (1) Soil Type, (2) Crop Type.
    """
    if df.empty or "Fertilizer Name" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    soil_crosstab = pd.DataFrame()
    if "Soil Type" in df.columns:
        soil_crosstab = pd.crosstab(df["Soil Type"], df["Fertilizer Name"])

    crop_crosstab = pd.DataFrame()
    if "Crop Type" in df.columns:
        crop_crosstab = pd.crosstab(df["Crop Type"], df["Fertilizer Name"])

    return soil_crosstab, crop_crosstab
