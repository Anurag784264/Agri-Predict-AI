"""
Soil chemical properties and crop nutrient requirement analytics for AgriPredict AI.
Analyzes macronutrients (N, P, K), pH, electrical conductivity, and micronutrients across crops.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def get_soil_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes distribution statistics (Mean, Median, Min, Max, Std) for all soil parameters.
    """
    if df.empty:
        return pd.DataFrame()

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
    stats = df[num_cols].describe().T[["mean", "50%", "min", "max", "std"]].reset_index()
    stats.columns = ["Nutrient / Parameter", "Mean", "Median", "Min", "Max", "StdDev"]
    stats["Mean"] = stats["Mean"].round(2)
    stats["Median"] = stats["Median"].round(2)
    stats["Min"] = stats["Min"].round(2)
    stats["Max"] = stats["Max"].round(2)
    stats["StdDev"] = stats["StdDev"].round(2)
    return stats


def get_crop_nutrient_requirements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates average required soil nutrient concentrations per crop label.
    """
    if df.empty or "label" not in df.columns:
        return pd.DataFrame()

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
    reqs = df.groupby("label")[num_cols].mean().round(2).reset_index()
    return reqs


def classify_soil_ph(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorizes soil samples into standard agronomic pH classifications:
    - Strongly Acidic (< 5.5)
    - Moderately Acidic (5.5 - 6.5)
    - Neutral (6.5 - 7.5)
    - Moderately Alkaline (7.5 - 8.5)
    - Strongly Alkaline (> 8.5)
    """
    if df.empty or "ph" not in df.columns:
        return pd.DataFrame()

    def get_ph_bucket(ph_val):
        if ph_val < 5.5:
            return "Strongly Acidic (<5.5)"
        elif ph_val < 6.5:
            return "Moderately Acidic (5.5-6.5)"
        elif ph_val <= 7.5:
            return "Neutral (6.5-7.5)"
        elif ph_val <= 8.5:
            return "Moderately Alkaline (7.5-8.5)"
        else:
            return "Strongly Alkaline (>8.5)"

    ph_series = df["ph"].apply(get_ph_bucket)
    counts = ph_series.value_counts().reset_index()
    counts.columns = ["pH_Category", "Sample_Count"]
    counts["Percentage"] = ((counts["Sample_Count"] / len(df)) * 100).round(1)
    return counts
