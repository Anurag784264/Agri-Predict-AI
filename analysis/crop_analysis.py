"""
Crop production analytical pipelines for AgriPredict AI.
Calculates production aggregates, regional rankings, temporal trends,
cultivation area distributions, and yield metrics.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


def filter_crop_dataframe(
    df: pd.DataFrame,
    states: Optional[List[str]] = None,
    crops: Optional[List[str]] = None,
    seasons: Optional[List[str]] = None,
    year_range: Optional[Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Applies multi-criteria dynamic filtering to the crop production dataset.
    """
    if df.empty:
        return df

    filtered = df.copy()

    if states and len(states) > 0 and "State_Name" in filtered.columns:
        filtered = filtered[filtered["State_Name"].isin(states)]

    if crops and len(crops) > 0 and "Crop" in filtered.columns:
        filtered = filtered[filtered["Crop"].isin(crops)]

    if seasons and len(seasons) > 0 and "Season" in filtered.columns:
        filtered = filtered[filtered["Season"].isin(seasons)]

    if year_range and "Crop_Year" in filtered.columns:
        min_yr, max_yr = year_range
        filtered = filtered[(filtered["Crop_Year"] >= min_yr) & (filtered["Crop_Year"] <= max_yr)]

    return filtered


def get_top_crops_by_production(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Ranks agricultural crops by total national/filtered production in tonnes.
    """
    if df.empty or "Crop" not in df.columns or "Production" not in df.columns:
        return pd.DataFrame()

    ranked = df.groupby("Crop")["Production"].sum().reset_index()
    ranked = ranked.sort_values(by="Production", ascending=False).head(top_n)
    ranked["Production_Tonnes"] = ranked["Production"].round(1)
    return ranked


def get_top_states_by_production(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Ranks Indian states by cumulative production.
    """
    if df.empty or "State_Name" not in df.columns or "Production" not in df.columns:
        return pd.DataFrame()

    ranked = df.groupby("State_Name")["Production"].sum().reset_index()
    ranked = ranked.sort_values(by="Production", ascending=False).head(top_n)
    ranked["Production_Tonnes"] = ranked["Production"].round(1)
    return ranked


def get_yearly_production_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes national annual trends for total production, cultivation area, and average yield.
    """
    if df.empty or "Crop_Year" not in df.columns:
        return pd.DataFrame()

    agg_dict = {"Production": "sum", "Area": "sum"}
    if "Yield" in df.columns:
        agg_dict["Yield"] = "mean"

    yearly = df.groupby("Crop_Year").agg(agg_dict).reset_index()
    yearly = yearly.sort_values(by="Crop_Year")
    yearly["Production"] = yearly["Production"].round(1)
    yearly["Area"] = yearly["Area"].round(1)
    if "Yield" in yearly.columns:
        yearly["Yield"] = yearly["Yield"].round(2)

    return yearly


def get_seasonal_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates production and area metrics across agricultural cropping seasons
    (Kharif, Rabi, Summer, Autumn, Winter, Whole Year).
    """
    if df.empty or "Season" not in df.columns:
        return pd.DataFrame()

    seasonal = df.groupby("Season").agg({
        "Production": "sum",
        "Area": "sum"
    }).reset_index()
    seasonal["Production_Share_Pct"] = ((seasonal["Production"] / seasonal["Production"].sum()) * 100).round(2)
    seasonal = seasonal.sort_values(by="Production", ascending=False)
    return seasonal


def get_state_crop_matrix(df: pd.DataFrame, top_crops: int = 8, top_states: int = 8) -> pd.DataFrame:
    """
    Constructs a pivot matrix of top crops vs top states by production for heatmap visualization.
    """
    if df.empty or "Crop" not in df.columns or "State_Name" not in df.columns or "Production" not in df.columns:
        return pd.DataFrame()

    popular_crops = df.groupby("Crop")["Production"].sum().nlargest(top_crops).index.tolist()
    popular_states = df.groupby("State_Name")["Production"].sum().nlargest(top_states).index.tolist()

    sub = df[df["Crop"].isin(popular_crops) & df["State_Name"].isin(popular_states)]
    pivot = sub.pivot_table(index="State_Name", columns="Crop", values="Production", aggfunc="sum").fillna(0)
    return pivot
