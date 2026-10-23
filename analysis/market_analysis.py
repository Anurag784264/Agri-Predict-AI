"""
Agricultural market intelligence and price analytics for AgriPredict AI.
Evaluates APMC mandi modal price distributions, price spreads (Min-Max),
and policy alignment against Government Minimum Support Prices (MSP).
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


def get_top_commodities(df: pd.DataFrame, top_n: int = 15, sort_col: str = "modal_rs_per_quintal") -> pd.DataFrame:
    """
    Ranks agricultural commodities by modal, minimum, or maximum market price (Rs/Quintal).
    """
    if df.empty or sort_col not in df.columns or "crop" not in df.columns:
        return pd.DataFrame()

    ranked = df.sort_values(by=sort_col, ascending=False).head(top_n).copy()
    return ranked[["crop", "modal_rs_per_quintal", "min", "max", "msp"]]


def get_msp_policy_comparison(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compares open market modal prices against declared Government Minimum Support Prices (MSP).
    Identifies which agricultural commodities are trading above vs below MSP support floors.
    """
    if df.empty or "msp" not in df.columns or "modal_rs_per_quintal" not in df.columns:
        return pd.DataFrame(), {}

    # Filter records with official MSP
    msp_df = df[df["msp"].notnull() & (df["msp"] > 0)].copy()
    if msp_df.empty:
        return pd.DataFrame(), {}

    msp_df["diff_rs"] = msp_df["modal_rs_per_quintal"] - msp_df["msp"]
    msp_df["pct_diff"] = ((msp_df["diff_rs"] / msp_df["msp"]) * 100).round(2)
    msp_df["status"] = msp_df["diff_rs"].apply(
        lambda x: "Trading Above MSP" if x > 0 else ("Trading Below MSP" if x < 0 else "Equal to MSP")
    )

    above_count = int((msp_df["diff_rs"] > 0).sum())
    below_count = int((msp_df["diff_rs"] < 0).sum())
    total_msp_crops = len(msp_df)

    summary = {
        "total_msp_crops": total_msp_crops,
        "above_msp_count": above_count,
        "below_msp_count": below_count,
        "above_msp_pct": round((above_count / total_msp_crops) * 100, 1),
        "below_msp_pct": round((below_count / total_msp_crops) * 100, 1),
        "avg_msp_rs": round(msp_df["msp"].mean(), 2),
        "avg_market_rs": round(msp_df["modal_rs_per_quintal"].mean(), 2)
    }

    return msp_df.sort_values(by="pct_diff", ascending=False), summary


def get_price_volatility_spread(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """
    Identifies commodities with highest mandi price dispersion (Max Price - Min Price).
    """
    if df.empty or "max" not in df.columns or "min" not in df.columns:
        return pd.DataFrame()

    df_spread = df.copy()
    df_spread["spread_rs"] = df_spread["max"] - df_spread["min"]
    df_spread["spread_pct"] = ((df_spread["spread_rs"] / df_spread["min"].replace(0, np.nan)) * 100).round(1)

    top_spread = df_spread.sort_values(by="spread_rs", ascending=False).head(top_n)
    return top_spread[["crop", "min", "max", "modal_rs_per_quintal", "spread_rs", "spread_pct"]]
