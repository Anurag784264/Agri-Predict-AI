"""
Data loading and inspection module for AgriPredict AI.
Provides cached loaders with encoding fallbacks and structured inspection metrics.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import os
import pandas as pd
import numpy as np

try:
    import streamlit as st
    cache_decorator = st.cache_data(show_spinner=False)
except Exception:
    # Fallback decorator if not running inside Streamlit
    def cache_decorator(func):
        return func

from config.settings import resolve_dataset_path, DATASET_SPECS


def safe_read_csv(filepath: str | Path, **kwargs) -> pd.DataFrame:
    """
    Safely load a CSV file testing UTF-8 first, then falling back to latin1/cp1252.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at: {path}")

    encodings_to_try = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
    last_err = None

    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as err:
            last_err = err
            continue
        except Exception as err:
            raise err

    raise ValueError(f"Could not read {path} with any supported encodings: {last_err}")


@cache_decorator
def load_crop_production(nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load the national crop production dataset.
    """
    path = resolve_dataset_path("crop_production")
    if not path:
        return pd.DataFrame()
    df = safe_read_csv(path, nrows=nrows)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    return df


@cache_decorator
def load_fertilizer_data() -> pd.DataFrame:
    """
    Load the fertilizer recommendation dataset.
    """
    path = resolve_dataset_path("fertilizer")
    if not path:
        return pd.DataFrame()
    df = safe_read_csv(path)
    # Strip whitespace from columns (e.g. 'Humidity ' -> 'Humidity')
    df.columns = df.columns.str.strip()
    return df


@cache_decorator
def load_market_prices() -> pd.DataFrame:
    """
    Load Indian Agricultural Market Prices dataset.
    """
    path = resolve_dataset_path("market_prices")
    if not path:
        return pd.DataFrame()
    df = safe_read_csv(path)
    df.columns = df.columns.str.strip()
    return df


@cache_decorator
def load_soil_data() -> pd.DataFrame:
    """
    Load Soil Nutrients and Crop Recommendation dataset.
    """
    path = resolve_dataset_path("soil_recommendation")
    if not path:
        return pd.DataFrame()
    df = safe_read_csv(path)
    df.columns = df.columns.str.strip()
    return df


@cache_decorator
def load_weather_city(city_name: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load weather data for a specific city.
    """
    weather_dir = resolve_dataset_path("weather")
    if not weather_dir:
        return pd.DataFrame()

    city_file = weather_dir / f"{city_name.lower()}.csv"
    if not city_file.exists():
        # Case-insensitive search
        for f in weather_dir.glob("*.csv"):
            if f.stem.lower() == city_name.lower():
                city_file = f
                break

    if not city_file.exists():
        return pd.DataFrame()

    df = safe_read_csv(city_file, nrows=nrows)
    df.columns = df.columns.str.strip()
    df["city"] = city_name.capitalize()
    return df


@cache_decorator
def load_weather_multi(cities: Optional[List[str]] = None, sample_every_n: int = 1) -> pd.DataFrame:
    """
    Load weather observations for multiple cities.
    sample_every_n allows downsampling hourly data (e.g. 24 for daily snapshots) for high performance.
    """
    weather_dir = resolve_dataset_path("weather")
    if not weather_dir:
        return pd.DataFrame()

    available_cities = [f.stem.lower() for f in weather_dir.glob("*.csv")]
    target_cities = cities if cities else available_cities

    frames = []
    for city in target_cities:
        df_c = load_weather_city(city)
        if not df_c.empty:
            if sample_every_n > 1:
                df_c = df_c.iloc[::sample_every_n].copy()
            frames.append(df_c)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_available_weather_cities() -> List[str]:
    """
    List names of cities present in the weather directory.
    """
    weather_dir = resolve_dataset_path("weather")
    if not weather_dir or not weather_dir.is_dir():
        return []
    return sorted([f.stem.capitalize() for f in weather_dir.glob("*.csv")])


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive statistical and structural inspection metadata for any DataFrame.
    """
    if df is None or df.empty:
        return {
            "status": "empty",
            "rows": 0,
            "columns": 0,
            "column_list": [],
            "dtypes": {},
            "missing_counts": {},
            "missing_percentages": {},
            "duplicate_count": 0,
            "memory_mb": 0.0,
            "numeric_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "stats": pd.DataFrame()
        }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    missing_counts = df.isnull().sum().to_dict()
    missing_pcts = ((df.isnull().sum() / len(df)) * 100).round(2).to_dict()
    dup_count = int(df.duplicated().sum())
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    return {
        "status": "ok",
        "rows": len(df),
        "columns": len(df.columns),
        "column_list": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": missing_counts,
        "missing_percentages": missing_pcts,
        "duplicate_count": dup_count,
        "memory_mb": memory_mb,
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "datetime_columns": dt_cols,
        "stats": df.describe(include="all").fillna("-")
    }
