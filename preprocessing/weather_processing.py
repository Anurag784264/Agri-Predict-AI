"""
Weather processing and aggregation pipeline for AgriPredict AI.
Processes hourly meteorological records, extracts temporal and seasonal features,
and computes daily, monthly, and cross-city aggregates for optimal dashboard performance.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

try:
    import streamlit as st
    cache_decorator = st.cache_data(show_spinner=False)
except Exception:
    def cache_decorator(func):
        return func

from preprocessing.data_loader import load_weather_city, get_available_weather_cities


def assign_indian_season(month: int) -> str:
    """
    Categorizes calendar months into standard Indian meteorological seasons:
    - Winter: December, January, February
    - Summer / Pre-Monsoon: March, April, May
    - Monsoon (Southwest Monsoon): June, July, August, September
    - Post-Monsoon / Autumn: October, November
    """
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"


def clean_weather_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses datetime, cleans numeric weather parameters, and extracts temporal attributes.
    """
    if df.empty:
        return df

    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()

    # Parse date_time
    if "date_time" in cleaned.columns:
        cleaned["date_time"] = pd.to_datetime(cleaned["date_time"], errors="coerce")
        cleaned = cleaned.dropna(subset=["date_time"])
        cleaned["year"] = cleaned["date_time"].dt.year
        cleaned["month"] = cleaned["date_time"].dt.month
        cleaned["month_name"] = cleaned["date_time"].dt.strftime("%b")
        cleaned["day"] = cleaned["date_time"].dt.day
        cleaned["date"] = cleaned["date_time"].dt.date
        cleaned["season"] = cleaned["month"].apply(assign_indian_season)

    # Core numeric columns
    numeric_weather_cols = [
        "tempC", "maxtempC", "mintempC", "humidity", "precipMM",
        "pressure", "cloudcover", "windspeedKmph", "sunHour", "uvIndex",
        "DewPointC", "FeelsLikeC", "HeatIndexC", "visibility"
    ]
    for col in numeric_weather_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    return cleaned


@cache_decorator
def get_daily_weather_summary(city_name: str) -> pd.DataFrame:
    """
    Aggregates hourly city weather into daily metrics:
    - mean, min, max temperature
    - total daily rainfall (precipMM)
    - mean humidity, wind speed, pressure
    """
    raw_df = load_weather_city(city_name)
    if raw_df.empty:
        return pd.DataFrame()

    df = clean_weather_dataframe(raw_df)

    agg_rules = {
        "tempC": "mean",
        "maxtempC": "max",
        "mintempC": "min",
        "precipMM": "sum",
        "humidity": "mean",
        "pressure": "mean",
        "windspeedKmph": "mean",
        "cloudcover": "mean",
        "sunHour": "first",
        "year": "first",
        "month": "first",
        "month_name": "first",
        "season": "first"
    }
    # Keep only available columns
    rules = {col: func for col, func in agg_rules.items() if col in df.columns}

    daily = df.groupby(["date"]).agg(rules).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["city"] = city_name.capitalize()
    daily["tempC"] = daily["tempC"].round(1)
    daily["humidity"] = daily["humidity"].round(1)
    daily["precipMM"] = daily["precipMM"].round(2)
    daily["pressure"] = daily["pressure"].round(1)
    daily["windspeedKmph"] = daily["windspeedKmph"].round(1)

    return daily


@cache_decorator
def get_monthly_weather_summary(city_name: str) -> pd.DataFrame:
    """
    Aggregates weather metrics by month across years for long-term climatology analysis.
    """
    daily = get_daily_weather_summary(city_name)
    if daily.empty:
        return pd.DataFrame()

    monthly = daily.groupby(["year", "month", "month_name", "season"]).agg({
        "tempC": "mean",
        "maxtempC": "max",
        "mintempC": "min",
        "precipMM": "sum",
        "humidity": "mean",
        "windspeedKmph": "mean"
    }).reset_index()

    monthly["city"] = city_name.capitalize()
    monthly["tempC"] = monthly["tempC"].round(2)
    monthly["humidity"] = monthly["humidity"].round(2)
    monthly["precipMM"] = monthly["precipMM"].round(1)
    return monthly


@cache_decorator
def get_all_cities_weather_comparison() -> pd.DataFrame:
    """
    Consolidates monthly climatological summaries across all 8 Indian cities.
    """
    cities = get_available_weather_cities()
    frames = []
    for city in cities:
        m = get_monthly_weather_summary(city)
        if not m.empty:
            frames.append(m)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
