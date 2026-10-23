"""
Meteorological and climatological analysis module for AgriPredict AI.
Computes multi-city climatology, rainfall distributions, extreme temperature events,
and weather parameter correlation matrices.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from preprocessing.weather_processing import (
    get_daily_weather_summary, get_monthly_weather_summary, get_all_cities_weather_comparison
)
from preprocessing.data_loader import get_available_weather_cities


def get_city_weather_kpis(city_name: str) -> Dict[str, Any]:
    """
    Computes key meteorological summary metrics for a given city across 2009-2020.
    """
    daily = get_daily_weather_summary(city_name)
    if daily.empty:
        return {}

    mean_temp = round(daily["tempC"].mean(), 1)
    max_recorded_temp = round(daily["maxtempC"].max(), 1)
    min_recorded_temp = round(daily["mintempC"].min(), 1)
    mean_humidity = round(daily["humidity"].mean(), 1)

    # Annual average rainfall
    annual_rain = daily.groupby("year")["precipMM"].sum().mean()

    # Extreme heat days (> 40°C) and heavy rainfall days (> 50mm)
    heatwave_days = int((daily["maxtempC"] >= 40).sum())
    heavy_rain_days = int((daily["precipMM"] >= 50).sum())

    return {
        "city": city_name.capitalize(),
        "mean_temp_c": mean_temp,
        "max_temp_c": max_recorded_temp,
        "min_temp_c": min_recorded_temp,
        "mean_humidity_pct": mean_humidity,
        "avg_annual_rainfall_mm": round(annual_rain, 1),
        "heatwave_days_count": heatwave_days,
        "heavy_rain_days_count": heavy_rain_days,
        "observation_days": len(daily)
    }


def compare_all_cities_climatology() -> pd.DataFrame:
    """
    Constructs a comparative summary table across all available cities.
    """
    cities = get_available_weather_cities()
    rows = []
    for c in cities:
        kpis = get_city_weather_kpis(c)
        if kpis:
            rows.append(kpis)

    if not rows:
        return pd.DataFrame()

    df_comp = pd.DataFrame(rows)
    return df_comp.sort_values(by="avg_annual_rainfall_mm", ascending=False)


def get_monthly_temperature_and_rainfall(city_name: str) -> pd.DataFrame:
    """
    Computes average monthly temperature and total rainfall climatology (Jan-Dec).
    """
    monthly = get_monthly_weather_summary(city_name)
    if monthly.empty:
        return pd.DataFrame()

    climatology = monthly.groupby("month_name", sort=False).agg({
        "month": "first",
        "tempC": "mean",
        "precipMM": "mean",
        "humidity": "mean"
    }).reset_index()

    climatology = climatology.sort_values(by="month")
    climatology["tempC"] = climatology["tempC"].round(1)
    climatology["precipMM"] = climatology["precipMM"].round(1)
    climatology["humidity"] = climatology["humidity"].round(1)

    return climatology
