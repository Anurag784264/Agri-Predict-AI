"""
AgriPredict AI — Meteorological Climatology & Weather Analytics Studio.
Analyzes 11 years of hourly meteorological records across 8 major Indian cities,
evaluating temperature extremes, monsoon precipitation, and seasonal patterns.
"""

import streamlit as st
import pandas as pd
import numpy as np

from config.settings import THEME, CUSTOM_CSS
from preprocessing.data_loader import get_available_weather_cities
from preprocessing.weather_processing import (
    get_daily_weather_summary, get_monthly_weather_summary, get_all_cities_weather_comparison
)
from analysis.weather_analysis import (
    get_city_weather_kpis, compare_all_cities_climatology, get_monthly_temperature_and_rainfall
)
from utils.charts import (
    create_bar_chart, create_line_chart, create_correlation_heatmap, apply_theme_layout
)
from utils.helpers import render_header, render_metric_card

import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Weather Analytics — AgriPredict AI", page_icon="🌾", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

render_header(
    "Historical Weather Analytics & Climatological Profiling",
    "Comprehensive meteorological intelligence across 8 Indian regional hubs spanning 11 years of hourly observations (2009–2020)."
)

cities = get_available_weather_cities()
if not cities:
    st.error("No weather dataset files found in WEATHER DATASET directory.")
else:
    # Mode selection
    mode = st.radio("Analytics Scope:", ["Single City Deep-Dive", "Multi-City Cross Comparison"], horizontal=True)

    if mode == "Single City Deep-Dive":
        selected_city = st.selectbox("Select City Observatory:", cities, index=0)
        
        with st.spinner(f"Aggregating meteorological data for {selected_city}..."):
            daily_weather = get_daily_weather_summary(selected_city)
            kpis = get_city_weather_kpis(selected_city)

        if daily_weather.empty or not kpis:
            st.warning(f"Unable to process weather data for {selected_city}.")
        else:
            # City KPIs
            st.markdown(f"### 📍 Climatological Summary: **{selected_city}**")
            w1, w2, w3, w4, w5 = st.columns(5)
            with w1:
                render_metric_card("Mean Temp", f"{kpis['mean_temp_c']} °C", f"Min {kpis['min_temp_c']}°C | Max {kpis['max_temp_c']}°C", THEME["primary"])
            with w2:
                render_metric_card("Annual Rainfall", f"{kpis['avg_annual_rainfall_mm']} mm", "Yearly mean precipitation", THEME["info"])
            with w3:
                render_metric_card("Relative Humidity", f"{kpis['mean_humidity_pct']} %", "Atmospheric moisture", THEME["primary_light"])
            with w4:
                render_metric_card("Heatwave Days", f"{kpis['heatwave_days_count']} Days", "Days exceeding 40°C", THEME["warning"])
            with w5:
                render_metric_card("Heavy Rain Days", f"{kpis['heavy_rain_days_count']} Days", "Days exceeding 50mm rain", THEME["accent_gold"])

            st.markdown("<br/>", unsafe_allow_html=True)

            # Monthly Climatology Chart (Temperature & Rainfall dual axes)
            st.markdown("### 🗓️ Monthly Climatology & Monsoon Distribution")
            climatology = get_monthly_temperature_and_rainfall(selected_city)
            
            fig_clim = make_subplots(specs=[[{"secondary_y": True}]])
            fig_clim.add_trace(
                go.Bar(
                    x=climatology["month_name"],
                    y=climatology["precipMM"],
                    name="Rainfall (mm)",
                    marker_color="#42A5F5",
                    opacity=0.75
                ),
                secondary_y=False
            )
            fig_clim.add_trace(
                go.Scatter(
                    x=climatology["month_name"],
                    y=climatology["tempC"],
                    name="Mean Temperature (°C)",
                    marker_color="#D32F2F",
                    mode="lines+markers",
                    line=dict(width=3)
                ),
                secondary_y=True
            )
            fig_clim.update_layout(
                title=f"<b>Average Monthly Temperature & Precipitation Climatology — {selected_city}</b>",
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
            )
            fig_clim.update_yaxes(title_text="Precipitation (mm)", secondary_y=False)
            fig_clim.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
            st.plotly_chart(fig_clim, use_container_width=True)
            st.caption("Depicts the monsoon seasonal pulse (June-September) critical for Kharif agricultural planning.")

            # Time Series with Year Range
            st.markdown("### 📈 Longitudinal Daily Temperature Trajectory")
            min_y = int(daily_weather["year"].min())
            max_y = int(daily_weather["year"].max())
            sel_yr = st.slider("Select Observation Year:", min_value=min_y, max_value=max_y, value=2015)
            
            sub_year = daily_weather[daily_weather["year"] == sel_yr]
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=sub_year["date"], y=sub_year["maxtempC"], name="Max Temp (°C)", line=dict(color="#FF7043")))
            fig_ts.add_trace(go.Scatter(x=sub_year["date"], y=sub_year["tempC"], name="Mean Temp (°C)", line=dict(color="#2E7D32")))
            fig_ts.add_trace(go.Scatter(x=sub_year["date"], y=sub_year["mintempC"], name="Min Temp (°C)", line=dict(color="#42A5F5")))
            fig_ts = apply_theme_layout(fig_ts, title=f"Daily Temperature Bounds for {selected_city} ({sel_yr})", height=380)
            st.plotly_chart(fig_ts, use_container_width=True)

            # Correlation Heatmap
            st.markdown("### 🌡️ Weather Variables Correlation Matrix")
            w_cols = ["tempC", "maxtempC", "mintempC", "humidity", "precipMM", "pressure", "windspeedKmph", "cloudcover"]
            avail_w_cols = [c for c in w_cols if c in daily_weather.columns]
            fig_w_corr = create_correlation_heatmap(daily_weather[avail_w_cols], title=f"Meteorological Parameter Correlation — {selected_city}")
            st.plotly_chart(fig_w_corr, use_container_width=True)

    else:
        # Multi-City Cross Comparison
        st.markdown("### 🏙️ Cross-City Climatological Benchmarking")
        with st.spinner("Aggregating multi-city climatology..."):
            df_comp = compare_all_cities_climatology()

        st.dataframe(df_comp, use_container_width=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_rain = create_bar_chart(
                df_comp, x="city", y="avg_annual_rainfall_mm",
                title="Average Annual Precipitation Across 8 Cities (mm)"
            )
            st.plotly_chart(fig_rain, use_container_width=True)

        with col_c2:
            fig_temp = create_bar_chart(
                df_comp, x="city", y="mean_temp_c",
                title="Mean Annual Temperature Across 8 Cities (°C)"
            )
            st.plotly_chart(fig_temp, use_container_width=True)
