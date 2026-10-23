"""
AgriPredict AI — Smart Agriculture Analytics & Crop Prediction Platform
Main Application Entry Point.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AgriPredict AI — Agriculture Analytics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

from config.settings import THEME, CUSTOM_CSS, get_all_dataset_status
from preprocessing.data_loader import (
    load_crop_production, load_fertilizer_data, load_market_prices,
    load_soil_data, get_available_weather_cities
)
from utils.metrics import compute_platform_kpis
from utils.helpers import render_header, render_metric_card, render_disclaimer

# Inject custom design tokens
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Sidebar brand
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 1rem 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 1rem;">
            <h2 style="color: #1B4D3E; margin: 0; font-size: 1.5rem;">🌾 AgriPredict AI</h2>
            <p style="color: #556B2F; margin: 0.2rem 0 0 0; font-size: 0.85rem;">Smart Agriculture & Prediction</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("### 📌 Navigation")
    st.info("Use the sidebar pages to explore data ingestion, exploratory charts, domain analytics, ML models, and reports.")

# Header
render_header(
    "AgriPredict AI — Smart Agriculture Analytics & Crop Prediction Platform",
    "An enterprise-grade multi-dataset agricultural intelligence platform integrating crop production, meteorological climatology, soil nutrients, chemical fertilizers, and mandi market prices."
)

# Load real datasets for KPI computation
with st.spinner("Initializing agricultural data engines..."):
    df_crop = load_crop_production(nrows=50000)
    df_fert = load_fertilizer_data()
    df_market = load_market_prices()
    df_soil = load_soil_data()
    weather_cities = get_available_weather_cities()

kpis = compute_platform_kpis(df_crop, df_fert, df_market, df_soil, len(weather_cities))

# KPI Overview Row
st.markdown("### 📊 Platform Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    render_metric_card("Total Dataset Records", kpis["total_records_formatted"], "Across 5 configured data sources", THEME["primary"])
with col2:
    render_metric_card("Distinct Crops", str(kpis["crop_count"]), "Covering Indian cultivation varieties", THEME["primary_light"])
with col3:
    render_metric_card("Reported Production", kpis["total_production_formatted"], "Cumulative national harvest records", THEME["accent_gold"])
with col4:
    render_metric_card("Weather Observatories", f"{kpis['weather_cities']} Cities", "Hourly historical records (2009-2020)", THEME["info"])

st.markdown("<br/>", unsafe_allow_html=True)


# Dataset status removed

st.markdown("<br/>", unsafe_allow_html=True)


render_disclaimer("general")
