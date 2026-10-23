"""
AgriPredict AI — Executive Overview & Key Agricultural Findings.
"""

import streamlit as st
import pandas as pd

from config.settings import THEME, CUSTOM_CSS
from preprocessing.data_loader import (
    load_crop_production, load_fertilizer_data, load_market_prices,
    load_soil_data, get_available_weather_cities
)
from preprocessing.weather_processing import get_all_cities_weather_comparison
from analysis.crop_analysis import get_top_crops_by_production, get_yearly_production_trends, get_seasonal_breakdown
from analysis.weather_analysis import compare_all_cities_climatology
from analysis.fertilizer_analysis import get_fertilizer_distribution
from analysis.market_analysis import get_top_commodities
from utils.charts import create_bar_chart, create_line_chart, create_donut_chart
from utils.helpers import render_header, render_metric_card

st.set_page_config(page_title="Executive Overview — AgriPredict AI", page_icon="🌾", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

render_header(
    "Executive Agricultural Overview & Key Findings",
    "Synthesized high-level cross-domain insights spanning national crop production, climatology, soil nutrients, and mandi market prices."
)

with st.spinner("Loading analytical datasets..."):
    df_crop = load_crop_production()
    df_fert = load_fertilizer_data()
    df_market = load_market_prices()
    df_soil = load_soil_data()
    df_weather_comp = compare_all_cities_climatology()

# Automated Key Findings Cards
st.markdown("### 🔍 Verified Empirical Findings from Real Data")
find_col1, find_col2, find_col3, find_col4 = st.columns(4)

# Compute top findings safely
top_crops = get_top_crops_by_production(df_crop, top_n=1)
top_crop_name = top_crops.iloc[0]["Crop"] if not top_crops.empty else "N/A"
top_crop_prod = top_crops.iloc[0]["Production_Tonnes"] if not top_crops.empty else 0

top_wet_city = df_weather_comp.iloc[0]["city"] if not df_weather_comp.empty else "N/A"
top_wet_val = df_weather_comp.iloc[0]["avg_annual_rainfall_mm"] if not df_weather_comp.empty else 0

fert_dist = get_fertilizer_distribution(df_fert)
top_fert_name = fert_dist.iloc[0]["Fertilizer"] if not fert_dist.empty else "N/A"

top_market = get_top_commodities(df_market, top_n=1)
top_commodity_name = top_market.iloc[0]["crop"] if not top_market.empty else "N/A"
top_commodity_price = top_market.iloc[0]["modal_rs_per_quintal"] if not top_market.empty else 0

with find_col1:
    render_metric_card("Highest Volume Crop", top_crop_name, f"{top_crop_prod:,.0f} Tonnes cumulative", THEME["primary"])
with find_col2:
    render_metric_card("Highest Rainfall Hub", top_wet_city, f"{top_wet_val:,.1f} mm annual average", THEME["info"])
with find_col3:
    render_metric_card("Dominant Fertilizer", top_fert_name, "Most frequently prescribed", THEME["primary_light"])
with find_col4:
    render_metric_card("Top Value Commodity", top_commodity_name, f"₹ {top_commodity_price:,.0f} / Quintal modal", THEME["accent_gold"])

st.markdown("<br/>", unsafe_allow_html=True)

# Overview Visualizations
st.markdown("### 📈 Cross-Domain Macro Indicators")

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    top_crops_df = get_top_crops_by_production(df_crop, top_n=8)
    fig_crops = create_bar_chart(
        top_crops_df, x="Crop", y="Production_Tonnes",
        title="Top 8 Agricultural Crops by Cumulative Production (Tonnes)"
    )
    st.plotly_chart(fig_crops, use_container_width=True)
    st.caption("Coconut, Sugarcane, and Rice constitute the dominant production volumes in national records.")

with row1_col2:
    seasonal_df = get_seasonal_breakdown(df_crop)
    fig_season = create_donut_chart(
        seasonal_df, names="Season", values="Production",
        title="National Production Volume Share by Agricultural Season"
    )
    st.plotly_chart(fig_season, use_container_width=True)
    st.caption("Production is categorized across Kharif (monsoon), Rabi (winter), and Whole Year cropping cycles.")

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    yearly_df = get_yearly_production_trends(df_crop)
    fig_yearly = create_line_chart(
        yearly_df, x="Crop_Year", y="Production",
        title="National Crop Production Trajectory (1997 - 2015)"
    )
    st.plotly_chart(fig_yearly, use_container_width=True)
    st.caption("Temporal trajectory reflects overall agricultural output reported across Indian state departments.")

with row2_col2:
    fig_weather = create_bar_chart(
        df_weather_comp, x="city", y="avg_annual_rainfall_mm",
        title="Average Annual Precipitation across 8 Meteorological Hubs (mm)"
    )
    st.plotly_chart(fig_weather, use_container_width=True)
    st.caption("Bombay and Bengaluru represent distinct coastal and plateau precipitation regimes.")
