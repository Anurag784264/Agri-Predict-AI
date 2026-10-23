"""
AgriPredict AI — National Crop Production Analytics Dashboard.
Evaluates agricultural production, state/district rankings, seasonal cycles,
and cultivation area versus yield metrics across India (1997-2015).
"""

import streamlit as st
import pandas as pd
import numpy as np

from config.settings import THEME, CUSTOM_CSS
from preprocessing.data_loader import load_crop_production
from preprocessing.data_cleaning import clean_crop_production
from analysis.crop_analysis import (
    filter_crop_dataframe, get_top_crops_by_production, get_top_states_by_production,
    get_yearly_production_trends, get_seasonal_breakdown, get_state_crop_matrix
)
from utils.charts import create_bar_chart, create_line_chart, create_scatter_chart, create_donut_chart, apply_theme_layout
from utils.helpers import render_header, render_metric_card, get_csv_download_bytes
from utils.metrics import format_compact_number

import plotly.graph_objects as go

st.set_page_config(page_title="Crop Production — AgriPredict AI", page_icon="🌾", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

render_header(
    "National Crop Production & Yield Analytics",
    "Comprehensive spatio-temporal analysis of agricultural production, cultivation area, and calculated yield across Indian states and seasons."
)

with st.spinner("Loading national crop production data..."):
    raw_df = load_crop_production()
    df_crop, _ = clean_crop_production(raw_df, impute_missing=True)

if df_crop.empty:
    st.error("Crop production dataset could not be loaded.")
else:
    # Sidebar Filters
    with st.sidebar:
        st.markdown("### 🔍 Crop Production Filters")
        
        all_states = sorted(df_crop["State_Name"].dropna().unique().tolist())
        selected_states = st.multiselect("Filter States:", all_states, default=[])

        all_crops = sorted(df_crop["Crop"].dropna().unique().tolist())
        selected_crops = st.multiselect("Filter Crops:", all_crops, default=[])

        all_seasons = sorted(df_crop["Season"].dropna().unique().tolist())
        selected_seasons = st.multiselect("Filter Cropping Seasons:", all_seasons, default=[])

        min_yr = int(df_crop["Crop_Year"].min())
        max_yr = int(df_crop["Crop_Year"].max())
        year_range = st.slider("Crop Year Range:", min_value=min_yr, max_value=max_yr, value=(min_yr, max_yr))

    # Apply filters
    filtered_df = filter_crop_dataframe(
        df_crop,
        states=selected_states if selected_states else None,
        crops=selected_crops if selected_crops else None,
        seasons=selected_seasons if selected_seasons else None,
        year_range=year_range
    )

    # Metric KPI Row
    st.markdown("### 📊 Production Indicators (Filtered Context)")
    tot_prod = filtered_df["Production"].sum()
    tot_area = filtered_df["Area"].sum()
    avg_yield = round(tot_prod / tot_area, 2) if tot_area > 0 else 0.0
    crop_types = filtered_df["Crop"].nunique()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card("Total Production", f"{format_compact_number(tot_prod)} Tonnes", "Cumulative output", THEME["primary"])
    with k2:
        render_metric_card("Cultivated Area", f"{format_compact_number(tot_area)} Ha", "Reported area", THEME["primary_light"])
    with k3:
        render_metric_card("Calculated Yield", f"{avg_yield} Tonnes/Ha", "Yield = Production / Area", THEME["accent_gold"])
    with k4:
        render_metric_card("Distinct Crops", str(crop_types), "Varieties grown in selection", THEME["info"])

    st.markdown("<br/>", unsafe_allow_html=True)

    # Production Analytics Charts
    col_crop_rank, col_state_rank = st.columns(2)

    with col_crop_rank:
        top_n_crops = st.slider("Top Crops to Display:", min_value=5, max_value=20, value=10)
        top_crop_data = get_top_crops_by_production(filtered_df, top_n=top_n_crops)
        fig_c = create_bar_chart(
            top_crop_data, x="Crop", y="Production_Tonnes",
            title=f"Top {top_n_crops} Crops by Total Production (Tonnes)"
        )
        st.plotly_chart(fig_c, use_container_width=True)

    with col_state_rank:
        top_n_states = st.slider("Top States to Display:", min_value=5, max_value=20, value=10)
        top_state_data = get_top_states_by_production(filtered_df, top_n=top_n_states)
        fig_s = create_bar_chart(
            top_state_data, x="State_Name", y="Production_Tonnes",
            title=f"Top {top_n_states} States by Total Production (Tonnes)"
        )
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("---")

    # Temporal & Seasonal Analysis
    col_yearly, col_season = st.columns(2)

    with col_yearly:
        yearly_data = get_yearly_production_trends(filtered_df)
        fig_y = create_line_chart(
            yearly_data, x="Crop_Year", y="Production",
            title="Annual Production Trend Across Selected Filters"
        )
        st.plotly_chart(fig_y, use_container_width=True)

    with col_season:
        seasonal_data = get_seasonal_breakdown(filtered_df)
        fig_sea = create_donut_chart(
            seasonal_data, names="Season", values="Production",
            title="Production Distribution Across Cropping Seasons"
        )
        st.plotly_chart(fig_sea, use_container_width=True)

    # Area vs Production Scatter Plot
    st.markdown("### 🌾 Cultivation Area vs Production Volume")
    scatter_sample = filtered_df.sample(min(len(filtered_df), 1500), random_state=42)
    fig_scatter = create_scatter_chart(
        scatter_sample,
        x="Area",
        y="Production",
        color="Season" if "Season" in scatter_sample.columns else None,
        hover_data=["Crop", "State_Name"],
        title="Cultivation Area (Hectares) vs Production (Tonnes)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption("Scatter relationship reveals production scaling across smallholder versus large-scale plantation farming.")

    # State-Crop Production Matrix Heatmap
    st.markdown("### 🗺️ State vs Crop Production Concentration Matrix")
    matrix_df = get_state_crop_matrix(filtered_df, top_crops=8, top_states=8)
    if not matrix_df.empty:
        fig_matrix = go.Figure(data=go.Heatmap(
            z=matrix_df.values,
            x=matrix_df.columns.tolist(),
            y=matrix_df.index.tolist(),
            colorscale=[[0, "#FFFFFF"], [1, THEME["primary"]]],
            text=np.round(matrix_df.values, 0),
            texttemplate="%{text:,}",
            textfont={"size": 10}
        ))
        fig_matrix = apply_theme_layout(fig_matrix, title="Cross-Tabulation: Top States vs Top Crops Production (Tonnes)", height=450)
        st.plotly_chart(fig_matrix, use_container_width=True)

    # Download Filtered Records
    st.download_button(
        label="📥 Download Filtered Production Records (CSV)",
        data=get_csv_download_bytes(filtered_df.head(10000)),
        file_name="filtered_crop_production.csv",
        mime="text/csv"
    )
