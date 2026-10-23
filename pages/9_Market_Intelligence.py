"""
AgriPredict AI — Agricultural Market Intelligence & APMC Price Analytics.
Analyzes mandi modal prices, price spreads (Min-Max), and policy divergence
against Government Minimum Support Prices (MSP) across 122 commodities.
"""

import streamlit as st
import pandas as pd
import numpy as np

from config.settings import THEME, CUSTOM_CSS
from preprocessing.data_loader import load_market_prices
from preprocessing.data_cleaning import clean_market_prices
from analysis.market_analysis import (
    get_top_commodities, get_msp_policy_comparison, get_price_volatility_spread
)
from utils.charts import create_bar_chart, create_box_plot, create_histogram, apply_theme_layout
from utils.helpers import render_header, render_metric_card, render_disclaimer, get_csv_download_bytes
from utils.metrics import format_rupees

import plotly.graph_objects as go

st.set_page_config(page_title="Market Intelligence — AgriPredict AI", page_icon="🌾", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

render_header(
    "Agricultural Market Intelligence & MSP Policy Analysis",
    "Empirical price distribution, Mandi modal pricing (Rs/Quintal), price spreads, and Government Minimum Support Price (MSP) policy benchmarking."
)

with st.spinner("Loading agricultural market prices..."):
    raw_df = load_market_prices()
    df_market, _ = clean_market_prices(raw_df)

if df_market.empty:
    st.error("Market prices dataset could not be loaded.")
else:
    # Methodology & Limitation Notice
    st.markdown(
        """
        <div style="background: #F4F7F4; border-left: 4px solid #1B4D3E; padding: 1rem 1.25rem; border-radius: 8px; margin-bottom: 1.5rem;">
            <b style="color: #1B4D3E;">📌 Analytical Methodology & Data Scope:</b>
            <span style="color: #132A13; font-size: 0.9rem;">
                This dataset captures a verified cross-sectional snapshot across <b>122 agricultural commodities</b> in Indian APMC mandis.
                Prices are officially denominated in <b>Rupees per Quintal (100 kg)</b>. In accordance with rigorous scientific practice,
                longitudinal time-series forecasting is intentionally excluded due to the lack of continuous daily date series timestamps,
                preventing spurious forecasting and ensuring strict empirical integrity.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    msp_df, msp_summary = get_msp_policy_comparison(df_market)

    # Market KPIs
    st.markdown("### 📊 Mandi Price & Policy Indicators")
    m1, m2, m3, m4 = st.columns(4)

    avg_modal = df_market["modal_rs_per_quintal"].mean()
    median_modal = df_market["modal_rs_per_quintal"].median()
    msp_count = msp_summary.get("total_msp_crops", 0)
    above_msp = msp_summary.get("above_msp_count", 0)

    with m1:
        render_metric_card("Analyzed Commodities", f"{len(df_market)} Crops", "Mandi reported varieties", THEME["primary"])
    with m2:
        render_metric_card("Mean Modal Price", format_rupees(avg_modal), f"Median: {format_rupees(median_modal)}", THEME["primary_light"])
    with m3:
        render_metric_card("Govt MSP Benchmark", f"{msp_count} Crops", "Commodities under MSP mandate", THEME["accent_gold"])
    with m4:
        render_metric_card("Trading Above MSP", f"{above_msp} / {msp_count}", f"{msp_summary.get('above_msp_pct', 0)}% of MSP commodities", THEME["success"])

    st.markdown("<br/>", unsafe_allow_html=True)

    # Tabbed Market Insights
    tab_rank, tab_msp, tab_spread = st.tabs([
        "🏆 Top Commodity Price Rankings",
        "⚖️ Open Market vs Government MSP",
        "📈 Mandi Price Spread & Volatility"
    ])

    with tab_rank:
        st.markdown("### 🥇 Top Valued Agricultural Commodities")
        top_n = st.slider("Select number of commodities to rank:", min_value=5, max_value=30, value=12)
        top_comm_df = get_top_commodities(df_market, top_n=top_n)

        fig_comm = create_bar_chart(
            top_comm_df, x="crop", y="modal_rs_per_quintal",
            title=f"Top {top_n} Commodities by Modal Price (Rs/Quintal)"
        )
        st.plotly_chart(fig_comm, use_container_width=True)

        st.dataframe(
            top_comm_df.rename(columns={
                "crop": "Commodity",
                "modal_rs_per_quintal": "Modal Price (Rs/Qtl)",
                "min": "Min Price (Rs/Qtl)",
                "max": "Max Price (Rs/Qtl)",
                "msp": "MSP (Rs/Qtl)"
            }),
            use_container_width=True
        )

    with tab_msp:
        st.markdown("### ⚖️ Government MSP vs Market Modal Price Policy Audit")
        st.write(
            "The Commission for Agricultural Costs and Prices (CACP) announces Minimum Support Prices (MSP) to safeguard farmers against price collapses. "
            "Below is an empirical comparison between the declared floor price and current Mandi modal prices:"
        )

        if not msp_df.empty:
            # Grouped bar chart comparing Modal vs MSP
            fig_msp = go.Figure()
            fig_msp.add_trace(go.Bar(
                x=msp_df["crop"],
                y=msp_df["modal_rs_per_quintal"],
                name="Market Modal Price (Rs/Qtl)",
                marker_color=THEME["primary_light"]
            ))
            fig_msp.add_trace(go.Bar(
                x=msp_df["crop"],
                y=msp_df["msp"],
                name="Government MSP (Rs/Qtl)",
                marker_color="#FF9800"
            ))
            fig_msp = apply_theme_layout(fig_msp, title="Mandi Modal Price vs Government Minimum Support Price (MSP)", height=450)
            fig_msp.update_layout(barmode="group")
            st.plotly_chart(fig_msp, use_container_width=True)

            st.markdown("#### 📋 Detailed MSP Divergence Table")
            display_msp = msp_df[["crop", "modal_rs_per_quintal", "msp", "diff_rs", "pct_diff", "status"]].copy()
            display_msp.columns = ["Commodity", "Market Modal (Rs)", "MSP (Rs)", "Divergence (Rs)", "Divergence (%)", "Policy Status"]
            st.dataframe(display_msp, use_container_width=True)

    with tab_spread:
        st.markdown("### 📉 Mandi Price Dispersion (Spread = Max - Min)")
        st.write("A wide gap between minimum and maximum recorded mandi prices indicates quality grade variation or localized supply bottlenecks:")
        
        spread_df = get_price_volatility_spread(df_market, top_n=15)
        fig_spread = create_bar_chart(
            spread_df, x="crop", y="spread_rs",
            title="Top 15 Commodities by Absolute Mandi Price Spread (Max - Min in Rs/Quintal)"
        )
        st.plotly_chart(fig_spread, use_container_width=True)

    st.markdown("---")
    
    # Filter and Download
    st.markdown("### 🔍 Search & Export Market Data")
    search_q = st.text_input("Filter commodity by name:", "")
    view_df = df_market
    if search_q:
        view_df = df_market[df_market["crop"].str.contains(search_q, case=False)]

    st.dataframe(view_df, use_container_width=True)
    st.download_button(
        label="📥 Download Market Data (CSV)",
        data=get_csv_download_bytes(view_df),
        file_name="market_prices_filtered.csv",
        mime="text/csv"
    )
    
    render_disclaimer("market")
