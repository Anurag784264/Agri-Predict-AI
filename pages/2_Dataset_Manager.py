"""
AgriPredict AI — Dataset Manager.
Inspects configured agricultural datasets, displays column metadata and statistics,
and provides schema inspection and data quality diagnostics.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from config.settings import THEME, CUSTOM_CSS, get_all_dataset_status, resolve_dataset_path
from preprocessing.data_loader import (
    load_crop_production, load_fertilizer_data, load_market_prices,
    load_soil_data, load_weather_city, get_available_weather_cities,
    inspect_dataset, safe_read_csv
)
from utils.helpers import render_header, render_metric_card, get_csv_download_bytes

st.set_page_config(page_title="Dataset Manager — AgriPredict AI", page_icon="🌾", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

render_header(
    "Dataset Governance & Inventory Studio",
    "Data inventory profiling, column schema inspection, missing value detection, and descriptive statistics."
)

dataset_options = [
    "Crop Production (crop_production.csv)",
    "Fertilizer Recommendation (Fertilizer Prediction.csv)",
    "Market Prices & MSP (Indian Agricultural Market Prices.csv)",
    "Soil Nutrients & Crop (Soil Nutrients and Crop Recommendation Dataset.csv)",
    "Weather Climatology (Weather Observatories)"
]

selected_opt = st.selectbox("Choose a dataset to inspect:", dataset_options)

df_inspect = pd.DataFrame()
dataset_name = ""
file_path_str = ""

if "Crop Production" in selected_opt:
    dataset_name = "Crop Production"
    path = resolve_dataset_path("crop_production")
    file_path_str = str(path) if path else "Not found"
    df_inspect = load_crop_production(nrows=50000)
elif "Fertilizer" in selected_opt:
    dataset_name = "Fertilizer Recommendation"
    path = resolve_dataset_path("fertilizer")
    file_path_str = str(path) if path else "Not found"
    df_inspect = load_fertilizer_data()
elif "Market" in selected_opt:
    dataset_name = "Market Prices"
    path = resolve_dataset_path("market_prices")
    file_path_str = str(path) if path else "Not found"
    df_inspect = load_market_prices()
elif "Soil" in selected_opt:
    dataset_name = "Soil Nutrients & Crop"
    path = resolve_dataset_path("soil_recommendation")
    file_path_str = str(path) if path else "Not found"
    df_inspect = load_soil_data()
elif "Weather" in selected_opt:
    cities = get_available_weather_cities()
    chosen_city = st.selectbox("Select weather observatory city:", cities, index=0)
    dataset_name = f"Weather ({chosen_city})"
    path = resolve_dataset_path("weather")
    file_path_str = str(path / f"{chosen_city.lower()}.csv") if path else "Not found"
    df_inspect = load_weather_city(chosen_city, nrows=20000)

if df_inspect.empty:
    st.warning(f"Could not load {dataset_name}. Please verify that the CSV file exists.")
else:
    meta = inspect_dataset(df_inspect)
    file_size_mb = 0.0
    if file_path_str and Path(file_path_str).exists():
        file_size_mb = round(Path(file_path_str).stat().st_size / (1024 * 1024), 2)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_metric_card("Total Rows", f"{meta['rows']:,}", "Loaded records", THEME["primary"])
    with k2:
        render_metric_card("Columns", str(meta["columns"]), "Feature dimensions", THEME["primary_light"])
    with k3:
        render_metric_card("File Size", f"{file_size_mb} MB" if file_size_mb else f"{meta['memory_mb']} MB", "Storage footprint", THEME["accent_gold"])
    with k4:
        render_metric_card("Duplicate Rows", str(meta["duplicate_count"]), "Identical records", THEME["warning"])
    with k5:
        total_nulls = sum(meta["missing_counts"].values())
        render_metric_card("Missing Values", f"{total_nulls:,}", "Null data cells", THEME["danger"] if total_nulls > 0 else THEME["success"])

    st.markdown("<br/>", unsafe_allow_html=True)

    tab_preview, tab_schema, tab_stats = st.tabs(["📋 Data Preview", "🧬 Column Schema & Missing Profiling", "📐 Descriptive Statistics"])

    with tab_preview:
        row_limit = st.select_slider("Preview rows:", options=[5, 10, 20, 50, 100], value=10)
        st.dataframe(df_inspect.head(row_limit), use_container_width=True)
        st.download_button(
            label=f"📥 Download {dataset_name} Preview (CSV)",
            data=get_csv_download_bytes(df_inspect.head(row_limit)),
            file_name=f"{dataset_name.lower().replace(' ', '_')}_preview.csv",
            mime="text/csv"
        )

    with tab_schema:
        schema_rows = []
        for col in df_inspect.columns:
            null_count = meta["missing_counts"].get(col, 0)
            null_pct = meta["missing_percentages"].get(col, 0.0)
            dtype_str = meta["dtypes"].get(col, "unknown")
            nunique = df_inspect[col].nunique()
            schema_rows.append({
                "Column Name": col,
                "Data Type": dtype_str,
                "Non-Null Count": len(df_inspect) - null_count,
                "Missing Count": null_count,
                "Missing %": f"{null_pct}%",
                "Unique Values": nunique,
                "Sample Value": str(df_inspect[col].dropna().iloc[0]) if not df_inspect[col].dropna().empty else "-"
            })
        schema_df = pd.DataFrame(schema_rows)

        col_search = st.text_input("🔍 Search column names:", "")
        if col_search:
            schema_df = schema_df[schema_df["Column Name"].str.contains(col_search, case=False)]

        st.dataframe(schema_df, use_container_width=True)

    with tab_stats:
        num_cols = meta["numeric_columns"]
        if num_cols:
            st.markdown(f"**Numerical Features ({len(num_cols)}):**")
            st.dataframe(df_inspect[num_cols].describe().T.round(3), use_container_width=True)
        else:
            st.info("No continuous numerical columns identified in this dataset.")
