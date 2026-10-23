"""
AgriPredict AI — Interactive Exploratory Data Analysis (EDA) Studio.
Empowers users to visualize feature relationships, distributions, and correlations
across all agricultural datasets with dynamic Plotly charts.
"""

import streamlit as st
import pandas as pd
import numpy as np

from config.settings import THEME, CUSTOM_CSS
from preprocessing.data_loader import (
    load_crop_production, load_fertilizer_data, load_market_prices,
    load_soil_data, load_weather_city, get_available_weather_cities
)
from preprocessing.weather_processing import get_all_cities_weather_comparison
from utils.charts import (
    create_bar_chart, create_line_chart, create_scatter_chart,
    create_box_plot, create_histogram, create_donut_chart, create_correlation_heatmap
)
from utils.helpers import render_header

st.set_page_config(page_title="Exploratory Analysis — AgriPredict AI", page_icon="🌾", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

render_header(
    "Interactive Exploratory Data Analysis (EDA) Studio",
    "Dynamically analyze distributions, multivariate relationships, and correlations across all agricultural datasets."
)

# Dataset Selection
col_ds, col_sample = st.columns([2, 1])
with col_ds:
    dataset_name = st.selectbox(
        "Select dataset for exploratory analysis:",
        ["Crop Production", "Fertilizer Prediction", "Soil Nutrients & Crop", "Market Prices", "Weather Multi-City Climatology"]
    )

df_eda = pd.DataFrame()
if dataset_name == "Crop Production":
    df_eda = load_crop_production(nrows=30000)
    # Compute yield if available
    if "Area" in df_eda.columns and "Production" in df_eda.columns:
        df_eda["Yield"] = (df_eda["Production"] / df_eda["Area"].replace(0, np.nan)).round(2)
elif dataset_name == "Fertilizer Prediction":
    df_eda = load_fertilizer_data()
elif dataset_name == "Soil Nutrients & Crop":
    df_eda = load_soil_data()
elif dataset_name == "Market Prices":
    df_eda = load_market_prices()
elif dataset_name == "Weather Multi-City Climatology":
    df_eda = get_all_cities_weather_comparison()

if df_eda.empty:
    st.warning("Selected dataset is empty or could not be loaded.")
else:
    numeric_cols = df_eda.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_eda.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    all_cols = df_eda.columns.tolist()

    st.markdown("### 🎨 Visual Chart Builder")
    
    chart_type = st.selectbox(
        "Choose visualization type:",
        ["Correlation Heatmap", "Bar Chart", "Scatter Plot", "Box Plot", "Histogram", "Line Chart", "Donut Chart"]
    )

    chart_title = st.text_input("Chart Title:", f"{chart_type} — {dataset_name}")

    if chart_type == "Correlation Heatmap":
        st.markdown("**Correlation Matrix Analysis**")
        selected_num_cols = st.multiselect("Select numeric features for correlation:", numeric_cols, default=numeric_cols[:10])
        if len(selected_num_cols) >= 2:
            fig = create_correlation_heatmap(df_eda, columns=selected_num_cols, title=chart_title)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Pearson correlation coefficient matrix. Deep green indicates positive association; red indicates negative association.")
        else:
            st.info("Select at least 2 numerical features to construct a correlation heatmap.")

    elif chart_type == "Bar Chart":
        c1, c2, c3 = st.columns(3)
        with c1:
            x_col = st.selectbox("Category Axis (X):", categorical_cols if categorical_cols else all_cols)
        with c2:
            y_col = st.selectbox("Metric Axis (Y):", numeric_cols)
        with c3:
            agg_func = st.selectbox("Aggregation Function:", ["Mean", "Sum", "Median", "Count"])

        if x_col and y_col:
            # Group by
            if agg_func == "Mean":
                plot_data = df_eda.groupby(x_col)[y_col].mean().reset_index()
            elif agg_func == "Sum":
                plot_data = df_eda.groupby(x_col)[y_col].sum().reset_index()
            elif agg_func == "Median":
                plot_data = df_eda.groupby(x_col)[y_col].median().reset_index()
            else:
                plot_data = df_eda.groupby(x_col)[y_col].count().reset_index()

            # Limit categories if too many
            if len(plot_data) > 25:
                plot_data = plot_data.sort_values(by=y_col, ascending=False).head(25)
                st.caption(f"Displaying top 25 categories by {agg_func} {y_col} to maintain readability.")

            fig = create_bar_chart(plot_data, x=x_col, y=y_col, title=chart_title)
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Plot":
        c1, c2, c3 = st.columns(3)
        with c1:
            x_col = st.selectbox("Independent Feature (X):", numeric_cols)
        with c2:
            y_col = st.selectbox("Dependent Feature (Y):", [c for c in numeric_cols if c != x_col] if len(numeric_cols) > 1 else numeric_cols)
        with c3:
            color_col = st.selectbox("Color Grouping (Optional):", ["None"] + categorical_cols)

        trend = st.checkbox("Overlay OLS Regression Trendline", value=False)
        fig = create_scatter_chart(
            df_eda,
            x=x_col,
            y=y_col,
            color=color_col if color_col != "None" else None,
            trendline="ols" if trend else None,
            title=chart_title
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box Plot":
        c1, c2 = st.columns(2)
        with c1:
            y_col = st.selectbox("Continuous Variable (Y):", numeric_cols)
        with c2:
            x_col = st.selectbox("Group by Category (Optional):", ["None"] + categorical_cols)

        fig = create_box_plot(
            df_eda,
            x=x_col if x_col != "None" else None,
            y=y_col,
            title=chart_title
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Histogram":
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("Numerical Feature (X):", numeric_cols)
        with c2:
            nbins = st.slider("Histogram Bins:", min_value=10, max_value=80, value=30)

        fig = create_histogram(df_eda, x=x_col, nbins=nbins, title=chart_title)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Line Chart":
        c1, c2, c3 = st.columns(3)
        with c1:
            x_col = st.selectbox("X-Axis (Time / Order):", all_cols)
        with c2:
            y_col = st.selectbox("Y-Axis (Metric):", numeric_cols)
        with c3:
            color_col = st.selectbox("Color Group (Optional):", ["None"] + categorical_cols)

        # Aggregate if multiple entries per x
        if color_col != "None":
            grouped = df_eda.groupby([x_col, color_col])[y_col].mean().reset_index()
            fig = create_line_chart(grouped, x=x_col, y=y_col, color=color_col, title=chart_title)
        else:
            grouped = df_eda.groupby(x_col)[y_col].mean().reset_index()
            fig = create_line_chart(grouped, x=x_col, y=y_col, title=chart_title)

        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Donut Chart":
        c1, c2 = st.columns(2)
        with c1:
            cat_col = st.selectbox("Category Dimension:", categorical_cols if categorical_cols else all_cols)
        with c2:
            val_col = st.selectbox("Values Dimension:", ["Count"] + numeric_cols)

        if val_col == "Count":
            p_data = df_eda[cat_col].value_counts().reset_index()
            p_data.columns = [cat_col, "Count"]
            v_field = "Count"
        else:
            p_data = df_eda.groupby(cat_col)[val_col].sum().reset_index()
            v_field = val_col

        if len(p_data) > 10:
            top_10 = p_data.head(10)
            st.caption("Displaying top 10 categories for visual clarity.")
            p_data = top_10

        fig = create_donut_chart(p_data, names=cat_col, values=v_field, title=chart_title)
        st.plotly_chart(fig, use_container_width=True)
