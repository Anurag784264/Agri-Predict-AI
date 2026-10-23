"""
AgriPredict AI — Unified Machine Learning Model Laboratory & Benchmarking Studio.
Supports multi-class classification, regression, and unsupervised K-Means clustering
with full evaluation scorecards and diagnostic visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np

from config.settings import THEME, CUSTOM_CSS
from preprocessing.data_loader import load_crop_production, load_fertilizer_data, load_soil_data
from preprocessing.weather_processing import get_all_cities_weather_comparison
from models.crop_recommendation import train_and_evaluate_crop_models
from models.fertilizer_model import train_and_evaluate_fertilizer_models
from models.yield_prediction import train_and_evaluate_yield_models
from models.clustering import perform_kmeans_clustering, compute_elbow_curve
from utils.charts import (
    create_bar_chart, create_line_chart, create_scatter_chart,
    create_confusion_matrix_chart, create_feature_importance_chart, apply_theme_layout
)
from utils.helpers import render_header, render_metric_card, get_csv_download_bytes

import plotly.graph_objects as go

st.set_page_config(page_title="ML Model Lab — AgriPredict AI", page_icon="🌾", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

render_header(
    "Machine Learning Model Laboratory & Benchmarking",
    "Comprehensive empirical laboratory for training, validating, and comparing classification, regression, and clustering algorithms."
)

task_type = st.radio(
    "Select Machine Learning Paradigm:",
    [
        "1. Classification — Soil Nutrient Crop Recommendation",
        "2. Classification — Chemical Fertilizer Selection",
        "3. Regression — Agricultural Crop Yield Prediction",
        "4. Unsupervised Clustering — K-Means Exploratory Segmentation"
    ]
)

st.markdown("---")

# 1. CROP RECOMMENDATION CLASSIFICATION
if "Crop Recommendation" in task_type:
    st.markdown("### 🌾 Multi-Class Classification: Crop Recommendation")
    st.write("Target: `label` (6 distinct crop classes) | Features: 11 Laboratory Soil Nutrients (N, P, K, pH, EC, S, Cu, Fe, Mn, Zn, B).")
    
    df_soil = load_soil_data()
    if df_soil.empty:
        st.error("Soil dataset could not be loaded.")
    else:
        st.dataframe(df_soil.head(5), use_container_width=True)

        if st.button("🚀 Train & Benchmark Classification Models (80/20 Stratified)", type="primary"):
            with st.spinner("Training Random Forest, Decision Tree, and Gradient Boosting..."):
                res = train_and_evaluate_crop_models(df_soil)
                st.session_state["lab_crop_res"] = res

        if "lab_crop_res" in st.session_state:
            res = st.session_state["lab_crop_res"]
            st.success(f"Benchmarking complete! Optimal Classifier: **{res['best_model_name']}**")

            eval_list = []
            for name, sc in res["results"].items():
                eval_list.append({
                    "Algorithm": name,
                    "Accuracy": sc.get("accuracy"),
                    "F1 Score (Weighted)": sc.get("f1_weighted"),
                    "Precision (Weighted)": sc.get("precision_weighted"),
                    "Recall (Weighted)": sc.get("recall_weighted"),
                    "F1 Score (Macro)": sc.get("f1_macro")
                })
            df_eval = pd.DataFrame(eval_list)
            st.dataframe(df_eval, use_container_width=True)

            best_sc = res["results"][res["best_model_name"]]
            c_cm, c_fi = st.columns(2)
            with c_cm:
                if "confusion_matrix" in best_sc and "classes" in best_sc:
                    fig_cm = create_confusion_matrix_chart(best_sc["confusion_matrix"], best_sc["classes"], title=f"Confusion Matrix — {res['best_model_name']}")
                    st.plotly_chart(fig_cm, use_container_width=True)
            with c_fi:
                if "feature_importances" in best_sc:
                    fi_d = best_sc["feature_importances"]
                    fig_fi = create_feature_importance_chart(list(fi_d.keys()), list(fi_d.values()), title="Nutrient Relative Importance")
                    st.plotly_chart(fig_fi, use_container_width=True)

            st.download_button("📥 Download Model Benchmark Scorecard (CSV)", get_csv_download_bytes(df_eval), "crop_recommendation_benchmark.csv", "text/csv")

# 2. FERTILIZER SELECTION CLASSIFICATION
elif "Fertilizer Selection" in task_type:
    st.markdown("### 🧪 Multi-Class Classification: Fertilizer Selection")
    st.write("Target: `Fertilizer Name` (7 classes) | Mixed Numerical (N, P, K, Temp, Humidity, Moisture) & Categorical Features (Soil Type, Crop Type).")
    
    df_fert = load_fertilizer_data()
    if df_fert.empty:
        st.error("Fertilizer dataset could not be loaded.")
    else:
        st.dataframe(df_fert.head(5), use_container_width=True)

        if st.button("🚀 Train & Benchmark Fertilizer Models", type="primary"):
            with st.spinner("Training Random Forest, Decision Tree, and Gradient Boosting with ColumnTransformer..."):
                res = train_and_evaluate_fertilizer_models(df_fert)
                st.session_state["lab_fert_res"] = res

        if "lab_fert_res" in st.session_state:
            res = st.session_state["lab_fert_res"]
            st.success(f"Benchmarking complete! Optimal Model: **{res['best_model_name']}**")

            eval_list = []
            for name, sc in res["results"].items():
                eval_list.append({
                    "Algorithm": name,
                    "Accuracy": sc.get("accuracy"),
                    "F1 Score (Weighted)": sc.get("f1_weighted"),
                    "Precision (Weighted)": sc.get("precision_weighted"),
                    "Recall (Weighted)": sc.get("recall_weighted"),
                    "F1 Score (Macro)": sc.get("f1_macro")
                })
            df_eval = pd.DataFrame(eval_list)
            st.dataframe(df_eval, use_container_width=True)

            best_sc = res["results"][res["best_model_name"]]
            if "confusion_matrix" in best_sc and "classes" in best_sc:
                fig_cm = create_confusion_matrix_chart(best_sc["confusion_matrix"], best_sc["classes"], title=f"Confusion Matrix — {res['best_model_name']}")
                st.plotly_chart(fig_cm, use_container_width=True)

            st.download_button("📥 Download Fertilizer Benchmark Scorecard (CSV)", get_csv_download_bytes(df_eval), "fertilizer_benchmark.csv", "text/csv")

# 3. YIELD REGRESSION
elif "Crop Yield" in task_type:
    st.markdown("### 🌾 Continuous Regression: Agricultural Crop Yield Prediction")
    st.write("Target: `Yield` (Tonnes per Hectare = Production / Area) | Features: State, Crop, Season, Area.")
    
    df_crop = load_crop_production(nrows=30000)
    if df_crop.empty:
        st.error("Crop production dataset could not be loaded.")
    else:
        st.dataframe(df_crop.head(5), use_container_width=True)

        if st.button("🚀 Train & Benchmark Regression Models", type="primary"):
            with st.spinner("Training Linear Regression, Decision Tree, and Random Forest Regressors..."):
                res = train_and_evaluate_yield_models(df_crop)
                st.session_state["lab_yield_res"] = res

        if "lab_yield_res" in st.session_state:
            res = st.session_state["lab_yield_res"]
            st.success(f"Benchmarking complete! Top Regressor: **{res['best_model_name']}** (Sample Size: {res['sample_size']:,})")

            reg_list = []
            for name, sc in res["results"].items():
                reg_list.append({
                    "Algorithm": name,
                    "R² Score": sc.get("r2_score"),
                    "RMSE (Tonnes/Ha)": sc.get("rmse"),
                    "MAE (Tonnes/Ha)": sc.get("mae")
                })
            df_reg = pd.DataFrame(reg_list)
            st.dataframe(df_reg, use_container_width=True)

            st.download_button("📥 Download Yield Regression Benchmark (CSV)", get_csv_download_bytes(df_reg), "yield_regression_benchmark.csv", "text/csv")

# 4. UNSUPERVISED CLUSTERING
elif "Clustering" in task_type:
    st.markdown("### 🧩 Unsupervised K-Means Clustering & PCA Projection")
    
    cluster_ds = st.selectbox("Select dataset for clustering:", ["Soil Nutrients Profile", "Weather Multi-City Climatology"])

    df_cluster = pd.DataFrame()
    if cluster_ds == "Soil Nutrients Profile":
        df_cluster = load_soil_data()
        feature_opts = ["N", "P", "K", "ph", "EC", "S", "Cu", "Fe", "Mn", "Zn", "B"]
    else:
        df_cluster = get_all_cities_weather_comparison()
        feature_opts = ["tempC", "maxtempC", "mintempC", "precipMM", "humidity", "windspeedKmph"]

    selected_features = st.multiselect("Select numerical features for clustering:", feature_opts, default=feature_opts[:5])
    k_val = st.slider("Select number of clusters (k):", min_value=2, max_value=8, value=3)

    if st.button("🔍 Execute K-Means Clustering", type="primary"):
        with st.spinner("Scaling features and running K-Means..."):
            clustered_df, diagnostics = perform_kmeans_clustering(df_cluster, selected_features, n_clusters=k_val)
            st.session_state["clustered_df"] = clustered_df
            st.session_state["cluster_diag"] = diagnostics

    if "cluster_diag" in st.session_state:
        diag = st.session_state["cluster_diag"]
        c_df = st.session_state["clustered_df"]

        if diag.get("status") == "success":
            st.markdown("### 📊 Clustering Quality Indicators")
            qc1, qc2, qc3 = st.columns(3)
            with qc1:
                render_metric_card("Clusters (k)", str(diag["n_clusters"]), "Configured partitions", THEME["primary"])
            with qc2:
                sil = diag.get("silhouette_score", "N/A")
                render_metric_card("Silhouette Score", str(sil), "Cluster cohesion (-1 to +1)", THEME["primary_light"])
            with qc3:
                db = diag.get("davies_bouldin_score", "N/A")
                render_metric_card("Davies-Bouldin Index", str(db), "Lower indicates better separation", THEME["accent_gold"])

            st.markdown("<br/>", unsafe_allow_html=True)

            col_pca, col_elbow = st.columns(2)
            with col_pca:
                fig_pca = create_scatter_chart(
                    c_df,
                    x="PCA_Dim1",
                    y="PCA_Dim2",
                    color="Cluster",
                    title="2D PCA Projection of Agricultural Clusters"
                )
                st.plotly_chart(fig_pca, use_container_width=True)

            with col_elbow:
                elbow_df = compute_elbow_curve(df_cluster, selected_features, max_k=8)
                if not elbow_df.empty:
                    fig_elbow = create_line_chart(elbow_df, x="k", y="Inertia", title="Elbow Inertia Curve (k=2..8)")
                    st.plotly_chart(fig_elbow, use_container_width=True)

            st.markdown("### 📋 Cluster Center Profiles & Membership")
            st.dataframe(diag["cluster_summary"], use_container_width=True)

            st.download_button("📥 Download Clustered Dataset (CSV)", get_csv_download_bytes(c_df), "clustered_agricultural_data.csv", "text/csv")
        else:
            st.error(diag.get("message", "Clustering failed."))
