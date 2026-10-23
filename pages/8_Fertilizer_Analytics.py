"""
AgriPredict AI — Fertilizer Utilization & Recommendation Intelligence Studio.
Analyzes chemical fertilizer distributions, trains multi-class classification pipelines,
and provides interactive fertilizer grade recommendations with safety disclaimers.
"""

import streamlit as st
import pandas as pd
import numpy as np

from config.settings import THEME, CUSTOM_CSS
from preprocessing.data_loader import load_fertilizer_data
from preprocessing.data_cleaning import clean_fertilizer_data
from analysis.fertilizer_analysis import (
    get_fertilizer_distribution, get_nutrient_by_fertilizer, get_fertilizer_by_soil_and_crop
)
from models.fertilizer_model import (
    train_and_evaluate_fertilizer_models, load_trained_fertilizer_model,
    predict_fertilizer, get_fertilizer_categories
)
from utils.charts import create_bar_chart, create_confusion_matrix_chart, apply_theme_layout
from utils.helpers import render_header, render_metric_card, render_disclaimer

import plotly.graph_objects as go

st.set_page_config(page_title="Fertilizer Analytics — AgriPredict AI", page_icon="🌾", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

render_header(
    "Fertilizer Analytics & Prescriptive Recommendation Engine",
    "Chemical fertilizer profiling, multi-class model benchmarking, and interactive fertilizer formulation recommendation."
)

with st.spinner("Loading fertilizer dataset..."):
    raw_df = load_fertilizer_data()
    df_fert, _ = clean_fertilizer_data(raw_df)

if df_fert.empty:
    st.error("Fertilizer dataset could not be loaded.")
else:
    model_artifact = load_trained_fertilizer_model(df_fallback=df_fert)
    categories = get_fertilizer_categories(df_fert)

    tab_pred, tab_eda, tab_bench = st.tabs([
        "🧪 Fertilizer Recommendation Predictor",
        "📊 Formulation & Usage Analytics",
        "🤖 ML Model Benchmarking"
    ])

    # TAB 1: PREDICTOR
    with tab_pred:
        st.markdown("### 🌾 Agricultural Conditions Input Form")
        st.write("Specify soil type, crop classification, environmental metrics, and current soil nutrient levels:")

        c_soil, c_crop = st.columns(2)
        with c_soil:
            soil_types = categories.get("soil_types", ["Clayey", "Loamy", "Black", "Red", "Sandy"])
            sel_soil = st.selectbox("Soil Classification:", soil_types, index=0)
        with c_crop:
            crop_types = categories.get("crop_types", ["Paddy", "Wheat", "Maize", "Sugarcane", "Cotton"])
            sel_crop = st.selectbox("Crop Variety:", crop_types, index=0)

        c_env1, c_env2, c_env3 = st.columns(3)
        with c_env1:
            temp_in = st.number_input("Ambient Temperature (°C):", min_value=10.0, max_value=50.0, value=28.0, step=0.5)
        with c_env2:
            hum_in = st.number_input("Relative Humidity (%):", min_value=10.0, max_value=100.0, value=55.0, step=1.0)
        with c_env3:
            moist_in = st.number_input("Soil Moisture Content (%):", min_value=10.0, max_value=100.0, value=40.0, step=1.0)

        c_n, c_k, c_p = st.columns(3)
        with c_n:
            n_in = st.number_input("Current Soil Nitrogen (N):", min_value=0.0, max_value=150.0, value=25.0, step=1.0)
        with c_k:
            k_in = st.number_input("Current Soil Potassium (K):", min_value=0.0, max_value=150.0, value=10.0, step=1.0)
        with c_p:
            p_in = st.number_input("Current Soil Phosphorus (P):", min_value=0.0, max_value=150.0, value=20.0, step=1.0)

        input_sample = {
            "Soil Type": sel_soil,
            "Crop Type": sel_crop,
            "Temparature": temp_in,
            "Humidity": hum_in,
            "Moisture": moist_in,
            "Nitrogen": n_in,
            "Potassium": k_in,
            "Phosphorous": p_in
        }

        if st.button("🌱 Generate Fertilizer Recommendation", type="primary"):
            if model_artifact:
                pred_res = predict_fertilizer(model_artifact, input_sample)
                fert_name = pred_res["predicted_fertilizer"]
                conf = pred_res.get("confidence", 0.0)

                st.markdown("---")
                st.markdown("### 🎯 Recommended Fertilizer Formulation")

                p_c1, p_c2 = st.columns([1, 1.5])
                with p_c1:
                    render_metric_card("Recommended Formulation", f"🧪 {fert_name}", f"Confidence: {conf}%", THEME["primary"])
                    st.markdown("<br/>", unsafe_allow_html=True)
                    st.write(f"Recommended formulation **{fert_name}** targets the nutrient balance for **{sel_crop}** on **{sel_soil}** soil under current ambient conditions.")

                with p_c2:
                    if "probabilities" in pred_res:
                        prob_df = pd.DataFrame(list(pred_res["probabilities"].items()), columns=["Fertilizer", "Probability (%)"])
                        fig_prob = create_bar_chart(prob_df, x="Fertilizer", y="Probability (%)", title="Fertilizer Recommendation Probabilities")
                        st.plotly_chart(fig_prob, use_container_width=True)

                render_disclaimer("fertilizer")
            else:
                st.error("Fertilizer model artifact could not be initialized.")

    # TAB 2: EDA
    with tab_eda:
        st.markdown("### 📊 Fertilizer Formulation Patterns")
        c_dist, c_nutr = st.columns(2)

        with c_dist:
            dist_df = get_fertilizer_distribution(df_fert)
            fig_dist = create_bar_chart(dist_df, x="Fertilizer", y="Count", title="Distribution of Fertilizer Formulations in Dataset")
            st.plotly_chart(fig_dist, use_container_width=True)

        with c_nutr:
            nutr_df = get_nutrient_by_fertilizer(df_fert)
            fig_nutr = create_bar_chart(nutr_df, x="Fertilizer Name", y="Nitrogen", title="Mean Soil Nitrogen Level Associated with Each Fertilizer")
            st.plotly_chart(fig_nutr, use_container_width=True)

        st.markdown("### 🗺️ Fertilizer vs Soil and Crop Cross-Tabulations")
        soil_tab, crop_tab = get_fertilizer_by_soil_and_crop(df_fert)

        c_st, c_ct = st.columns(2)
        with c_st:
            st.markdown("**Soil Type vs Fertilizer Cross-Tabulation:**")
            st.dataframe(soil_tab, use_container_width=True)
        with c_ct:
            st.markdown("**Crop Type vs Fertilizer Cross-Tabulation:**")
            st.dataframe(crop_tab, use_container_width=True)

    # TAB 3: BENCHMARKING
    with tab_bench:
        st.markdown("### 🤖 Algorithmic Benchmarking (80/20 Stratified Split)")

        if st.button("🔄 Retrain and Benchmark Fertilizer Models"):
            with st.spinner("Training Random Forest, Decision Tree, and Gradient Boosting..."):
                train_out = train_and_evaluate_fertilizer_models(df_fert)
                st.session_state["fert_train_out"] = train_out
                st.success(f"Benchmarking complete! Optimal Model: {train_out['best_model_name']}")

        train_out = st.session_state.get("fert_train_out", None)
        if not train_out and model_artifact:
            train_out = {"best_model_name": model_artifact["model_name"], "results": {model_artifact["model_name"]: model_artifact["scorecard"]}}

        if train_out:
            eval_rows = []
            for name, sc in train_out["results"].items():
                eval_rows.append({
                    "Algorithm": name,
                    "Accuracy": f"{sc.get('accuracy', 0.0):.4f}",
                    "F1 Score (Weighted)": f"{sc.get('f1_weighted', 0.0):.4f}",
                    "Precision (Weighted)": f"{sc.get('precision_weighted', 0.0):.4f}",
                    "Recall (Weighted)": f"{sc.get('recall_weighted', 0.0):.4f}"
                })
            st.dataframe(pd.DataFrame(eval_rows), use_container_width=True)

            best_sc = train_out["results"].get(train_out["best_model_name"], {})
            if "confusion_matrix" in best_sc and "classes" in best_sc:
                fig_cm = create_confusion_matrix_chart(
                    best_sc["confusion_matrix"],
                    best_sc["classes"],
                    title=f"Confusion Matrix ({train_out['best_model_name']})"
                )
                st.plotly_chart(fig_cm, use_container_width=True)
