"""
AgriPredict AI — Soil Nutrients & Machine Learning Crop Recommendation Studio.
Evaluates laboratory soil parameters, trains multi-class classifiers,
and provides interactive crop recommendation with class probability distributions.
"""

import streamlit as st
import pandas as pd
import numpy as np

from config.settings import THEME, CUSTOM_CSS
from preprocessing.data_loader import load_soil_data
from preprocessing.data_cleaning import clean_soil_data
from analysis.soil_analysis import (
    get_soil_summary_statistics, get_crop_nutrient_requirements, classify_soil_ph
)
from models.crop_recommendation import (
    train_and_evaluate_crop_models, load_trained_crop_model,
    predict_crop_suitability, get_feature_ranges, FEATURE_COLUMNS
)
from utils.charts import (
    create_bar_chart, create_confusion_matrix_chart,
    create_feature_importance_chart, create_radar_chart
)
from utils.helpers import render_header, render_metric_card, render_disclaimer

st.set_page_config(page_title="Soil & Crop Recommendation — AgriPredict AI", page_icon="🌾", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

render_header(
    "Soil Nutrient Analysis & Machine Learning Crop Recommendation",
    "Laboratory-grade soil nutrient profiling, multi-class algorithmic benchmarking, and interactive crop suitability prediction."
)

with st.spinner("Loading soil nutrient dataset..."):
    raw_df = load_soil_data()
    df_soil, _ = clean_soil_data(raw_df)

if df_soil.empty:
    st.error("Soil nutrients dataset could not be loaded.")
else:
    # Ensure trained model is ready (cached on disk)
    model_artifact = load_trained_crop_model(df_fallback=df_soil)

    tab_pred, tab_soil_eda, tab_model_eval = st.tabs([
        "🔮 Crop Suitability Predictor",
        "🧪 Soil Nutrients Analytics",
        "🤖 ML Model Benchmarking"
    ])

    # TAB 1: PREDICTOR
    with tab_pred:
        st.markdown("### 🌾 Soil Nutrient Input Form")
        st.write("Enter laboratory soil test values to predict the most biologically suitable agricultural crop:")

        ranges = get_feature_ranges(df_soil)

        # 3-column input layout
        col_macro, col_chem, col_micro = st.columns(3)

        with col_macro:
            st.markdown("**Primary Macronutrients (kg/ha)**")
            n_val = st.number_input("Nitrogen (N):", min_value=0.0, max_value=300.0, value=ranges.get("N", {}).get("default", 140.0), step=1.0)
            p_val = st.number_input("Phosphorus (P):", min_value=0.0, max_value=200.0, value=ranges.get("P", {}).get("default", 50.0), step=1.0)
            k_val = st.number_input("Potassium (K):", min_value=0.0, max_value=500.0, value=ranges.get("K", {}).get("default", 160.0), step=1.0)

        with col_chem:
            st.markdown("**Chemical Properties**")
            ph_val = st.number_input("Soil pH (Acidity/Alkalinity):", min_value=3.0, max_value=11.0, value=ranges.get("ph", {}).get("default", 6.5), step=0.1)
            ec_val = st.number_input("Electrical Conductivity (EC dS/m):", min_value=0.0, max_value=5.0, value=ranges.get("EC", {}).get("default", 0.5), step=0.05)
            s_val = st.number_input("Sulfur (S ppm):", min_value=0.0, max_value=100.0, value=ranges.get("S", {}).get("default", 15.0), step=0.5)

        with col_micro:
            st.markdown("**Micronutrients (ppm)**")
            cu_val = st.number_input("Copper (Cu):", min_value=0.0, max_value=50.0, value=ranges.get("Cu", {}).get("default", 3.0), step=0.2)
            fe_val = st.number_input("Iron (Fe):", min_value=0.0, max_value=250.0, value=ranges.get("Fe", {}).get("default", 35.0), step=1.0)
            mn_val = st.number_input("Manganese (Mn):", min_value=0.0, max_value=1000.0, value=ranges.get("Mn", {}).get("default", 75.0), step=1.0)
            zn_val = st.number_input("Zinc (Zn):", min_value=0.0, max_value=100.0, value=ranges.get("Zn", {}).get("default", 30.0), step=0.5)
            b_val = st.number_input("Boron (B):", min_value=0.0, max_value=100.0, value=ranges.get("B", {}).get("default", 25.0), step=0.5)

        sample_inputs = {
            "N": n_val, "P": p_val, "K": k_val, "ph": ph_val, "EC": ec_val,
            "S": s_val, "Cu": cu_val, "Fe": fe_val, "Mn": mn_val, "Zn": zn_val, "B": b_val
        }

        if st.button("🌱 Predict Recommended Crop", type="primary"):
            if model_artifact:
                pred_result = predict_crop_suitability(model_artifact, sample_inputs)
                pred_crop = pred_result["predicted_crop"]
                confidence = pred_result.get("confidence", 0.0)

                st.markdown("---")
                st.markdown("### 🎯 Model Recommendation Output")
                
                res_col1, res_col2 = st.columns([1, 1.5])
                with res_col1:
                    render_metric_card("Recommended Crop", f"🌾 {pred_crop}", f"Model Confidence: {confidence}%", THEME["primary"])
                    st.markdown("<br/>", unsafe_allow_html=True)
                    st.write(f"The algorithm identifies this soil profile as optimal for **{pred_crop}** cultivation based on chemical and nutrient balance.")

                with res_col2:
                    if "probabilities" in pred_result:
                        prob_df = pd.DataFrame(list(pred_result["probabilities"].items()), columns=["Crop", "Suitability Probability (%)"])
                        fig_prob = create_bar_chart(prob_df, x="Crop", y="Suitability Probability (%)", title="Class Probability Distribution")
                        st.plotly_chart(fig_prob, use_container_width=True)

                # Radar profile comparison
                st.markdown("#### 🕸️ Soil Nutrient Radar Comparison")
                radar_cats = ["N", "P", "K", "Fe", "Mn", "Zn", "B"]
                radar_vals = [sample_inputs[k] for k in radar_cats]
                fig_radar = create_radar_chart(radar_cats, radar_vals, title=f"Sample Soil Nutrient Signature")
                st.plotly_chart(fig_radar, use_container_width=True)

                render_disclaimer("crop")
            else:
                st.error("Model artifact could not be initialized.")

    # TAB 2: SOIL EDA
    with tab_soil_eda:
        st.markdown("### 📊 Soil Nutrients Chemical Distributions")
        stats_table = get_soil_summary_statistics(df_soil)
        st.dataframe(stats_table, use_container_width=True)

        col_ph, col_reqs = st.columns(2)
        with col_ph:
            ph_df = classify_soil_ph(df_soil)
            fig_ph = create_bar_chart(ph_df, x="pH_Category", y="Sample_Count", title="Distribution of Soil pH Categories")
            st.plotly_chart(fig_ph, use_container_width=True)

        with col_reqs:
            crop_reqs = get_crop_nutrient_requirements(df_soil)
            fig_npk = create_bar_chart(crop_reqs, x="label", y="N", title="Average Nitrogen (N) Requirement by Crop")
            st.plotly_chart(fig_npk, use_container_width=True)

        st.markdown("### 📋 Mean Nutrient Profiles by Crop Label")
        st.dataframe(crop_reqs, use_container_width=True)

    # TAB 3: MODEL BENCHMARKING
    with tab_model_eval:
        st.markdown("### 🤖 Algorithmic Benchmarking (80/20 Stratified Split)")
        
        if st.button("🔄 Retrain and Benchmark Classifiers"):
            with st.spinner("Training Random Forest, Decision Tree, and Gradient Boosting..."):
                train_out = train_and_evaluate_crop_models(df_soil)
                st.session_state["crop_train_out"] = train_out
                st.success(f"Benchmarking complete! Optimal Model: {train_out['best_model_name']}")

        train_out = st.session_state.get("crop_train_out", None)
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
            
            c_cm, c_fi = st.columns(2)
            with c_cm:
                if "confusion_matrix" in best_sc and "classes" in best_sc:
                    fig_cm = create_confusion_matrix_chart(
                        best_sc["confusion_matrix"],
                        best_sc["classes"],
                        title=f"Confusion Matrix ({train_out['best_model_name']})"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

            with c_fi:
                if "feature_importances" in best_sc:
                    fi_dict = best_sc["feature_importances"]
                    fig_fi = create_feature_importance_chart(
                        list(fi_dict.keys()),
                        list(fi_dict.values()),
                        title="Soil Parameter Relative Importance"
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)
