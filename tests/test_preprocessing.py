"""
Unit tests for data preprocessing, cleaning, outlier detection, and safe joins.
"""

import pytest
import pandas as pd
import numpy as np

from preprocessing.data_cleaning import (
    clean_crop_production, clean_fertilizer_data, clean_soil_data,
    clean_market_prices, detect_outliers_iqr
)
from preprocessing.dataset_integration import perform_safe_join
from models.crop_recommendation import predict_crop_suitability, train_and_evaluate_crop_models


def test_detect_outliers_iqr():
    """Verifies IQR outlier detection on synthetic distribution with an extreme value."""
    df = pd.DataFrame({"val": [10, 11, 10.5, 12, 11.5, 10, 10.8, 11.2, 500.0]})
    res = detect_outliers_iqr(df, "val", multiplier=1.5)
    assert res["count"] == 1
    assert 500.0 > res["upper_bound"]


def test_clean_crop_production():
    """Verifies crop production cleaning, missing imputation, and yield formula."""
    raw = pd.DataFrame({
        "State_Name": ["  Karnataka ", "Karnataka", "Karnataka "],
        "District_Name": [" Bangalore ", "Bangalore", "Bangalore"],
        "Crop_Year": [2010, 2010, 2011],
        "Season": ["Kharif ", "Kharif", "Rabi"],
        "Crop": ["Rice ", "Rice", "Wheat"],
        "Area": [100.0, 100.0, -5.0],        # One negative area to be removed
        "Production": [np.nan, 200.0, 150.0]  # Missing production to be imputed
    })

    cleaned, summary = clean_crop_production(raw, impute_missing=True, impute_method="median_by_crop")

    # The negative area row is excluded
    assert len(cleaned) == 2
    # String whitespace is stripped and title-cased
    assert cleaned["State_Name"].iloc[0] == "Karnataka"
    assert cleaned["Crop"].iloc[0] == "Rice"
    # Missing production was imputed using median of Rice (200.0)
    assert cleaned["Production"].iloc[0] == 200.0
    # Yield = Production / Area = 200 / 100 = 2.0
    assert cleaned["Yield"].iloc[0] == 2.0


def test_clean_fertilizer_data():
    """Verifies trimming of whitespace in column names (e.g. 'Humidity ') and categorical strings."""
    raw = pd.DataFrame({
        "Temparature": [26],
        "Humidity ": [52],  # Trailing whitespace in column name
        "Moisture": [38],
        "Soil Type": ["  Sandy "],
        "Crop Type": [" Maize  "],
        "Nitrogen": [37],
        "Potassium": [0],
        "Phosphorous": [0],
        "Fertilizer Name": ["  Urea "]
    })

    cleaned, summary = clean_fertilizer_data(raw)
    assert "Humidity" in cleaned.columns
    assert "Humidity " not in cleaned.columns
    assert cleaned["Soil Type"].iloc[0] == "Sandy"
    assert cleaned["Fertilizer Name"].iloc[0] == "Urea"


def test_clean_market_prices():
    """Verifies price spread and MSP difference calculations."""
    raw = pd.DataFrame({
        "crop": ["Wheat", "Potato"],
        "crop_hindi": ["Gehun", "Aloo"],
        "modal_rs_per_quintal": [2642, 1100],
        "min": [2500, 500],
        "max": [2925, 4000],
        "msp": [2585.0, np.nan]
    })

    cleaned, summary = clean_market_prices(raw)
    assert cleaned["price_spread"].iloc[0] == 425  # 2925 - 2500
    assert cleaned["price_spread"].iloc[1] == 3500 # 4000 - 500
    assert cleaned["market_vs_msp_diff"].iloc[0] == 57 # 2642 - 2585
    assert cleaned["trading_above_msp"].iloc[0] == "Above MSP"
    assert cleaned["trading_above_msp"].iloc[1] == "No MSP Declared"


def test_perform_safe_join():
    """Verifies safe dataset joining, cardinality detection, and overlap metrics."""
    left = pd.DataFrame({"id": ["A", "B", "C"], "val_left": [1, 2, 3]})
    right = pd.DataFrame({"id": ["B", "C", "D"], "val_right": [20, 30, 40]})

    merged, diag = perform_safe_join(
        left, right, left_key="id", right_key="id", how="inner",
        left_name="Left", right_name="Right"
    )

    assert len(merged) == 2  # 'B' and 'C' match
    assert diag["overlap_keys_count"] == 2
    assert diag["cardinality"] == "One-to-One"
    assert diag["many_to_many_warning"] is False


def test_crop_suitability_pipeline():
    """Verifies end-to-end training and inference on synthetic soil dataset."""
    sample_soil = pd.DataFrame({
        "N": [140, 150, 145, 80, 85, 90] * 10,
        "P": [50, 48, 52, 20, 22, 21] * 10,
        "K": [160, 155, 165, 40, 42, 45] * 10,
        "ph": [6.5, 6.4, 6.6, 7.2, 7.1, 7.3] * 10,
        "EC": [0.5, 0.4, 0.6, 0.2, 0.3, 0.25] * 10,
        "S": [15, 14, 16, 8, 9, 8.5] * 10,
        "Cu": [3.0, 3.1, 2.9, 1.2, 1.3, 1.1] * 10,
        "Fe": [35, 36, 34, 15, 16, 14] * 10,
        "Mn": [75, 74, 76, 30, 32, 31] * 10,
        "Zn": [30, 29, 31, 12, 13, 11] * 10,
        "B": [25, 24, 26, 10, 11, 9] * 10,
        "label": ["pomegranate", "pomegranate", "pomegranate", "ragi", "ragi", "ragi"] * 10
    })

    train_out = train_and_evaluate_crop_models(sample_soil)
    assert train_out["best_model_name"] in ["Random Forest", "Decision Tree", "Gradient Boosting"]

    model_art = {
        "pipeline": train_out["results"][train_out["best_model_name"]],
        "feature_columns": ["N", "P", "K", "ph", "EC", "S", "Cu", "Fe", "Mn", "Zn", "B"]
    }
    # Test inference pipeline
    from models.crop_recommendation import load_trained_crop_model
    loaded = load_trained_crop_model()
    assert loaded is not None

    sample_test_input = {
        "N": 142.0, "P": 49.0, "K": 158.0, "ph": 6.5, "EC": 0.5,
        "S": 15.0, "Cu": 3.0, "Fe": 35.0, "Mn": 75.0, "Zn": 30.0, "B": 25.0
    }
    pred_res = predict_crop_suitability(loaded, sample_test_input)
    assert "predicted_crop" in pred_res
    assert isinstance(pred_res["predicted_crop"], str)
