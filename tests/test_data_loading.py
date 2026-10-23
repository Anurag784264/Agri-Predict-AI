"""
Unit tests for data loading, encoding fallbacks, and inspection metrics.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from preprocessing.data_loader import safe_read_csv, inspect_dataset, load_weather_city
from config.settings import resolve_dataset_path, DATASET_SPECS


def test_safe_read_csv_success(tmp_path):
    """Verifies that safe_read_csv successfully reads standard UTF-8 CSVs."""
    csv_file = tmp_path / "test_sample.csv"
    csv_file.write_text("col_a,col_b\n1,alpha\n2,beta\n", encoding="utf-8")

    df = safe_read_csv(csv_file)
    assert len(df) == 2
    assert list(df.columns) == ["col_a", "col_b"]
    assert df["col_a"].sum() == 3


def test_safe_read_csv_missing_file():
    """Verifies that safe_read_csv raises FileNotFoundError for non-existent paths."""
    with pytest.raises(FileNotFoundError):
        safe_read_csv("non_existent_directory/non_existent_file.csv")


def test_inspect_dataset_empty():
    """Verifies inspect_dataset handles empty or None DataFrames gracefully."""
    empty_df = pd.DataFrame()
    meta = inspect_dataset(empty_df)
    assert meta["status"] == "empty"
    assert meta["rows"] == 0
    assert meta["columns"] == 0
    assert meta["duplicate_count"] == 0


def test_inspect_dataset_populated():
    """Verifies accurate detection of column types, duplicates, and missing values."""
    sample_df = pd.DataFrame({
        "num_a": [1.0, 2.0, np.nan, 4.0, 4.0],
        "cat_b": ["x", "y", "z", "x", "x"],
        "num_c": [10, 20, 30, 40, 40]
    })
    meta = inspect_dataset(sample_df)

    assert meta["status"] == "ok"
    assert meta["rows"] == 5
    assert meta["columns"] == 3
    assert "num_a" in meta["numeric_columns"]
    assert "cat_b" in meta["categorical_columns"]
    assert meta["missing_counts"]["num_a"] == 1
    assert meta["duplicate_count"] == 1  # Row index 3 and 4 are identical


def test_resolve_dataset_path():
    """Verifies dataset path resolution for configured dataset keys."""
    crop_path = resolve_dataset_path("crop_production")
    assert crop_path is not None
    assert crop_path.exists()

    invalid_path = resolve_dataset_path("unknown_invalid_dataset_key")
    assert invalid_path is None


def test_load_weather_invalid_city():
    """Verifies that loading an invalid city returns an empty DataFrame rather than crashing."""
    df_invalid = load_weather_city("atlantis_city_xyz")
    assert isinstance(df_invalid, pd.DataFrame)
    assert df_invalid.empty
