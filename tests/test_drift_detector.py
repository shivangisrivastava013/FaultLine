"""
Unit tests for FaultLine Drift & Data Quality Detector.
"""

import pandas as pd
import numpy as np
import pytest
from src.monitoring.drift_detector import DriftDetector, calculate_psi


@pytest.fixture
def drift_detector(tmp_path):
    ref_df = pd.DataFrame({
        "distance_km": np.random.uniform(1.0, 15.0, 200),
        "traffic_level": np.random.choice(["low", "medium", "high", "jam"], 200),
        "weather": np.random.choice(["clear", "rain", "storm", "snow"], 200),
        "restaurant_load": np.random.uniform(0.1, 0.9, 200),
        "hour_of_day": np.random.randint(0, 24, 200),
        "delivery_duration_minutes": np.random.uniform(15.0, 50.0, 200),
    })
    ref_path = tmp_path / "ref.parquet"
    ref_df.to_parquet(ref_path)
    return DriftDetector(reference_data_path=str(ref_path))


def test_data_quality_detects_missing_and_unknowns(drift_detector):
    bad_df = pd.DataFrame({
        "distance_km": [-5.0, 10.0, np.nan],  # out of range + missing
        "traffic_level": ["low", "warp_speed", "medium"],  # unknown category
        "weather": ["clear", "clear", "clear"],
        "restaurant_load": [0.5, 1.8, 0.2],  # out of range > 1.0
        "hour_of_day": [10, 28, 5],  # out of range > 23
    })

    issues = drift_detector.check_data_quality(bad_df)
    assert issues["has_quality_issues"] is True
    assert "distance_km" in issues["missing_values"]
    assert "traffic_level" in issues["unknown_categories"]
    assert "distance_km" in issues["out_of_range"]
    assert "restaurant_load" in issues["out_of_range"]
    assert "hour_of_day" in issues["out_of_range"]


def test_distribution_drift_on_shifted_features(drift_detector):
    # Severely shifted continuous distribution (distances 80-120km vs 1-15km)
    shifted_df = pd.DataFrame({
        "distance_km": np.random.uniform(80.0, 120.0, 100),
        "traffic_level": np.random.choice(["low", "medium"], 100),
        "weather": np.random.choice(["clear", "rain"], 100),
        "restaurant_load": np.random.uniform(0.1, 0.9, 100),
        "hour_of_day": np.random.randint(0, 24, 100),
    })

    res = drift_detector.detect_distribution_drift(shifted_df)
    assert res["drift_detected"] is True
    assert "distance_km" in res["drifted_features"]


def test_psi_calculation():
    base = np.array(["a", "b", "c"] * 100)
    identical = np.array(["a", "b", "c"] * 100)
    shifted = np.array(["a"] * 280 + ["b"] * 10 + ["c"] * 10)

    psi_same = calculate_psi(base, identical)
    psi_diff = calculate_psi(base, shifted)

    assert psi_same < 0.05
    assert psi_diff > 0.25
