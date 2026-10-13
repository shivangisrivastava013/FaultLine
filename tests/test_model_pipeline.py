"""
Unit tests for FaultLine Model Pipelines.
"""

import pandas as pd
import numpy as np
import pytest
from simulation.generate_data import generate_delivery_records
from src.models.pipeline import (
    create_linear_baseline_pipeline,
    create_tree_pipeline,
    create_degraded_pipeline,
    evaluate_pipeline,
    ALL_FEATURES,
    TARGET_COLUMN,
)


@pytest.fixture
def sample_data():
    df = generate_delivery_records(num_samples=600, seed=42)
    return df


def test_tree_pipeline_training_and_evaluation(sample_data):
    pipe = create_tree_pipeline(max_iter=60)
    X = sample_data[ALL_FEATURES]
    y = sample_data[TARGET_COLUMN].values
    pipe.fit(X, y)

    metrics = evaluate_pipeline(pipe, sample_data)
    assert metrics["overall_mae"] < 5.0
    assert metrics["overall_r2"] > 0.7
    assert "low" in metrics["segment_mae_traffic"]


def test_linear_baseline_training(sample_data):
    pipe = create_linear_baseline_pipeline()
    X = sample_data[ALL_FEATURES]
    y = sample_data[TARGET_COLUMN].values
    pipe.fit(X, y)

    metrics = evaluate_pipeline(pipe, sample_data)
    assert metrics["overall_mae"] > 0.0
    assert metrics["overall_rmse"] > 0.0


def test_degraded_model_exhibits_higher_error(sample_data):
    tree_pipe = create_tree_pipeline(max_iter=50)
    degraded_pipe = create_degraded_pipeline()

    X = sample_data[ALL_FEATURES]
    y = sample_data[TARGET_COLUMN].values

    tree_pipe.fit(X, y)
    degraded_pipe.fit(X.head(30), y[:30])

    good_eval = evaluate_pipeline(tree_pipe, sample_data)
    degraded_eval = evaluate_pipeline(degraded_pipe, sample_data)

    # Degraded model must have substantially higher MAE
    assert degraded_eval["overall_mae"] > good_eval["overall_mae"]
