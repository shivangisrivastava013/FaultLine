"""
FaultLine Model Pipeline definitions and utilities.

Encapsulates complete preprocessing + estimator pipelines for:
- Baseline linear regressor (Ridge)
- Tree-based regressor (HistGradientBoostingRegressor) - Champion model
- Deliberately degraded candidate regressor - Candidate model (for regression & rollback tests)
"""

from __future__ import annotations

import json
import os
import joblib
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


NUMERICAL_FEATURES = ["distance_km", "hour_of_day", "restaurant_load"]
CATEGORICAL_FEATURES = ["traffic_level", "weather"]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "delivery_duration_minutes"


def build_preprocessor() -> ColumnTransformer:
    """Creates scikit-learn preprocessor with scaling and one-hot encoding."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def create_linear_baseline_pipeline(alpha: float = 1.0) -> Pipeline:
    """Builds preprocessor + Ridge linear baseline."""
    preprocessor = build_preprocessor()
    model = Ridge(alpha=alpha, random_state=42)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model),
    ])


def create_tree_pipeline(
    max_iter: int = 150,
    max_leaf_nodes: int = 31,
    learning_rate: float = 0.08,
) -> Pipeline:
    """Builds preprocessor + HistGradientBoostingRegressor (champion model)."""
    preprocessor = build_preprocessor()
    model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        learning_rate=learning_rate,
        random_state=42,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model),
    ])


def create_degraded_pipeline() -> Pipeline:
    """
    Builds a deliberately degraded candidate model.
    Degradation characteristics:
    - Severely restricted depth (max_leaf_nodes=2, max_iter=5)
    - Underfits complex interactions between traffic and weather.
    - Used to test performance regression detection and automated rollback.
    """
    preprocessor = build_preprocessor()
    model = HistGradientBoostingRegressor(
        max_iter=5,
        max_leaf_nodes=2,
        learning_rate=0.01,
        random_state=42,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model),
    ])


def evaluate_pipeline(
    pipeline: BaseEstimator,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Evaluates model predictions against ground truth.
    Computes overall MAE, RMSE, R2, and slice MAE across traffic categories.
    """
    X = df[ALL_FEATURES]
    y_true = df[TARGET_COLUMN].values
    y_pred = pipeline.predict(X)

    overall_mae = float(mean_absolute_error(y_true, y_pred))
    overall_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    overall_r2 = float(r2_score(y_true, y_pred))

    segment_mae: Dict[str, float] = {}
    for traffic_val in ["low", "medium", "high", "jam"]:
        mask = df["traffic_level"] == traffic_val
        if mask.sum() > 0:
            seg_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            segment_mae[traffic_val] = float(round(seg_mae, 3))

    return {
        "overall_mae": round(overall_mae, 3),
        "overall_rmse": round(overall_rmse, 3),
        "overall_r2": round(overall_r2, 4),
        "segment_mae_traffic": segment_mae,
        "sample_count": len(df),
    }


def save_model_artifact(
    pipeline: BaseEstimator,
    metadata: Dict[str, Any],
    output_dir: str,
    model_name: str = "model.joblib",
) -> str:
    """Persists pipeline bundle and metadata JSON to disk."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_name)
    meta_path = os.path.join(output_dir, "metadata.json")

    joblib.dump(pipeline, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return model_path


def load_model_artifact(model_dir: str, model_name: str = "model.joblib") -> Tuple[BaseEstimator, Dict[str, Any]]:
    """Loads pipeline and metadata from disk."""
    model_path = os.path.join(model_dir, model_name)
    meta_path = os.path.join(model_dir, "metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    pipeline = joblib.load(model_path)
    metadata: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return pipeline, metadata
