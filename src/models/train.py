"""
FaultLine Model Training & Experiment Tracking.

Trains:
1. Linear Baseline (Ridge)
2. Champion Tree Model (HistGradientBoostingRegressor)
3. Degraded Candidate Model (underfit HistGradientBoostingRegressor for regression testing)

Logs parameters, metrics, slice errors, and model artifacts to MLflow and local registry.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from simulation.generate_data import generate_and_save_dataset_splits
from src.models.pipeline import (
    ALL_FEATURES,
    TARGET_COLUMN,
    create_linear_baseline_pipeline,
    create_tree_pipeline,
    create_degraded_pipeline,
    evaluate_pipeline,
    save_model_artifact,
)


def train_and_evaluate_models(
    data_dir: str = "data",
    artifacts_dir: str = "artifacts/models",
    mlflow_tracking_uri: str = "sqlite:///data/mlflow.db",
    seed: int = 42,
) -> Dict[str, Any]:
    """Trains and compares models, logs artifacts to MLflow and disk."""
    print("=== FaultLine: Training Delivery Time Prediction Models ===")

    train_path = os.path.join(data_dir, "train.parquet")
    val_path = os.path.join(data_dir, "val.parquet")
    test_path = os.path.join(data_dir, "test.parquet")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        print("Data partitions not found. Generating fresh synthetic dataset...")
        train_df, val_df, test_df = generate_and_save_dataset_splits(output_dir=data_dir, seed=seed)
    else:
        print(f"Loading existing datasets from {data_dir}...")
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)

    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET_COLUMN].values

    # Setup MLflow if available
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("faultline_delivery_service")
        has_mlflow = True
    except Exception as e:
        print(f"MLflow not available or failed to initialize: {e}. Skipping MLflow logging.")
        has_mlflow = False

    results: Dict[str, Any] = {}

    # 1. Linear Baseline
    print("\n[1/3] Training Linear Baseline (Ridge)...")
    linear_pipe = create_linear_baseline_pipeline()
    linear_pipe.fit(X_train, y_train)
    val_eval_linear = evaluate_pipeline(linear_pipe, val_df)
    test_eval_linear = evaluate_pipeline(linear_pipe, test_df)

    linear_meta = {
        "model_version": "v0_linear",
        "model_type": "RidgeLinearRegression",
        "status": "baseline",
        "val_metrics": val_eval_linear,
        "test_metrics": test_eval_linear,
        "features": ALL_FEATURES,
        "seed": seed,
    }
    save_model_artifact(linear_pipe, linear_meta, os.path.join(artifacts_dir, "baseline_linear"))
    results["baseline_linear"] = linear_meta

    if has_mlflow:
        with mlflow.start_run(run_name="baseline_linear_v0"):
            mlflow.log_params({"model_type": "Ridge", "alpha": 1.0, "seed": seed})
            mlflow.log_metrics({
                "val_mae": val_eval_linear["overall_mae"],
                "test_mae": test_eval_linear["overall_mae"],
                "test_r2": test_eval_linear["overall_r2"],
            })
            for k, v in test_eval_linear["segment_mae_traffic"].items():
                mlflow.log_metric(f"test_mae_traffic_{k}", v)

    # 2. Champion Tree Model (v1)
    print("\n[2/3] Training Champion Tree Model (HistGradientBoostingRegressor)...")
    tree_pipe = create_tree_pipeline()
    tree_pipe.fit(X_train, y_train)
    val_eval_tree = evaluate_pipeline(tree_pipe, val_df)
    test_eval_tree = evaluate_pipeline(tree_pipe, test_df)

    champion_meta = {
        "model_version": "champion_v1",
        "model_type": "HistGradientBoostingRegressor",
        "status": "champion",
        "val_metrics": val_eval_tree,
        "test_metrics": test_eval_tree,
        "features": ALL_FEATURES,
        "seed": seed,
    }
    save_model_artifact(tree_pipe, champion_meta, os.path.join(artifacts_dir, "champion_v1"))
    # Also save as the active default model
    save_model_artifact(tree_pipe, champion_meta, os.path.join(artifacts_dir, "active"))
    results["champion_v1"] = champion_meta

    if has_mlflow:
        with mlflow.start_run(run_name="champion_tree_v1"):
            mlflow.log_params({
                "model_type": "HistGradientBoostingRegressor",
                "max_iter": 150,
                "learning_rate": 0.08,
                "seed": seed,
            })
            mlflow.log_metrics({
                "val_mae": val_eval_tree["overall_mae"],
                "test_mae": test_eval_tree["overall_mae"],
                "test_r2": test_eval_tree["overall_r2"],
            })
            for k, v in test_eval_tree["segment_mae_traffic"].items():
                mlflow.log_metric(f"test_mae_traffic_{k}", v)
            mlflow.set_tag("alias", "champion")

    # 3. Degraded Candidate Model (v2)
    print("\n[3/3] Training Degraded Candidate Model (v2) for failure/rollback simulation...")
    degraded_pipe = create_degraded_pipeline()
    # Train on small biased slice (only clear weather & low traffic)
    biased_mask = (train_df["weather"] == "clear") & (train_df["traffic_level"] == "low")
    X_train_biased = train_df.loc[biased_mask, ALL_FEATURES].head(250)
    y_train_biased = train_df.loc[biased_mask, TARGET_COLUMN].head(250).values
    degraded_pipe.fit(X_train_biased, y_train_biased)

    val_eval_degraded = evaluate_pipeline(degraded_pipe, val_df)
    test_eval_degraded = evaluate_pipeline(degraded_pipe, test_df)

    degraded_meta = {
        "model_version": "v2_degraded",
        "model_type": "HistGradientBoostingRegressor_Underfit",
        "status": "candidate_degraded",
        "degradation_reason": "Trained on underfit 250 samples biased exclusively toward clear weather & low traffic",
        "val_metrics": val_eval_degraded,
        "test_metrics": test_eval_degraded,
        "features": ALL_FEATURES,
        "seed": seed,
    }
    save_model_artifact(degraded_pipe, degraded_meta, os.path.join(artifacts_dir, "candidate_degraded_v2"))
    results["candidate_degraded_v2"] = degraded_meta

    if has_mlflow:
        with mlflow.start_run(run_name="candidate_degraded_v2"):
            mlflow.log_params({"model_type": "HistGradientBoosting_Degraded", "samples": len(X_train_biased)})
            mlflow.log_metrics({
                "val_mae": val_eval_degraded["overall_mae"],
                "test_mae": test_eval_degraded["overall_mae"],
                "test_r2": test_eval_degraded["overall_r2"],
            })
            for k, v in test_eval_degraded["segment_mae_traffic"].items():
                mlflow.log_metric(f"test_mae_traffic_{k}", v)
            mlflow.set_tag("alias", "candidate_degraded")

    # Summary table
    print("\n=== Model Evaluation Comparison (Test Set) ===")
    print(f"{'Model':<24} | {'Test MAE':<10} | {'Test RMSE':<10} | {'Test R²':<10} | Traffic Low/Med/High/Jam MAE")
    print("-" * 88)
    for name, meta in results.items():
        t_met = meta["test_metrics"]
        seg = t_met["segment_mae_traffic"]
        seg_str = f"{seg.get('low', '-'):.2f} / {seg.get('medium', '-'):.2f} / {seg.get('high', '-'):.2f} / {seg.get('jam', '-'):.2f}"
        print(f"{name:<24} | {t_met['overall_mae']:<10.2f} | {t_met['overall_rmse']:<10.2f} | {t_met['overall_r2']:<10.2f} | {seg_str}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaultLine Model Training")
    parser.add_argument("--data-dir", default="data", help="Directory for datasets")
    parser.add_argument("--artifacts-dir", default="artifacts/models", help="Directory for model artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_and_evaluate_models(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        seed=args.seed,
    )
