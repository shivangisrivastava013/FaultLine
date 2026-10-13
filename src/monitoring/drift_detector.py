"""
Data Quality and Statistical Distribution Drift Detector for FaultLine.

Detects:
- Data Quality violations: missing values, out-of-range values, unknown categories
- Covariate shift: 2-sample Kolmogorov-Smirnov test for continuous features
- Categorical shift: Population Stability Index (PSI) for discrete categories
- Concept / Prediction drift: KS test on predicted delivery minutes
"""

from __future__ import annotations

import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from src.models.pipeline import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
)


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 5) -> float:
    """
    Computes Population Stability Index (PSI) between expected and actual distributions.
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.25: Moderate shift
    PSI >= 0.25: Significant shift
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Categorical PSI
    unique_vals = np.unique(np.concatenate([expected, actual]))
    exp_counts = pd.Series(expected).value_counts(normalize=True).to_dict()
    act_counts = pd.Series(actual).value_counts(normalize=True).to_dict()

    psi_total = 0.0
    for val in unique_vals:
        exp_pct = exp_counts.get(val, 0.0001)
        act_pct = act_counts.get(val, 0.0001)
        # Avoid zero division
        exp_pct = max(exp_pct, 0.0001)
        act_pct = max(act_pct, 0.0001)
        psi_total += (act_pct - exp_pct) * np.log(act_pct / exp_pct)

    return float(round(psi_total, 4))


class DriftDetector:
    """Evaluates incoming batches against reference baseline."""

    def __init__(
        self,
        reference_data_path: str = "data/reference_baseline.parquet",
        ks_pvalue_threshold: float = 0.01,
        psi_threshold: float = 0.25,
    ):
        self.reference_path = reference_data_path
        self.ks_threshold = ks_pvalue_threshold
        self.psi_threshold = psi_threshold

        if os.path.exists(reference_data_path):
            self.ref_df = pd.read_parquet(reference_data_path)
        else:
            self.ref_df = pd.DataFrame()

        # Known valid categorical values from reference
        self.valid_categories = {
            col: set(self.ref_df[col].dropna().unique()) if col in self.ref_df else set()
            for col in CATEGORICAL_FEATURES
        }

    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Checks missing values, unknown categories, and out-of-range bounds."""
        issues: Dict[str, Any] = {
            "missing_values": {},
            "unknown_categories": {},
            "out_of_range": {},
            "total_rows": len(df),
            "violating_rows": 0,
        }

        if df.empty:
            return issues

        # 1. Missing values
        for col in ALL_FEATURES:
            if col in df.columns:
                n_miss = int(df[col].isna().sum())
                if n_miss > 0:
                    issues["missing_values"][col] = {
                        "count": n_miss,
                        "rate": round(n_miss / len(df), 4),
                    }
            else:
                issues["missing_values"][col] = {
                    "count": len(df),
                    "rate": 1.0,
                    "error": "Column completely missing",
                }

        # 2. Unknown categories
        for col in CATEGORICAL_FEATURES:
            if col in df.columns and col in self.valid_categories:
                valid_set = self.valid_categories[col]
                present_vals = set(df[col].dropna().astype(str).unique())
                unknowns = list(present_vals - valid_set)
                if unknowns:
                    issues["unknown_categories"][col] = unknowns

        # 3. Numerical range checks
        if "distance_km" in df.columns:
            invalid_dist = int((df["distance_km"] <= 0).sum() + (df["distance_km"] > 100.0).sum())
            if invalid_dist > 0:
                issues["out_of_range"]["distance_km"] = invalid_dist

        if "restaurant_load" in df.columns:
            invalid_load = int((df["restaurant_load"] < 0.0).sum() + (df["restaurant_load"] > 1.0).sum())
            if invalid_load > 0:
                issues["out_of_range"]["restaurant_load"] = invalid_load

        if "hour_of_day" in df.columns:
            invalid_hour = int((df["hour_of_day"] < 0).sum() + (df["hour_of_day"] > 23).sum())
            if invalid_hour > 0:
                issues["out_of_range"]["hour_of_day"] = invalid_hour

        has_issues = bool(
            issues["missing_values"] or issues["unknown_categories"] or issues["out_of_range"]
        )
        issues["has_quality_issues"] = has_issues
        return issues

    def detect_distribution_drift(self, df_current: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs statistical tests against reference baseline:
        - 2-sample KS test for continuous features
        - PSI for categorical features
        """
        if self.ref_df.empty or df_current.empty or len(df_current) < 5:
            return {"drift_detected": False, "features": {}, "drifted_features": []}

        drift_results: Dict[str, Any] = {}
        drifted_features: List[str] = []

        # 1. Continuous features via KS test
        for col in NUMERICAL_FEATURES:
            if col in df_current.columns and col in self.ref_df.columns:
                ref_vals = self.ref_df[col].dropna().values
                curr_vals = df_current[col].dropna().values

                if len(curr_vals) >= 5:
                    ks_stat, p_value = stats.ks_2samp(ref_vals, curr_vals)
                    is_drift = bool(p_value < self.ks_threshold)
                    drift_results[col] = {
                        "test": "ks_2samp",
                        "statistic": round(float(ks_stat), 4),
                        "p_value": float(p_value),
                        "threshold": self.ks_threshold,
                        "drift_detected": is_drift,
                        "ref_mean": round(float(np.mean(ref_vals)), 2),
                        "curr_mean": round(float(np.mean(curr_vals)), 2),
                    }
                    if is_drift:
                        drifted_features.append(col)

        # 2. Categorical features via PSI
        for col in CATEGORICAL_FEATURES:
            if col in df_current.columns and col in self.ref_df.columns:
                ref_vals = self.ref_df[col].dropna().astype(str).values
                curr_vals = df_current[col].dropna().astype(str).values

                if len(curr_vals) >= 5:
                    psi_val = calculate_psi(ref_vals, curr_vals)
                    is_drift = bool(psi_val >= self.psi_threshold)
                    drift_results[col] = {
                        "test": "psi",
                        "psi_score": psi_val,
                        "threshold": self.psi_threshold,
                        "drift_detected": is_drift,
                    }
                    if is_drift:
                        drifted_features.append(col)

        return {
            "drift_detected": len(drifted_features) > 0,
            "drifted_features": drifted_features,
            "features": drift_results,
            "sample_size": len(df_current),
        }

    def detect_prediction_drift(
        self,
        current_predictions: np.ndarray,
        reference_predictions: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Runs KS test on predicted delivery duration distribution."""
        if len(current_predictions) < 5:
            return {"drift_detected": False, "sample_size": len(current_predictions)}

        if reference_predictions is None:
            # Fallback to reference dataset ground truth as proxy baseline
            reference_predictions = self.ref_df["delivery_duration_minutes"].dropna().values

        ks_stat, p_value = stats.ks_2samp(reference_predictions, current_predictions)
        is_drift = bool(p_value < self.ks_threshold)

        return {
            "test": "ks_2samp",
            "statistic": round(float(ks_stat), 4),
            "p_value": float(p_value),
            "threshold": self.ks_threshold,
            "drift_detected": is_drift,
            "ref_mean": round(float(np.mean(reference_predictions)), 2),
            "curr_mean": round(float(np.mean(current_predictions)), 2),
            "sample_size": len(current_predictions),
        }
