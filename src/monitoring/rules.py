"""
Rule Evaluation Engine for FaultLine.

Evaluates operational metrics, data drift, and model regression against configured rules,
compiling structured evidence bundles and triggering/updating incidents.
"""

from __future__ import annotations

import os
from typing import Dict, Any, List, Optional
import yaml
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from src.incidents.manager import IncidentManager
from src.incidents.models import Incident


DEFAULT_RULES_PATH = "configs/monitoring_rules.yaml"


def load_rules_config(config_path: str = DEFAULT_RULES_PATH) -> Dict[str, Any]:
    """Loads YAML rules configuration."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class RuleEngine:
    """Evaluates window metrics against configured alert thresholds."""

    def __init__(self, config_path: str = DEFAULT_RULES_PATH):
        self.config = load_rules_config(config_path)
        self.op_rules = self.config.get("operational_rules", {})
        self.data_rules = self.config.get("data_rules", {})
        self.acc_rules = self.config.get("accuracy_rules", {})

    def evaluate_window(
        self,
        db: Session,
        window_summary: Dict[str, Any],
        df_requests: pd.DataFrame,
        drift_report: Optional[Dict[str, Any]] = None,
        model_version: str = "v1",
    ) -> List[Incident]:
        """
        Runs all enabled rules over current window metrics.
        Returns list of new or updated incidents.
        """
        triggered_incidents: List[Incident] = []

        total_requests = int(window_summary.get("total_requests", 0))
        if total_requests == 0:
            return triggered_incidents

        # Extract metrics
        p95_lat = float(window_summary.get("p95_latency_ms", 0.0))
        err_rate = float(window_summary.get("error_rate_percent", 0.0))
        val_fail_rate = float(window_summary.get("val_failure_rate_percent", 0.0))
        window_mae = window_summary.get("window_mae")
        completed_outcomes = int(window_summary.get("completed_outcomes", 0))

        # Sample request representations for evidence
        sample_requests = []
        if not df_requests.empty:
            for _, r in df_requests.head(5).iterrows():
                sample_requests.append({
                    "request_id": str(r.get("request_id", "")),
                    "status_code": int(r.get("status_code", 200)),
                    "latency_ms": float(r.get("latency_ms", 0.0)),
                    "predicted_minutes": float(r.get("predicted_minutes", 0.0)) if pd.notna(r.get("predicted_minutes")) else None,
                })

        # ---------------------------------------------------------------------
        # 1. Operational Rule: High Latency
        # ---------------------------------------------------------------------
        r_lat = self.op_rules.get("high_latency", {})
        if r_lat.get("enabled", True) and total_requests >= r_lat.get("min_samples", 10):
            warn_th = float(r_lat.get("warning_threshold_ms", 150.0))
            crit_th = float(r_lat.get("critical_threshold_ms", 400.0))

            if p95_lat >= warn_th:
                sev = "critical" if p95_lat >= crit_th else "warning"
                evidence = {
                    "rule": "high_latency",
                    "rule_threshold": warn_th,
                    "metrics": {
                        "p95_latency_ms": p95_lat,
                        "p50_latency_ms": window_summary.get("p50_latency_ms", 0.0),
                        "total_requests": total_requests,
                    },
                    "sample_requests": sample_requests,
                }
                inc = IncidentManager.record_alert(
                    db=db,
                    incident_type="high_latency",
                    severity=sev,
                    model_version=model_version,
                    evidence=evidence,
                    recommended_action=r_lat.get(
                        "recommended_action",
                        "Investigate service latency bottlenecks or injected delay faults.",
                    ),
                )
                triggered_incidents.append(inc)

        # ---------------------------------------------------------------------
        # 2. Operational Rule: Server Errors (5xx)
        # ---------------------------------------------------------------------
        r_err = self.op_rules.get("server_errors", {})
        if r_err.get("enabled", True) and total_requests >= r_err.get("min_samples", 10):
            warn_th = float(r_err.get("warning_threshold_percent", 5.0))
            crit_th = float(r_err.get("critical_threshold_percent", 15.0))

            if err_rate >= warn_th:
                sev = "critical" if err_rate >= crit_th else "warning"
                evidence = {
                    "rule": "server_errors",
                    "rule_threshold": warn_th,
                    "metrics": {
                        "error_rate_percent": err_rate,
                        "server_errors_count": window_summary.get("server_errors_count", 0),
                        "total_requests": total_requests,
                    },
                    "sample_requests": sample_requests,
                }
                inc = IncidentManager.record_alert(
                    db=db,
                    incident_type="server_errors",
                    severity=sev,
                    model_version=model_version,
                    evidence=evidence,
                    recommended_action=r_err.get(
                        "recommended_action",
                        "Check exception tracebacks. Disable active server error chaos fault.",
                    ),
                )
                triggered_incidents.append(inc)

        # ---------------------------------------------------------------------
        # 3. Operational Rule: Validation Failures (422)
        # ---------------------------------------------------------------------
        r_val = self.op_rules.get("validation_failures", {})
        if r_val.get("enabled", True) and total_requests >= r_val.get("min_samples", 10):
            warn_th = float(r_val.get("warning_threshold_percent", 10.0))
            crit_th = float(r_val.get("critical_threshold_percent", 25.0))

            if val_fail_rate >= warn_th:
                sev = "critical" if val_fail_rate >= crit_th else "warning"
                evidence = {
                    "rule": "validation_failures",
                    "rule_threshold": warn_th,
                    "metrics": {
                        "validation_failure_rate_percent": val_fail_rate,
                        "validation_errors_count": window_summary.get("validation_errors_count", 0),
                        "total_requests": total_requests,
                    },
                    "sample_requests": sample_requests,
                }
                inc = IncidentManager.record_alert(
                    db=db,
                    incident_type="validation_failures",
                    severity=sev,
                    model_version=model_version,
                    evidence=evidence,
                    recommended_action=r_val.get(
                        "recommended_action",
                        "Investigate malformed client requests or upstream schema breakage.",
                    ),
                )
                triggered_incidents.append(inc)

        # ---------------------------------------------------------------------
        # 4. Data Drift Rule
        # ---------------------------------------------------------------------
        r_drift = self.data_rules.get("data_drift", {})
        if r_drift.get("enabled", True) and drift_report and drift_report.get("drift_detected"):
            drifted_feats = drift_report.get("drifted_features", [])
            evidence = {
                "rule": "data_drift",
                "affected_features": drifted_feats,
                "metrics": {
                    "drifted_feature_count": len(drifted_feats),
                    "feature_details": drift_report.get("features", {}),
                    "total_samples": drift_report.get("sample_size", total_requests),
                },
                "sample_requests": sample_requests,
            }
            inc = IncidentManager.record_alert(
                db=db,
                incident_type="data_drift",
                severity=r_drift.get("severity", "warning"),
                model_version=model_version,
                evidence=evidence,
                recommended_action=r_drift.get(
                    "recommended_action",
                    "Investigate covariate shift in input features. Compare against reference distributions.",
                ),
            )
            triggered_incidents.append(inc)

        # ---------------------------------------------------------------------
        # 5. Accuracy Rule: Model Regression
        # ---------------------------------------------------------------------
        r_acc = self.acc_rules.get("model_regression", {})
        min_outcomes = int(r_acc.get("min_outcomes", 12))
        th_mae = float(r_acc.get("threshold_mae_minutes", 5.5))

        if r_acc.get("enabled", True) and window_mae is not None and completed_outcomes >= min_outcomes:
            if float(window_mae) >= th_mae:
                evidence = {
                    "rule": "model_regression",
                    "rule_threshold": th_mae,
                    "metrics": {
                        "window_mae": float(window_mae),
                        "completed_outcomes": completed_outcomes,
                        "segment_mae": window_summary.get("segment_mae", {}),
                    },
                    "sample_requests": sample_requests,
                }
                inc = IncidentManager.record_alert(
                    db=db,
                    incident_type="model_regression",
                    severity="critical",
                    model_version=model_version,
                    evidence=evidence,
                    recommended_action=r_acc.get(
                        "recommended_action",
                        "Severe model accuracy degradation detected. Perform rollback to champion_v1.",
                    ),
                )
                triggered_incidents.append(inc)

        return triggered_incidents
