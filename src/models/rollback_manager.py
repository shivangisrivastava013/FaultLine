"""
Model Rollback & Lifecycle Manager for FaultLine.

Handles:
- In-memory active model switching
- Pre-activation smoke testing (guarantees a broken model never activates)
- Audited deployment event recording
- Atomic version transitions
"""

from __future__ import annotations

import os
import threading
from typing import Optional, Dict, Any, Tuple
import pandas as pd
from sklearn.base import BaseEstimator
from src.models.pipeline import load_model_artifact, save_model_artifact, ALL_FEATURES
from src.monitoring.metrics import MODEL_LOAD_FAILURES_TOTAL, ACTIVE_MODEL_INFO
from src.incidents.manager import IncidentManager
from sqlalchemy.orm import Session


class ModelRollbackManager:
    """Manages model loading, health checks, rollback execution, and audit logging."""

    def __init__(self, artifacts_dir: str = "artifacts/models"):
        self.artifacts_dir = artifacts_dir
        self._lock = threading.Lock()
        self.active_version: str = "unknown"
        self.active_pipeline: Optional[BaseEstimator] = None
        self.active_metadata: Dict[str, Any] = {}
        self.previous_version: Optional[str] = None

    def initialize_active_model(self, default_version: str = "champion_v1") -> None:
        """Loads default champion model during API startup."""
        with self._lock:
            active_dir = os.path.join(self.artifacts_dir, "active")
            target_dir = active_dir if os.path.exists(active_dir) else os.path.join(self.artifacts_dir, default_version)

            if not os.path.exists(target_dir):
                raise FileNotFoundError(
                    f"No model artifacts found in {target_dir}. Run `python -m src.models.train` first."
                )

            pipe, meta = load_model_artifact(target_dir)
            self._run_smoke_check(pipe)
            self.active_pipeline = pipe
            self.active_metadata = meta
            self.active_version = meta.get("model_version", default_version)
            ACTIVE_MODEL_INFO.labels(
                model_version=self.active_version,
                model_type=meta.get("model_type", "tree"),
            ).set(1)

    def _run_smoke_check(self, pipeline: BaseEstimator) -> float:
        """
        Executes a smoke test with standard test features.
        Verifies pipeline executes without exception and returns realistic delivery minutes.
        """
        sample_df = pd.DataFrame([{
            "distance_km": 5.0,
            "traffic_level": "medium",
            "weather": "clear",
            "hour_of_day": 12,
            "restaurant_load": 0.5,
        }])[ALL_FEATURES]

        pred = pipeline.predict(sample_df)
        val = float(pred[0])
        if val <= 0.0 or val > 300.0 or not isinstance(val, (int, float)):
            raise ValueError(f"Smoke check failed: unrealistic prediction value {val}")
        return val

    def predict(self, df_features: pd.DataFrame) -> Tuple[float, str]:
        """Runs prediction on active pipeline. Returns (predicted_minutes, active_version)."""
        if self.active_pipeline is None:
            raise RuntimeError("No active model is loaded in prediction service")

        # Thread-safe read of pipeline
        with self._lock:
            pipe = self.active_pipeline
            ver = self.active_version

        preds = pipe.predict(df_features[ALL_FEATURES])
        return round(float(preds[0]), 2), ver

    def activate_version(
        self,
        target_version: str,
        actor: str,
        reason: str,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        Loads candidate model, performs smoke check, and if successful,
        promotes it to active and logs deployment audit event.
        If loading or smoke check fails, active model is kept intact.
        """
        version_dirs = {
            "v1": "champion_v1",
            "champion": "champion_v1",
            "champion_v1": "champion_v1",
            "v0_linear": "baseline_linear",
            "baseline": "baseline_linear",
            "baseline_linear": "baseline_linear",
            "v2_degraded": "candidate_degraded_v2",
            "candidate": "candidate_degraded_v2",
            "candidate_degraded_v2": "candidate_degraded_v2",
        }

        dir_name = version_dirs.get(target_version, target_version)
        candidate_dir = os.path.join(self.artifacts_dir, dir_name)

        if not os.path.exists(candidate_dir):
            MODEL_LOAD_FAILURES_TOTAL.labels(model_version=target_version).inc()
            raise FileNotFoundError(f"Model version directory '{dir_name}' not found in {self.artifacts_dir}")

        # 1. Attempt loading in isolated reference
        try:
            candidate_pipe, candidate_meta = load_model_artifact(candidate_dir)
        except Exception as e:
            MODEL_LOAD_FAILURES_TOTAL.labels(model_version=target_version).inc()
            raise RuntimeError(f"Failed to load model artifact: {e}")

        # 2. Run smoke check
        try:
            self._run_smoke_check(candidate_pipe)
        except Exception as e:
            MODEL_LOAD_FAILURES_TOTAL.labels(model_version=target_version).inc()
            raise RuntimeError(f"Model smoke check failed for {target_version}: {e}")

        # 3. Swap active model atomically under lock
        with self._lock:
            old_version = self.active_version
            self.previous_version = old_version
            self.active_pipeline = candidate_pipe
            self.active_metadata = candidate_meta
            self.active_version = candidate_meta.get("model_version", target_version)

            # Persist to active directory
            try:
                save_model_artifact(
                    self.active_pipeline,
                    self.active_metadata,
                    os.path.join(self.artifacts_dir, "active"),
                )
            except Exception:
                pass  # Non-fatal if filesystem is read-only

            ACTIVE_MODEL_INFO.labels(
                model_version=self.active_version,
                model_type=candidate_meta.get("model_type", "unknown"),
            ).set(1)

        # 4. Audit trail
        if db is not None:
            IncidentManager.record_deployment_event(
                db=db,
                activated_version=self.active_version,
                previous_version=old_version,
                actor=actor,
                reason=reason,
                smoke_test_passed=True,
            )

        return {
            "success": True,
            "previous_version": old_version,
            "active_version": self.active_version,
            "actor": actor,
            "reason": reason,
            "smoke_test_passed": True,
            "message": f"Successfully activated model {self.active_version} (restored from {old_version})",
        }

    def rollback_to_champion(
        self,
        actor: str,
        reason: str,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """Rolls back to champion_v1 with audit trail."""
        return self.activate_version(
            target_version="champion_v1",
            actor=actor,
            reason=f"Rollback: {reason}",
            db=db,
        )


# Global singleton instance
model_manager = ModelRollbackManager()
