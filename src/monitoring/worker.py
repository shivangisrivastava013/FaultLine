"""
FaultLine Background Monitoring Worker.

Polls recent prediction windows, evaluates drift and accuracy,
computes operational metrics, and creates/updates incidents via RuleEngine.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
import uuid
from typing import Dict, Any, Optional

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.incidents.database import SessionLocal, init_db
from src.incidents.models import (
    PredictionRecord,
    OutcomeRecord,
    MonitoringWindow,
    utcnow,
)
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.rules import RuleEngine
from src.monitoring.metrics import PREDICTION_ERROR_MAE_GAUGE


class MonitoringWorker:
    """Worker evaluating windowed operational and ML metrics."""

    def __init__(
        self,
        window_size_seconds: int = 30,
        reference_data_path: str = "data/reference_baseline.parquet",
    ):
        self.window_size_seconds = window_size_seconds
        self.drift_detector = DriftDetector(reference_data_path=reference_data_path)
        self.rule_engine = RuleEngine()

    def process_window(
        self,
        db: Session,
        since: Optional[datetime.datetime] = None,
    ) -> Optional[MonitoringWindow]:
        """Processes the most recent time window of prediction requests and outcomes."""
        db.expire_all()
        now = utcnow()
        if since is not None:
            window_start = since
        else:
            window_start = now - datetime.timedelta(seconds=self.window_size_seconds)

        # 1. Fetch prediction records
        records = (
            db.query(PredictionRecord)
            .filter(PredictionRecord.timestamp >= window_start)
            .order_by(PredictionRecord.timestamp.asc())
            .all()
        )

        if not records:
            # Fallback to last 100 records if no recent records within window_size_seconds
            records = (
                db.query(PredictionRecord)
                .order_by(PredictionRecord.timestamp.desc())
                .limit(100)
                .all()
            )
            records = list(reversed(records))

        if not records:
            return None

        # Build DataFrame
        rows = [
            {
                "request_id": r.request_id,
                "timestamp": r.timestamp,
                "model_version": r.model_version,
                "distance_km": r.distance_km,
                "traffic_level": r.traffic_level,
                "weather": r.weather,
                "hour_of_day": r.hour_of_day,
                "restaurant_load": r.restaurant_load,
                "predicted_minutes": r.predicted_minutes,
                "latency_ms": r.latency_ms,
                "status_code": r.status_code,
                "error_message": r.error_message,
            }
            for r in records
        ]
        df_reqs = pd.DataFrame(rows)

        total_reqs = len(df_reqs)
        successes = int((df_reqs["status_code"] == 200).sum())
        val_errors = int((df_reqs["status_code"] == 422).sum())
        server_errors = int((df_reqs["status_code"] >= 500).sum())

        error_rate = round((server_errors / total_reqs) * 100.0, 2)
        val_fail_rate = round((val_errors / total_reqs) * 100.0, 2)

        latencies = df_reqs["latency_ms"].dropna().values
        p50_lat = round(float(np.percentile(latencies, 50)), 2) if len(latencies) else 0.0
        p95_lat = round(float(np.percentile(latencies, 95)), 2) if len(latencies) else 0.0

        # Model version in this window (dominant version)
        active_ver = df_reqs["model_version"].mode()[0] if not df_reqs.empty else "v1"

        # 2. Join outcomes
        req_ids = df_reqs["request_id"].tolist()
        outcomes = (
            db.query(OutcomeRecord)
            .filter(OutcomeRecord.request_id.in_(req_ids))
            .all()
        )
        outcomes_map = {o.request_id: o.actual_delivery_minutes for o in outcomes}

        # Calculate Window MAE
        df_reqs["actual_minutes"] = df_reqs["request_id"].map(outcomes_map)
        completed_mask = df_reqs["actual_minutes"].notna() & df_reqs["predicted_minutes"].notna()
        completed_count = int(completed_mask.sum())

        window_mae = None
        segment_mae: Dict[str, float] = {}

        if completed_count > 0:
            y_true = df_reqs.loc[completed_mask, "actual_minutes"].values
            y_pred = df_reqs.loc[completed_mask, "predicted_minutes"].values
            window_mae = round(float(np.mean(np.abs(y_true - y_pred))), 2)
            PREDICTION_ERROR_MAE_GAUGE.labels(model_version=active_ver).set(window_mae)

            # Segment MAE
            for t_level in ["low", "medium", "high", "jam"]:
                seg_mask = completed_mask & (df_reqs["traffic_level"] == t_level)
                if seg_mask.sum() > 0:
                    yt = df_reqs.loc[seg_mask, "actual_minutes"].values
                    yp = df_reqs.loc[seg_mask, "predicted_minutes"].values
                    segment_mae[t_level] = round(float(np.mean(np.abs(yt - yp))), 2)

        # 3. Statistical Data Drift check
        valid_features_df = df_reqs.dropna(subset=["distance_km", "restaurant_load"])
        drift_report = None
        if len(valid_features_df) >= 5:
            drift_report = self.drift_detector.detect_distribution_drift(valid_features_df)

        # 4. Package window summary
        window_summary = {
            "total_requests": total_reqs,
            "success_count": successes,
            "server_errors_count": server_errors,
            "validation_errors_count": val_errors,
            "error_rate_percent": error_rate,
            "val_failure_rate_percent": val_fail_rate,
            "p50_latency_ms": p50_lat,
            "p95_latency_ms": p95_lat,
            "completed_outcomes": completed_count,
            "window_mae": window_mae,
            "segment_mae": segment_mae,
            "model_version": active_ver,
        }

        # 5. Persist MonitoringWindow
        w_id = f"win_{uuid.uuid4().hex[:12]}"
        mon_window = MonitoringWindow(
            window_id=w_id,
            start_time=df_reqs["timestamp"].min(),
            end_time=df_reqs["timestamp"].max(),
            sample_count=total_reqs,
            error_rate=error_rate,
            validation_failure_rate=val_fail_rate,
            p95_latency_ms=p95_lat,
            mae=window_mae,
            status="evaluated",
            metrics_summary=window_summary,
        )
        db.add(mon_window)
        db.commit()

        # 6. Evaluate Rules and Trigger Incidents
        incidents = self.rule_engine.evaluate_window(
            db=db,
            window_summary=window_summary,
            df_requests=df_reqs,
            drift_report=drift_report,
            model_version=active_ver,
        )

        if incidents:
            print(f"[Worker] Window {w_id}: {len(incidents)} active incident(s) triggered/updated:")
            for inc in incidents:
                print(f"  -> [{inc.status}] {inc.incident_id} ({inc.incident_type}, severity={inc.severity})")
        else:
            print(f"[Worker] Window {w_id}: Healthy (reqs={total_reqs}, p95={p95_lat}ms, err={error_rate}%, val_fail={val_fail_rate}%, mae={window_mae})")

        return mon_window

    def run_loop(self, interval_seconds: int = 5) -> None:
        """Runs the monitoring loop indefinitely."""
        print(f"=== Starting FaultLine Monitoring Worker (interval={interval_seconds}s) ===")
        init_db()
        while True:
            db = SessionLocal()
            try:
                self.process_window(db)
            except Exception as e:
                print(f"[Worker Error] {e}")
            finally:
                db.close()
            time.sleep(interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaultLine Monitoring Worker")
    parser.add_argument("--once", action="store_true", help="Run single window evaluation and exit")
    parser.add_argument("--interval", type=int, default=5, help="Loop interval seconds")
    parser.add_argument("--window", type=int, default=120, help="Window size seconds")
    args = parser.parse_args()

    worker = MonitoringWorker(window_size_seconds=args.window)
    if args.once:
        db = SessionLocal()
        try:
            worker.process_window(db)
        finally:
            db.close()
    else:
        worker.run_loop(interval_seconds=args.interval)
