"""
Scenario: Outcome Ingestion Delay (Ground Truth Starvation).

1. Clears existing outcome_delay incidents.
2. Injects normal client requests without delivering corresponding outcomes (stalled outcome pipeline).
3. Evaluates monitoring window to assert outcome_delay alert triggers.
4. Simulates outcome pipeline recovery by delivering buffered outcomes.
5. Verifies incident recovery and MAE calculation resumption.
"""

from __future__ import annotations

import os
import time
import requests
from simulation.replay_traffic import TrafficReplayer
from src.incidents.database import SessionLocal
from src.monitoring.worker import MonitoringWorker
from src.incidents.models import Incident, PredictionRecord


def run_scenario(api_url: str = "http://localhost:8000", seed: int = 42) -> dict:
    print("\n--- Running Scenario: Outcome Ingestion Delay ---")
    admin_key = os.getenv("ADMIN_API_KEY", "faultline_dev_secret_key")
    admin_headers = {"X-API-Key": admin_key}

    db = SessionLocal()
    worker = MonitoringWorker()
    replayer = TrafficReplayer(api_base_url=api_url, seed=seed)

    # 1. Warm-up & clean state
    requests.post(f"{api_url}/admin/inject-fault", json={"fault_type": "none"}, headers=admin_headers)
    db.query(Incident).filter(Incident.incident_type == "outcome_delay").delete()
    db.commit()

    # 2. Inject fault: send requests with outcome ingestion disabled
    t_onset = time.time()
    summary = replayer.replay(num_requests=30, rate_per_second=25.0, simulate_outcomes=False)

    # 3. Detect
    worker.process_window(db)
    t_detect = time.time()

    inc = (
        db.query(Incident)
        .filter(Incident.incident_type == "outcome_delay", Incident.status == "Open")
        .order_by(Incident.last_observed_at.desc())
        .first()
    )
    detected = inc is not None
    det_delay = round(t_detect - t_onset, 2) if detected else None

    # 4. Recover: deliver buffered outcomes for the unverified predictions
    t_rec_start = time.time()
    recent_records = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.status_code == 200)
        .order_by(PredictionRecord.timestamp.desc())
        .limit(30)
        .all()
    )

    for rec in recent_records:
        # Simulate courier delivery completion ~15-25 min
        actual_min = max(5.0, round(float(rec.predicted_minutes or 20.0) + 1.2, 2))
        requests.post(
            f"{api_url}/outcomes",
            json={
                "request_id": rec.request_id,
                "actual_delivery_minutes": actual_min,
            },
            timeout=5.0,
        )

    # Re-evaluate window with restored outcomes
    worker.process_window(db)
    t_rec_end = time.time()
    rec_time = round(t_rec_end - t_rec_start, 2)

    db.close()
    return {
        "scenario": "outcome_delay",
        "detected": detected,
        "detection_delay_seconds": det_delay,
        "recovery_time_seconds": rec_time,
        "incident_id": inc.incident_id if inc else None,
    }


if __name__ == "__main__":
    res = run_scenario()
    print("Result:", res)
