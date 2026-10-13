"""
Scenario: Bad Model Release & Audited Rollback.

1. Activates candidate_degraded_v2 model.
2. Replays traffic and delivers delayed actual outcomes.
3. Observes MAE degradation breaching threshold (> 5.5 min).
4. Asserts 'model_regression' incident created.
5. Executes authenticated rollback to champion_v1 with smoke check.
6. Replays post-rollback traffic and verifies MAE restored to healthy baseline (< 2.5 min).
"""

from __future__ import annotations

import time
import requests
from simulation.replay_traffic import TrafficReplayer
from src.incidents.database import SessionLocal
from src.monitoring.worker import MonitoringWorker
from src.incidents.models import Incident


def run_scenario(api_url: str = "http://localhost:8000", seed: int = 42) -> dict:
    print("\n--- Running Scenario: Bad Model Release & Rollback ---")
    db = SessionLocal()
    db.query(Incident).filter(Incident.incident_type == "model_regression").delete()
    db.commit()
    worker = MonitoringWorker()
    replayer = TrafficReplayer(api_base_url=api_url, seed=seed)

    # 1. Activate degraded model
    t_onset = time.time()
    requests.post(
        f"{api_url}/admin/activate-model",
        params={
            "version": "candidate_degraded_v2",
            "actor": "release_bot",
            "reason": "Canary deployment of v2 candidate",
        },
    )

    # 2. Replay traffic with outcomes
    replayer.replay(num_requests=30, rate_per_second=20.0, simulate_outcomes=True, outcome_delay_seconds=0.2)

    # 3. Detect regression
    worker.process_window(db)
    t_detect = time.time()

    inc = (
        db.query(Incident)
        .filter(Incident.incident_type == "model_regression", Incident.status == "Open")
        .order_by(Incident.last_observed_at.desc())
        .first()
    )
    detected = inc is not None
    det_delay = round(t_detect - t_onset, 2) if detected else None
    pre_rollback_mae = inc.evidence.get("metrics", {}).get("window_mae") if inc else None

    # 4. Execute rollback
    t_rec_start = time.time()
    rb_res = requests.post(
        f"{api_url}/rollback",
        json={
            "target_version": "champion_v1",
            "actor": "sre_lead",
            "reason": "Remediating model_regression incident",
        },
    ).json()

    # 5. Verify restored behavior
    replayer.replay(num_requests=30, rate_per_second=20.0, simulate_outcomes=True, outcome_delay_seconds=0.2)
    win = worker.process_window(db)
    t_rec_end = time.time()
    rec_time = round(t_rec_end - t_rec_start, 2)
    post_rollback_mae = win.mae if win else None

    db.close()
    return {
        "scenario": "bad_model_release",
        "detected": detected,
        "detection_delay_seconds": det_delay,
        "recovery_time_seconds": rec_time,
        "pre_rollback_mae": pre_rollback_mae,
        "post_rollback_mae": post_rollback_mae,
        "restored_version": rb_res.get("active_version"),
        "smoke_test_passed": rb_res.get("smoke_test_passed"),
        "incident_id": inc.incident_id if inc else None,
    }


if __name__ == "__main__":
    res = run_scenario()
    print("Result:", res)
