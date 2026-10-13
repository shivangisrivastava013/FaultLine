"""
Scenario: Data Drift (Covariate Shift).

1. Replays shifted distributions (extreme distance long-hauls and storms).
2. Runs statistical KS and PSI tests against reference baseline.
3. Asserts 'data_drift' incident created with affected features list.
"""

from __future__ import annotations

import time
import requests
from simulation.replay_traffic import TrafficReplayer
from src.incidents.database import SessionLocal
from src.monitoring.worker import MonitoringWorker
from src.incidents.models import Incident


def run_scenario(api_url: str = "http://localhost:8000", seed: int = 42) -> dict:
    print("\n--- Running Scenario: Data Drift ---")
    db = SessionLocal()
    worker = MonitoringWorker()
    replayer = TrafficReplayer(api_base_url=api_url, seed=seed)

    # 1. Warm-up
    requests.post(f"{api_url}/admin/inject-fault", json={"fault_type": "none"})
    db.query(Incident).filter(Incident.incident_type == "data_drift").delete()
    db.commit()

    # 2. Replay drifted data
    t_onset = time.time()
    replayer.replay(
        num_requests=35,
        rate_per_second=20.0,
        simulate_outcomes=False,
        corrupt_mode="data_drift",
        corrupt_probability=0.85,
    )

    # 3. Detect
    worker.process_window(db)
    t_detect = time.time()

    inc = (
        db.query(Incident)
        .filter(Incident.incident_type == "data_drift", Incident.status == "Open")
        .order_by(Incident.last_observed_at.desc())
        .first()
    )
    detected = inc is not None
    det_delay = round(t_detect - t_onset, 2) if detected else None

    # 4. Recover: return to normal baseline traffic
    t_rec_start = time.time()
    replayer.replay(num_requests=30, rate_per_second=25.0, simulate_outcomes=False, corrupt_mode=None)
    worker.process_window(db)
    t_rec_end = time.time()
    rec_time = round(t_rec_end - t_rec_start, 2)

    db.close()
    return {
        "scenario": "data_drift",
        "detected": detected,
        "detection_delay_seconds": det_delay,
        "recovery_time_seconds": rec_time,
        "incident_id": inc.incident_id if inc else None,
        "affected_features": inc.evidence.get("affected_features") if inc else [],
    }


if __name__ == "__main__":
    res = run_scenario()
    print("Result:", res)
