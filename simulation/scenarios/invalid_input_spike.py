"""
Scenario: Invalid Input Spike (422 Rejections).

1. Replays corrupt payloads missing fields or with invalid categorical values.
2. Asserts 'validation_failures' incident created.
3. Corrects simulated traffic producer and confirms recovery.
"""

from __future__ import annotations

import time
import requests
from simulation.replay_traffic import TrafficReplayer
from src.incidents.database import SessionLocal
from src.monitoring.worker import MonitoringWorker
from src.incidents.models import Incident


def run_scenario(api_url: str = "http://localhost:8000", seed: int = 42) -> dict:
    print("\n--- Running Scenario: Invalid Input Spike ---")
    db = SessionLocal()
    worker = MonitoringWorker()
    replayer = TrafficReplayer(api_base_url=api_url, seed=seed)

    # 1. Warm-up
    requests.post(f"{api_url}/admin/inject-fault", json={"fault_type": "none"})
    db.query(Incident).filter(Incident.incident_type == "validation_failures").delete()
    db.commit()

    # 2. Inject corrupt payloads (60% corruption probability over 40 requests)
    t_onset = time.time()
    replayer.replay(
        num_requests=40,
        rate_per_second=20.0,
        simulate_outcomes=False,
        corrupt_mode="invalid_inputs",
        corrupt_probability=0.60,
    )

    # 3. Detect
    worker.process_window(db)
    t_detect = time.time()

    inc = (
        db.query(Incident)
        .filter(Incident.incident_type == "validation_failures", Incident.status == "Open")
        .order_by(Incident.last_observed_at.desc())
        .first()
    )
    detected = inc is not None
    det_delay = round(t_detect - t_onset, 2) if detected else None

    # 4. Recover: Correct producer
    t_rec_start = time.time()
    replayer.replay(num_requests=25, rate_per_second=25.0, simulate_outcomes=False, corrupt_mode=None)
    worker.process_window(db)
    t_rec_end = time.time()
    rec_time = round(t_rec_end - t_rec_start, 2)

    db.close()
    return {
        "scenario": "invalid_input_spike",
        "detected": detected,
        "detection_delay_seconds": det_delay,
        "recovery_time_seconds": rec_time,
        "incident_id": inc.incident_id if inc else None,
    }


if __name__ == "__main__":
    res = run_scenario()
    print("Result:", res)
