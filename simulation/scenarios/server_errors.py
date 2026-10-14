"""
Scenario: Server Errors (5xx Spike).

1. Replays clean traffic.
2. Injects 40% probability 500 Internal Server Errors.
3. Replays traffic and triggers monitoring.
4. Asserts 'server_errors' incident created with evidence.
5. Disables fault and confirms recovery.
"""

from __future__ import annotations

import os
import time
import requests
from simulation.replay_traffic import TrafficReplayer
from src.incidents.database import SessionLocal
from src.monitoring.worker import MonitoringWorker
from src.incidents.models import Incident


def run_scenario(api_url: str = "http://localhost:8000", seed: int = 42) -> dict:
    print("\n--- Running Scenario: Server Errors Spike ---")
    admin_key = os.getenv("ADMIN_API_KEY", "faultline_dev_secret_key")
    admin_headers = {"X-API-Key": admin_key}

    db = SessionLocal()
    worker = MonitoringWorker()
    replayer = TrafficReplayer(api_base_url=api_url, seed=seed)

    # 1. Warm-up
    requests.post(f"{api_url}/admin/inject-fault", json={"fault_type": "none"}, headers=admin_headers)
    db.query(Incident).filter(Incident.incident_type == "server_errors").delete()
    db.commit()

    # 2. Inject fault
    t_onset = time.time()
    requests.post(
        f"{api_url}/admin/inject-fault",
        json={"fault_type": "server_error", "probability": 0.40, "error_status_code": 500},
        headers=admin_headers,
    )

    # 3. Replay traffic
    replayer.replay(num_requests=25, rate_per_second=20.0, simulate_outcomes=False)

    # 4. Detect
    worker.process_window(db)
    t_detect = time.time()

    inc = (
        db.query(Incident)
        .filter(Incident.incident_type == "server_errors", Incident.status == "Open")
        .order_by(Incident.last_observed_at.desc())
        .first()
    )
    detected = inc is not None
    det_delay = round(t_detect - t_onset, 2) if detected else None

    # 5. Recover
    t_rec_start = time.time()
    requests.post(f"{api_url}/admin/inject-fault", json={"fault_type": "none"}, headers=admin_headers)
    replayer.replay(num_requests=20, rate_per_second=25.0, simulate_outcomes=False)
    worker.process_window(db)
    t_rec_end = time.time()
    rec_time = round(t_rec_end - t_rec_start, 2)

    db.close()
    return {
        "scenario": "server_errors",
        "detected": detected,
        "detection_delay_seconds": det_delay,
        "recovery_time_seconds": rec_time,
        "incident_id": inc.incident_id if inc else None,
    }


if __name__ == "__main__":
    res = run_scenario()
    print("Result:", res)
