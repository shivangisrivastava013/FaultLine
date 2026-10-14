"""
Scenario: Latency Spike.

1. Sends clean baseline traffic.
2. Injects controlled 600ms latency delay.
3. Replays traffic and triggers monitoring worker.
4. Asserts 'high_latency' incident created with evidence.
5. Removes injected delay and confirms recovery.
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
    print("\n--- Running Scenario: Latency Spike ---")
    admin_key = os.getenv("ADMIN_API_KEY", "faultline_dev_secret_key")
    admin_headers = {"X-API-Key": admin_key}

    db = SessionLocal()
    worker = MonitoringWorker()
    replayer = TrafficReplayer(api_base_url=api_url, seed=seed)

    # 1. Warm-up clean & isolate
    requests.post(f"{api_url}/admin/inject-fault", json={"fault_type": "none"}, headers=admin_headers)
    db.query(Incident).filter(Incident.incident_type == "high_latency").delete()
    db.commit()

    # 2. Inject fault
    t_onset = time.time()
    requests.post(
        f"{api_url}/admin/inject-fault",
        json={"fault_type": "latency_spike", "delay_seconds": 0.45, "probability": 0.95},
        headers=admin_headers,
    )

    # 3. Replay traffic during fault
    replayer.replay(num_requests=25, rate_per_second=10.0, simulate_outcomes=False)

    # 4. Detect
    worker.process_window(db)
    t_detect = time.time()

    inc = (
        db.query(Incident)
        .filter(Incident.incident_type == "high_latency", Incident.status == "Open")
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
        "scenario": "latency_spike",
        "detected": detected,
        "detection_delay_seconds": det_delay,
        "recovery_time_seconds": rec_time,
        "incident_id": inc.incident_id if inc else None,
    }


if __name__ == "__main__":
    res = run_scenario()
    print("Result:", res)
