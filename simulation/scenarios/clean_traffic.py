"""
Scenario: Clean Traffic (Control baseline).

1. Replays normal healthy traffic.
2. Runs monitoring worker.
3. Verifies zero false positive alerts are created.
"""

from __future__ import annotations

import requests
from simulation.replay_traffic import TrafficReplayer
from src.incidents.database import SessionLocal
from src.monitoring.worker import MonitoringWorker
from src.incidents.models import Incident


def run_scenario(api_url: str = "http://localhost:8000", seed: int = 101) -> dict:
    print("\n--- Running Scenario: Clean Traffic (Control) ---")
    db = SessionLocal()
    worker = MonitoringWorker()
    replayer = TrafficReplayer(api_base_url=api_url, seed=seed)

    # Ensure clean state
    requests.post(f"{api_url}/admin/inject-fault", json={"fault_type": "none"})
    requests.post(
        f"{api_url}/rollback",
        json={"target_version": "champion_v1", "actor": "system", "reason": "clean baseline run"},
    )

    # Count open incidents before run
    initial_open_count = db.query(Incident).filter(Incident.status == "Open").count()

    # Replay 50 normal requests
    summary = replayer.replay(num_requests=50, rate_per_second=20.0, simulate_outcomes=True, outcome_delay_seconds=0.2)
    win = worker.process_window(db)

    final_open_count = db.query(Incident).filter(Incident.status == "Open").count()
    new_incidents_count = max(0, final_open_count - initial_open_count)

    win_mae = win.mae if win else None
    win_p95 = win.p95_latency_ms if win else None
    db.close()
    return {
        "scenario": "clean_traffic",
        "requests_sent": summary["requests_sent"],
        "false_alert_count": new_incidents_count,
        "is_clean": (new_incidents_count == 0),
        "observed_mae": win_mae,
        "p95_latency_ms": win_p95,
    }


if __name__ == "__main__":
    res = run_scenario()
    print("Result:", res)
