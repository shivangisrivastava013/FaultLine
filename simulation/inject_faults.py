"""
Fault Injection Controller for FaultLine.

Allows programmatic control of chaos and fault injection across all 6 scenarios:
1. invalid_input_spike
2. latency_spike
3. server_errors
4. data_drift
5. bad_model_release
6. outcome_delay
"""

from __future__ import annotations

import argparse
from typing import Dict, Any
import requests


class FaultInjector:
    """Controls fault injection flags against the running prediction service."""

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url.rstrip("/")

    def inject_latency(self, delay_seconds: float = 0.6, probability: float = 0.8) -> Dict[str, Any]:
        """Injects artificial latency delay into predictions."""
        print(f"Injecting Latency Spike: delay={delay_seconds}s, prob={probability}")
        resp = requests.post(
            f"{self.api_base_url}/admin/inject-fault",
            json={
                "fault_type": "latency_spike",
                "delay_seconds": delay_seconds,
                "probability": probability,
            },
            timeout=5.0,
        )
        return resp.json()

    def inject_server_errors(self, probability: float = 0.35, status_code: int = 500) -> Dict[str, Any]:
        """Injects simulated 5xx internal server errors."""
        print(f"Injecting Server Errors: status={status_code}, prob={probability}")
        resp = requests.post(
            f"{self.api_base_url}/admin/inject-fault",
            json={
                "fault_type": "server_error",
                "probability": probability,
                "error_status_code": status_code,
                "error_message": "Chaos injection: DB pool exhaustion error",
            },
            timeout=5.0,
        )
        return resp.json()

    def activate_degraded_model(self) -> Dict[str, Any]:
        """Activates deliberately underfit candidate model (v2_degraded)."""
        print("Activating Degraded Candidate Model (v2_degraded)...")
        resp = requests.post(
            f"{self.api_base_url}/admin/activate-model",
            params={
                "version": "candidate_degraded_v2",
                "actor": "release_bot",
                "reason": "Canary deployment of v2 candidate",
            },
            timeout=5.0,
        )
        return resp.json()

    def rollback_to_champion(self) -> Dict[str, Any]:
        """Performs audited rollback to champion_v1."""
        print("Executing Rollback to champion_v1...")
        resp = requests.post(
            f"{self.api_base_url}/rollback",
            json={
                "target_version": "champion_v1",
                "actor": "sre_lead",
                "reason": "Automated incident recovery rollback",
            },
            timeout=5.0,
        )
        return resp.json()

    def reset_all_faults(self) -> Dict[str, Any]:
        """Restores service to healthy clean state."""
        print("Resetting all fault injection parameters to normal clean state...")
        # Reset runtime chaos
        r1 = requests.post(
            f"{self.api_base_url}/admin/inject-fault",
            json={"fault_type": "none"},
            timeout=5.0,
        )
        # Restore champion model
        try:
            self.rollback_to_champion()
        except Exception:
            pass
        return r1.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaultLine Fault Injector CLI")
    parser.add_argument("--url", default="http://localhost:8000", help="Prediction API URL")
    parser.add_argument(
        "--fault",
        required=True,
        choices=["latency", "server_error", "bad_release", "rollback", "reset"],
        help="Fault scenario to activate",
    )
    parser.add_argument("--delay", type=float, default=0.6, help="Latency delay in seconds")
    parser.add_argument("--prob", type=float, default=0.8, help="Fault probability")
    args = parser.parse_args()

    injector = FaultInjector(api_base_url=args.url)
    if args.fault == "latency":
        injector.inject_latency(delay_seconds=args.delay, probability=args.prob)
    elif args.fault == "server_error":
        injector.inject_server_errors(probability=args.prob)
    elif args.fault == "bad_release":
        injector.activate_degraded_model()
    elif args.fault == "rollback":
        injector.rollback_to_champion()
    elif args.fault == "reset":
        injector.reset_all_faults()
