"""
Traffic Replay Simulator for FaultLine.

Replays synthetic delivery requests against the prediction service API at a configurable rate,
with fixed seeds, scenario start/end time tracking, and optional outcome ingestion.
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
import uuid
from typing import List, Dict, Any, Optional
import requests
import numpy as np
from simulation.generate_data import generate_delivery_records


class TrafficReplayer:
    """Replays traffic to a running prediction service."""

    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        seed: int = 42,
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def replay(
        self,
        num_requests: int = 100,
        rate_per_second: float = 10.0,
        simulate_outcomes: bool = True,
        outcome_delay_seconds: float = 0.5,
        corrupt_mode: Optional[str] = None,
        corrupt_probability: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Sends requests to POST /predict at given rate.
        Supports corrupt_mode:
          - 'invalid_inputs': missing fields or invalid categorical values
          - 'data_drift': severe shift in distances or weather
        """
        start_time = datetime.datetime.now(datetime.timezone.utc)
        print(f"[{start_time.isoformat()}] Starting traffic replay: {num_requests} requests @ {rate_per_second} req/s")

        records_df = generate_delivery_records(
            num_samples=num_requests,
            seed=self.seed,
            start_time=start_time,
        )

        results: List[Dict[str, Any]] = []
        pending_outcomes: List[Dict[str, Any]] = []

        delay_between_requests = 1.0 / max(rate_per_second, 0.1)

        for i, row in records_df.iterrows():
            req_id_val = f"sim_{uuid.uuid4().hex[:12]}"
            req_payload: Dict[str, Any] = {
                "distance_km": float(row["distance_km"]),
                "traffic_level": str(row["traffic_level"]),
                "weather": str(row["weather"]),
                "hour_of_day": int(row["hour_of_day"]),
                "restaurant_load": float(row["restaurant_load"]),
                "request_id": req_id_val,
            }

            # Optional corruption
            if corrupt_mode and self.rng.random() < corrupt_probability:
                if corrupt_mode == "invalid_inputs":
                    c_type = self.rng.choice(["missing_field", "unknown_category", "out_of_range"])
                    if c_type == "missing_field":
                        del req_payload["traffic_level"]
                    elif c_type == "unknown_category":
                        req_payload["traffic_level"] = "hyperspace_jam"
                    elif c_type == "out_of_range":
                        req_payload["distance_km"] = -12.5
                elif corrupt_mode == "data_drift":
                    # Severe distribution shift: extreme distances (long-haul) and storms
                    req_payload["distance_km"] = float(round(self.rng.uniform(35.0, 85.0), 2))
                    req_payload["weather"] = "storm"
                    req_payload["traffic_level"] = "jam"

            t0 = time.perf_counter()
            try:
                resp = requests.post(
                    f"{self.api_base_url}/predict",
                    json=req_payload,
                    timeout=5.0,
                )
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                res_data = {
                    "request_id": req_payload.get("request_id"),
                    "status_code": resp.status_code,
                    "latency_ms": latency_ms,
                }
                if resp.status_code == 200:
                    body = resp.json()
                    res_data["predicted_minutes"] = body.get("predicted_minutes")
                    res_data["model_version"] = body.get("model_version")

                    if simulate_outcomes:
                        pending_outcomes.append({
                            "request_id": body["request_id"],
                            "actual_delivery_minutes": float(row["delivery_duration_minutes"]),
                        })
                results.append(res_data)
            except Exception as e:
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                results.append({
                    "request_id": req_payload.get("request_id"),
                    "status_code": 0,
                    "error": str(e),
                    "latency_ms": latency_ms,
                })

            time.sleep(delay_between_requests)

        # Ingest outcomes if enabled
        outcomes_ingested = 0
        if simulate_outcomes and pending_outcomes:
            if outcome_delay_seconds > 0:
                time.sleep(outcome_delay_seconds)
            for out in pending_outcomes:
                try:
                    r_out = requests.post(f"{self.api_base_url}/outcomes", json=out, timeout=5.0)
                    if r_out.status_code == 200:
                        outcomes_ingested += 1
                except Exception:
                    pass

        end_time = datetime.datetime.now(datetime.timezone.utc)
        duration_s = (end_time - start_time).total_seconds()
        status_codes = [r.get("status_code", 0) for r in results]
        latencies = [r.get("latency_ms", 0.0) for r in results]

        summary = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": round(duration_s, 2),
            "requests_sent": len(results),
            "success_count": status_codes.count(200),
            "validation_error_count": status_codes.count(422),
            "server_error_count": sum(1 for c in status_codes if c >= 500),
            "outcomes_ingested": outcomes_ingested,
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0,
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        }

        print(f"[{end_time.isoformat()}] Replay completed in {summary['duration_seconds']}s:")
        print(f"  Success (200): {summary['success_count']} | 422: {summary['validation_error_count']} | 5xx: {summary['server_error_count']}")
        print(f"  p95 Latency: {summary['p95_latency_ms']} ms | Outcomes Ingested: {outcomes_ingested}")

        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaultLine Traffic Replayer")
    parser.add_argument("--url", default="http://localhost:8000", help="Prediction API URL")
    parser.add_argument("--requests", type=int, default=50, help="Number of requests")
    parser.add_argument("--rate", type=float, default=15.0, help="Requests per second")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--corrupt-mode", default=None, choices=["invalid_inputs", "data_drift"], help="Fault corruption")
    parser.add_argument("--corrupt-prob", type=float, default=0.3, help="Corruption probability")
    args = parser.parse_args()

    replayer = TrafficReplayer(api_base_url=args.url, seed=args.seed)
    replayer.replay(
        num_requests=args.requests,
        rate_per_second=args.rate,
        corrupt_mode=args.corrupt_mode,
        corrupt_probability=args.corrupt_prob,
    )
