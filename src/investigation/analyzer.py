"""
Grounded AI Investigation Analyzer for FaultLine.

Analyzes structured incident evidence and produces an auditable root-cause report:
- Observations strictly cite evidence IDs (e.g. [EVID-METRIC-1], [EVID-DEP-1]).
- Causes are labeled as hypotheses.
- Missing evidence triggers an explicit uncertainty statement.
- The analyzer has NO ability to execute rollbacks directly.
- Fully operational even when external LLM API is unavailable (deterministic fallback).
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Any, List, Optional
import httpx


class InvestigationAnalyzer:
    """Produces grounded investigations from structured incident evidence bundles."""

    def __init__(self, timeout_seconds: float = 6.0):
        self.timeout_seconds = timeout_seconds
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

    def investigate(self, evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates an investigation report. Attempts external LLM first if configured;
        falls back to deterministic grounded analysis on timeout or missing credentials.
        """
        if self.api_key and os.getenv("OPENAI_API_KEY"):
            try:
                return self._call_openai_llm(evidence_bundle)
            except Exception as e:
                print(f"[InvestigationAnalyzer] LLM call failed or timed out ({e}). Using deterministic fallback.")

        return self._generate_deterministic_analysis(evidence_bundle)

    def _call_openai_llm(self, evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Calls OpenAI compatible endpoint with strict JSON schema instructions."""
        prompt = (
            "You are an expert SRE and ML Operations investigator analyzing a production failure in the delivery prediction service.\n"
            "Below is a structured evidence bundle containing verified evidence items with unique IDs.\n\n"
            f"Evidence Bundle:\n{json.dumps(evidence_bundle, indent=2)}\n\n"
            "Requirements:\n"
            "1. Output MUST be valid JSON with keys: summary, observations, possible_causes, evidence_ids, recommended_checks, suggested_action.\n"
            "2. Every observation MUST explicitly cite one or more evidence IDs in square brackets (e.g. [EVID-METRIC-1], [EVID-DEP-1]).\n"
            "3. Every possible cause MUST be prefixed with '[Hypothesis]'.\n"
            "4. If any critical evidence is missing, state it under observations or summary as an uncertainty statement.\n"
            "5. You cannot execute rollbacks or actions; only recommend them.\n"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            self._validate_grounding(parsed, evidence_bundle)
            return parsed

    def _generate_deterministic_analysis(self, evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        High-fidelity deterministic investigation engine.
        Guarantees strict citation of evidence IDs, hypothesis tagging, and runbook alignment.
        """
        inc_type = evidence_bundle.get("incident_type", "unknown")
        evidence_items = evidence_bundle.get("evidence_items", {})
        model_version = evidence_bundle.get("model_version", "unknown")
        runbook = evidence_bundle.get("runbook", {})

        cited_evidence_ids: List[str] = ["EVID-RULE"]
        observations: List[str] = []
        possible_causes: List[str] = []
        recommended_checks: List[str] = []
        suggested_action = ""

        # 1. Rule trigger observation
        if "EVID-RULE" in evidence_items:
            r_ev = evidence_items["EVID-RULE"]
            observations.append(
                f"Incident triggered by rule '{inc_type}' for model version '{model_version}' (threshold: {r_ev.get('threshold')}) [EVID-RULE]."
            )

        # 2. Metric observations
        if "EVID-METRIC-1" in evidence_items:
            cited_evidence_ids.append("EVID-METRIC-1")
            metrics = evidence_items["EVID-METRIC-1"].get("metrics", {})
            if inc_type == "high_latency":
                p95 = metrics.get("p95_latency_ms", "N/A")
                observations.append(
                    f"Observed p95 latency spike of {p95} ms against healthy baseline [EVID-METRIC-1]."
                )
                possible_causes.append("[Hypothesis] Injected chaos latency fault or downstream connection stall.")
                possible_causes.append("[Hypothesis] Increased algorithmic complexity or CPU saturation on inference.")
            elif inc_type == "server_errors":
                err_rate = metrics.get("error_rate_percent", "N/A")
                observations.append(
                    f"Observed internal server error rate of {err_rate}% on incoming prediction requests [EVID-METRIC-1]."
                )
                possible_causes.append("[Hypothesis] Unhandled exception in prediction handler or database query timeout.")
                possible_causes.append("[Hypothesis] Injected chaos server error active in testing environment.")
            elif inc_type == "validation_failures":
                val_rate = metrics.get("validation_failure_rate_percent", "N/A")
                observations.append(
                    f"Observed {val_rate}% of requests rejected with 422 Unprocessable Entity [EVID-METRIC-1]."
                )
                possible_causes.append("[Hypothesis] Upstream client service deploying schema breaking changes.")
                possible_causes.append("[Hypothesis] Malformed payload containing missing fields or invalid categories.")
            elif inc_type == "model_regression":
                mae = metrics.get("window_mae", "N/A")
                observations.append(
                    f"Observed Mean Absolute Error (MAE) degraded to {mae} minutes, breaching threshold [EVID-METRIC-1]."
                )
                possible_causes.append("[Hypothesis] Newly activated model candidate (v2_degraded) exhibits poor generalization on non-trivial traffic.")
                possible_causes.append("[Hypothesis] Underfit training slices causing large prediction divergence in jam traffic.")

        # 3. Affected features observation
        if "EVID-FEAT-1" in evidence_items:
            cited_evidence_ids.append("EVID-FEAT-1")
            feats = evidence_items["EVID-FEAT-1"].get("features", [])
            observations.append(
                f"Statistical drift detected in input feature distributions: {', '.join(feats)} [EVID-FEAT-1]."
            )
            possible_causes.append("[Hypothesis] Real-world operational shift (e.g. severe weather storm or expanded delivery delivery radius).")
            possible_causes.append("[Hypothesis] Sensor calibration change or traffic data source anomaly.")

        # 4. Recent deployment observation
        dep_found = False
        for k in evidence_items:
            if k.startswith("EVID-DEP"):
                cited_evidence_ids.append(k)
                dep = evidence_items[k].get("details", {})
                observations.append(
                    f"Recent deployment event: {dep.get('activated_version')} activated by {dep.get('actor')} ({dep.get('reason')}) [{k}]."
                )
                dep_found = True
                break

        if not dep_found:
            observations.append(
                "No recent model activations detected in the last hour; code/model change unlikely primary cause [Uncertainty Statement]."
            )

        # 5. Request sample citations
        req_keys = [k for k in evidence_items if k.startswith("EVID-REQ")]
        if req_keys:
            cited_evidence_ids.extend(req_keys)
            observations.append(
                f"Inspected sample failing/delayed requests from the current window [{', '.join(req_keys)}]."
            )

        # 6. Runbook diagnostic steps
        diag_steps = runbook.get("diagnostic_steps", [])
        recommended_checks.extend(diag_steps)

        # 7. Suggested action
        rec_actions = runbook.get("recovery_actions", [])
        if inc_type == "model_regression":
            suggested_action = "Execute model rollback to champion_v1 via authenticated control panel."
        elif inc_type in ["high_latency", "server_errors"]:
            suggested_action = "Inspect error logs, verify container health, and disable active chaos faults."
        elif inc_type == "validation_failures":
            suggested_action = "Notify upstream client teams to fix payload schema violations."
        elif inc_type == "data_drift":
            suggested_action = "Flag feature drift to data science team; monitor outcome accuracy closely."
        else:
            suggested_action = rec_actions[0] if rec_actions else "Inspect service logs."

        summary = (
            f"FaultLine detected a {inc_type.replace('_', ' ')} incident on model '{model_version}'. "
            f"Analysis of telemetry indicates anomalous behavior supported by {len(cited_evidence_ids)} verified evidence sources."
        )

        result = {
            "summary": summary,
            "observations": observations,
            "possible_causes": possible_causes,
            "evidence_ids": cited_evidence_ids,
            "recommended_checks": recommended_checks,
            "suggested_action": suggested_action,
        }

        self._validate_grounding(result, evidence_bundle)
        return result

    def _validate_grounding(self, report: Dict[str, Any], evidence_bundle: Dict[str, Any]) -> None:
        """Verifies that all cited evidence IDs exist in the provided evidence bundle."""
        available_ids = set(evidence_bundle.get("evidence_items", {}).keys())
        # Check citations in observations
        found_citations = set()
        for obs in report.get("observations", []):
            matches = re.findall(r"\[(EVID-[A-Z0-9\-]+)\]", obs)
            for m in matches:
                found_citations.add(m)

        invalid_citations = found_citations - available_ids
        if invalid_citations:
            print(f"[Warning] Analyzer cited unrecognized evidence IDs: {invalid_citations}")

        report["evidence_ids"] = list(found_citations.intersection(available_ids))
