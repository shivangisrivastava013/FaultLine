"""
Unit tests for FaultLine Grounded AI Investigation Analyzer.
"""

import pytest
from src.investigation.analyzer import InvestigationAnalyzer


@pytest.fixture
def sample_evidence_bundle():
    return {
        "incident_id": "inc_latency_123",
        "incident_type": "high_latency",
        "model_version": "v1",
        "severity": "critical",
        "evidence_items": {
            "EVID-RULE": {
                "description": "high_latency rule breached (threshold 150ms)",
                "threshold": 150.0,
            },
            "EVID-METRIC-1": {
                "description": "p95 latency reached 512ms",
                "metrics": {"p95_latency_ms": 512.0, "total_requests": 50},
            },
            "EVID-DEP-1": {
                "description": "Model activation v2_degraded by release_bot",
                "details": {"activated_version": "v2_degraded", "actor": "release_bot", "reason": "canary"},
            },
            "EVID-REQ-1": {
                "description": "Sample request req_999 latency=520ms",
                "details": {"request_id": "req_999", "status_code": 200, "latency_ms": 520.0},
            },
        },
        "runbook": {
            "diagnostic_steps": ["Check container CPU", "Inspect DB connection pool"],
            "recovery_actions": ["Execute model rollback if recent deployment"],
        },
    }


def test_deterministic_investigation_grounding(sample_evidence_bundle):
    analyzer = InvestigationAnalyzer()
    report = analyzer.investigate(sample_evidence_bundle)

    # 1. Output structure
    assert "summary" in report
    assert "observations" in report
    assert "possible_causes" in report
    assert "evidence_ids" in report
    assert "recommended_checks" in report
    assert "suggested_action" in report

    # 2. Observations must cite evidence IDs
    obs_text = " ".join(report["observations"])
    assert "[EVID-RULE]" in obs_text
    assert "[EVID-METRIC-1]" in obs_text
    assert "[EVID-DEP-1]" in obs_text

    # 3. Possible causes must be labeled as hypotheses
    for cause in report["possible_causes"]:
        assert cause.startswith("[Hypothesis]")

    # 4. Validated evidence IDs list
    for ev_id in report["evidence_ids"]:
        assert ev_id in sample_evidence_bundle["evidence_items"]


def test_uncertainty_statement_when_deployment_is_missing():
    bundle_no_dep = {
        "incident_id": "inc_latency_456",
        "incident_type": "high_latency",
        "model_version": "v1",
        "severity": "warning",
        "evidence_items": {
            "EVID-RULE": {"threshold": 150.0},
            "EVID-METRIC-1": {"metrics": {"p95_latency_ms": 180.0}},
        },
        "runbook": {},
    }
    analyzer = InvestigationAnalyzer()
    report = analyzer.investigate(bundle_no_dep)
    obs_text = " ".join(report["observations"])
    assert "uncertainty statement" in obs_text.lower()
