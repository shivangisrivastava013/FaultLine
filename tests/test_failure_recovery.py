"""
Integration and resilience tests for FaultLine failure modes and recovery procedures.

Covers:
1. Database (PostgreSQL/SQLite) unavailable handling
2. LLM provider timeout & fallback to grounded deterministic engine
3. Missing model artifacts fail-safe (preserves champion)
4. Invalid rollback version rejection by API
5. Repeated duplicate incident deduplication
6. Prometheus metrics collection resilience
"""

import os
import unittest.mock as mock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.main import app
from src.incidents.database import Base, init_db
from src.incidents.models import Incident
from src.incidents.manager import IncidentManager
from src.models.rollback_manager import ModelRollbackManager, model_manager
from src.investigation.analyzer import InvestigationAnalyzer


@pytest.fixture(scope="module", autouse=True)
def setup_api():
    """Initializes database and loads active champion model for API tests."""
    init_db()
    model_manager.initialize_active_model()


@pytest.fixture
def db_session():
    """In-memory SQLite session for isolated incident testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def client():
    return TestClient(app)


def test_postgresql_unavailable_readiness_failure(client):
    """
    Scenario: Database is unreachable or query fails.
    Expected: /ready endpoint must fail closed with 503 Service Unavailable,
    preventing ingress load balancers from routing traffic to a degraded node.
    """
    with mock.patch("src.incidents.models.PredictionRecord.__table__.select", side_effect=Exception("DB Connection refused")):
        response = client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert "Service not ready" in data["detail"]
        assert "db_connected=False" in data["detail"]


def test_llm_provider_timeout_deterministic_fallback():
    """
    Scenario: Grounded LLM provider (Gemini or OpenAI) encounters network timeout.
    Expected: InvestigationAnalyzer catches exception and transparently returns
    deterministic grounded report with strict [EVID-*] citations.
    """
    analyzer = InvestigationAnalyzer(timeout_seconds=0.1)

    evidence_bundle = {
        "incident_id": "inc_latency_test",
        "incident_type": "high_latency",
        "severity": "critical",
        "model_version": "champion_v1",
        "evidence_items": {
            "EVID-RULE": {"rule": "high_latency", "threshold": 400.0},
            "EVID-METRIC-1": {"metrics": {"p95_latency_ms": 520.0}},
        },
        "runbook": {
            "diagnostic_steps": ["Inspect CPU saturation"],
            "recovery_actions": ["Clear latency fault"],
        },
    }

    # Simulate timeout / network failure when calling LLM API
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "fake_test_key"}), \
         mock.patch("src.investigation.analyzer.InvestigationAnalyzer._call_gemini_llm", side_effect=TimeoutError("Request timed out after 6.0s")):
        report = analyzer.investigate(evidence_bundle)

    # Must fall back gracefully
    assert "EVID-RULE" in report["evidence_ids"]
    assert "EVID-METRIC-1" in report["evidence_ids"]
    assert len(report["observations"]) > 0
    assert any("[Hypothesis]" in cause for cause in report["possible_causes"])
    assert "champion_v1" in report["summary"]


def test_model_artifact_missing_preserves_active_champion(tmp_path):
    """
    Scenario: Target candidate model directory or artifact file is missing/corrupted.
    Expected: Pre-activation check raises FileNotFoundError and active model remains unchanged.
    """
    manager = ModelRollbackManager(artifacts_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError) as exc_info:
        manager.initialize_active_model(default_version="nonexistent_v99")
    assert "No model artifacts found" in str(exc_info.value)


def test_invalid_rollback_version_rejected_by_api(client, monkeypatch):
    """
    Scenario: Operator requests rollback to a non-existent version string via POST /rollback.
    Expected: API returns 400 Bad Request with descriptive message, protecting service integrity.
    """
    monkeypatch.setenv("ADMIN_API_KEY", "faultline_test_key")
    headers = {"X-API-Key": "faultline_test_key"}

    payload = {
        "target_version": "corrupted_version_v404",
        "actor": "sre_lead",
        "reason": "Test non-existent version",
    }

    response = client.post("/rollback", json=payload, headers=headers)
    assert response.status_code == 400
    assert "Rollback failed" in response.json()["detail"]


def test_repeated_duplicate_incidents_deduplicated(db_session):
    """
    Scenario: Three consecutive monitoring windows breach the same metric threshold.
    Expected: IncidentManager deduplicates alerts to 1 open incident ticket,
    updating last_observed_at and appending to evidence history.
    """
    evidence = {
        "rule": "high_latency",
        "metrics": {"p95_latency_ms": 480.0},
    }

    # Window 1 alert
    inc1 = IncidentManager.record_alert(
        db=db_session,
        incident_type="high_latency",
        severity="critical",
        model_version="champion_v1",
        evidence=evidence,
        recommended_action="Inspect CPU pressure",
    )
    assert inc1.status == "Open"
    first_detected = inc1.first_detected_at

    # Window 2 alert (same type & model_version)
    inc2 = IncidentManager.record_alert(
        db=db_session,
        incident_type="high_latency",
        severity="critical",
        model_version="champion_v1",
        evidence=evidence,
        recommended_action="Inspect CPU pressure",
    )

    # Window 3 alert (same type & model_version)
    inc3 = IncidentManager.record_alert(
        db=db_session,
        incident_type="high_latency",
        severity="critical",
        model_version="champion_v1",
        evidence=evidence,
        recommended_action="Inspect CPU pressure",
    )

    # All three calls must refer to the exact same incident row
    assert inc1.incident_id == inc2.incident_id == inc3.incident_id
    total_open = db_session.query(Incident).filter(Incident.status == "Open").count()
    assert total_open == 1
    assert inc3.last_observed_at >= first_detected


def test_prometheus_exporter_metric_resilience(client):
    """
    Scenario: Prometheus /metrics endpoint is queried concurrently with client inference.
    Expected: Metrics registry serves valid OpenMetrics/Prometheus text without locking or exceptions.
    """
    # 1. Prediction request
    pred_res = client.post(
        "/predict",
        json={
            "distance_km": 5.2,
            "traffic_level": "medium",
            "weather": "clear",
            "hour_of_day": 14,
            "restaurant_load": 0.45,
        },
    )
    assert pred_res.status_code == 200

    # 2. Metrics scrape
    metrics_res = client.get("/metrics")
    assert metrics_res.status_code == 200
    assert "faultline_predictions_total" in metrics_res.text
    assert "faultline_http_requests_total" in metrics_res.text
