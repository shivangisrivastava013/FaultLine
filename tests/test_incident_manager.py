"""
Unit tests for FaultLine Incident Lifecycle & Deduplication.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.incidents.database import Base
from src.incidents.manager import IncidentManager
from src.incidents.models import Incident, IncidentAction


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_incident_creation_and_deduplication(db):
    # 1. First alert creates incident
    evidence1 = {"rule": "high_latency", "metrics": {"p95_latency_ms": 350.0}}
    inc1 = IncidentManager.record_alert(
        db=db,
        incident_type="high_latency",
        severity="warning",
        model_version="v1",
        evidence=evidence1,
        recommended_action="Check latency",
    )
    assert inc1.status == "Open"
    assert inc1.model_version == "v1"
    inc_id = inc1.incident_id

    # 2. Repeated alert for SAME rule and model version MUST update existing incident, NOT create a second one!
    evidence2 = {"rule": "high_latency", "metrics": {"p95_latency_ms": 420.0}}
    inc2 = IncidentManager.record_alert(
        db=db,
        incident_type="high_latency",
        severity="critical",
        model_version="v1",
        evidence=evidence2,
        recommended_action="Check latency",
    )
    assert inc2.incident_id == inc_id  # Same incident updated
    assert inc2.severity == "critical"

    # Verify total incidents count is still 1
    total = db.query(Incident).count()
    assert total == 1


def test_incident_state_machine_transitions(db):
    inc = IncidentManager.record_alert(
        db=db,
        incident_type="server_errors",
        severity="critical",
        model_version="v1",
        evidence={"metrics": {"error_rate_percent": 12.0}},
        recommended_action="Inspect logs",
    )
    assert inc.status == "Open"

    # Open -> Acknowledged
    ack_inc = IncidentManager.acknowledge_incident(db, inc.incident_id, actor="alice_sre")
    assert ack_inc.status == "Acknowledged"

    # Acknowledged -> Resolved
    res_inc = IncidentManager.resolve_incident(db, inc.incident_id, actor="alice_sre", resolution_notes="Restarted node")
    assert res_inc.status == "Resolved"
    assert res_inc.resolved_at is not None

    # Check audit trail
    actions = IncidentManager.get_actions_for_incident(db, inc.incident_id)
    action_types = [a.action_type for a in actions]
    assert "created" in action_types
    assert "acknowledged" in action_types
    assert "resolved" in action_types
