"""
Unit tests for FaultLine Monitoring Rules.
"""

import pytest
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.incidents.database import Base
from src.monitoring.rules import RuleEngine
from src.incidents.models import Incident


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_low_sample_window_does_not_alert(db):
    engine = RuleEngine()

    # Only 3 samples (< min_samples 10), but high latency
    window_summary = {
        "total_requests": 3,
        "p95_latency_ms": 600.0,
        "error_rate_percent": 33.3,
        "val_failure_rate_percent": 0.0,
    }
    df_reqs = pd.DataFrame([{"request_id": f"r_{i}", "status_code": 200, "latency_ms": 600.0} for i in range(3)])

    incidents = engine.evaluate_window(
        db=db,
        window_summary=window_summary,
        df_requests=df_reqs,
        model_version="v1",
    )
    # MUST NOT generate alerts due to low sample protection!
    assert len(incidents) == 0
    assert db.query(Incident).count() == 0


def test_sufficient_samples_latency_breach_alerts(db):
    engine = RuleEngine()

    # 15 samples (> min_samples 10) with p95 = 350ms (> warning 150ms)
    window_summary = {
        "total_requests": 15,
        "p95_latency_ms": 350.0,
        "error_rate_percent": 0.0,
        "val_failure_rate_percent": 0.0,
    }
    df_reqs = pd.DataFrame([{"request_id": f"r_{i}", "status_code": 200, "latency_ms": 350.0} for i in range(15)])

    incidents = engine.evaluate_window(
        db=db,
        window_summary=window_summary,
        df_requests=df_reqs,
        model_version="v1",
    )
    assert len(incidents) == 1
    assert incidents[0].incident_type == "high_latency"
    assert incidents[0].status == "Open"


def test_outcome_delay_breach_alerts(db):
    engine = RuleEngine()

    # 30 samples (>= min_predictions 25) with completed_outcomes = 0 (< 20% ratio)
    window_summary = {
        "total_requests": 30,
        "completed_outcomes": 0,
        "p95_latency_ms": 20.0,
        "error_rate_percent": 0.0,
        "val_failure_rate_percent": 0.0,
    }
    df_reqs = pd.DataFrame([{"request_id": f"r_{i}", "status_code": 200, "latency_ms": 20.0} for i in range(30)])

    incidents = engine.evaluate_window(
        db=db,
        window_summary=window_summary,
        df_requests=df_reqs,
        model_version="champion_v1",
    )
    assert len(incidents) == 1
    assert incidents[0].incident_type == "outcome_delay"
    assert incidents[0].status == "Open"
    assert incidents[0].evidence["metrics"]["outcome_ratio"] == 0.0

