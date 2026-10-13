"""
SQLAlchemy ORM Models for FaultLine.
"""

from __future__ import annotations

import datetime
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    DateTime,
    Text,
    Boolean,
    JSON,
    Index,
)
from src.incidents.database import Base


def utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class PredictionRecord(Base):
    """Logs incoming prediction requests, features, outputs, and latencies."""
    __tablename__ = "prediction_requests"

    request_id = Column(String(64), primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, index=True, nullable=False)
    model_version = Column(String(32), index=True, nullable=False)
    distance_km = Column(Float, nullable=True)
    traffic_level = Column(String(16), nullable=True)
    weather = Column(String(16), nullable=True)
    hour_of_day = Column(Integer, nullable=True)
    restaurant_load = Column(Float, nullable=True)
    predicted_minutes = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=False)
    status_code = Column(Integer, default=200, nullable=False)
    error_message = Column(Text, nullable=True)


class OutcomeRecord(Base):
    """Ground truth delivery outcomes recorded after transit finishes."""
    __tablename__ = "outcomes"

    outcome_id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), unique=True, index=True, nullable=False)
    actual_delivery_minutes = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)


class MonitoringWindow(Base):
    """Aggregated operational and statistical windows computed by monitoring worker."""
    __tablename__ = "monitoring_windows"

    window_id = Column(String(64), primary_key=True, index=True)
    start_time = Column(DateTime(timezone=True), index=True, nullable=False)
    end_time = Column(DateTime(timezone=True), index=True, nullable=False)
    sample_count = Column(Integer, default=0, nullable=False)
    error_rate = Column(Float, default=0.0, nullable=False)
    validation_failure_rate = Column(Float, default=0.0, nullable=False)
    p95_latency_ms = Column(Float, default=0.0, nullable=False)
    mae = Column(Float, nullable=True)
    status = Column(String(32), default="ok", nullable=False)
    metrics_summary = Column(JSON, nullable=True)


class Incident(Base):
    """Operational or data incident detected by FaultLine."""
    __tablename__ = "incidents"

    incident_id = Column(String(64), primary_key=True, index=True)
    incident_type = Column(String(64), index=True, nullable=False)
    severity = Column(String(16), default="warning", nullable=False)
    status = Column(String(16), default="Open", index=True, nullable=False)  # Open, Acknowledged, Resolved
    first_detected_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    last_observed_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    model_version = Column(String(32), index=True, nullable=False)
    evidence = Column(JSON, nullable=False)
    recommended_action = Column(Text, nullable=False)
    resolution_notes = Column(Text, nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_incident_rule_version_status", "incident_type", "model_version", "status"),
    )


class DeploymentEvent(Base):
    """Audit trail of model activations, promotions, and rollbacks."""
    __tablename__ = "deployment_events"

    event_id = Column(String(64), primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    activated_version = Column(String(32), nullable=False)
    previous_version = Column(String(32), nullable=False)
    actor = Column(String(64), nullable=False)
    reason = Column(Text, nullable=False)
    smoke_test_passed = Column(Boolean, default=True, nullable=False)


class IncidentAction(Base):
    """Lifecycle actions applied to an incident (acknowledge, rollback, investigate, resolve)."""
    __tablename__ = "incident_actions"

    action_id = Column(String(64), primary_key=True, index=True)
    incident_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    actor = Column(String(64), nullable=False)
    action_type = Column(String(32), nullable=False)
    details = Column(JSON, nullable=True)
