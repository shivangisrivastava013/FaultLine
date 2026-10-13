"""
Incident Lifecycle & State Machine Manager for FaultLine.

Implements:
- Incident deduplication (updates existing Open/Acknowledged incident instead of creating duplicates)
- State transitions: Open -> Acknowledged -> Resolved
- Audit trail recording in incident_actions table
"""

from __future__ import annotations

import uuid
import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from src.incidents.models import Incident, IncidentAction, DeploymentEvent, utcnow


class IncidentManager:
    """Manages creation, deduplication, updates, and transitions for incidents."""

    @staticmethod
    def record_alert(
        db: Session,
        incident_type: str,
        severity: str,
        model_version: str,
        evidence: Dict[str, Any],
        recommended_action: str,
    ) -> Incident:
        """
        Records an alert. If an active (Open or Acknowledged) incident already exists
        for this (incident_type, model_version), updates last_observed_at and evidence.
        Otherwise, creates a new Open incident.
        """
        # Look for existing active incident
        existing = (
            db.query(Incident)
            .filter(
                Incident.incident_type == incident_type,
                Incident.model_version == model_version,
                Incident.status.in_(["Open", "Acknowledged"]),
            )
            .first()
        )

        now = utcnow()

        if existing:
            existing.last_observed_at = now
            existing.severity = severity  # escalate if needed

            # Merge evidence history
            ev = dict(existing.evidence or {})
            history = ev.get("history", [])
            history.append({
                "timestamp": now.isoformat(),
                "metrics": evidence.get("metrics", {}),
            })
            # keep latest 20 snapshots
            ev["history"] = history[-20:]
            ev["latest_metrics"] = evidence.get("metrics", {})
            if "sample_requests" in evidence:
                ev["sample_requests"] = evidence["sample_requests"]
            if "affected_features" in evidence:
                ev["affected_features"] = evidence["affected_features"]
            if "rule_threshold" in evidence:
                ev["rule_threshold"] = evidence["rule_threshold"]
            existing.evidence = ev

            db.commit()
            db.refresh(existing)

            # Record action update
            action = IncidentAction(
                action_id=f"act_{uuid.uuid4().hex[:12]}",
                incident_id=existing.incident_id,
                timestamp=now,
                actor="monitoring_worker",
                action_type="updated",
                details={"evidence_update": evidence.get("metrics", {})},
            )
            db.add(action)
            db.commit()
            return existing

        # Create new incident
        inc_id = f"inc_{incident_type}_{uuid.uuid4().hex[:8]}"
        initial_evidence = dict(evidence)
        initial_evidence["history"] = [{
            "timestamp": now.isoformat(),
            "metrics": evidence.get("metrics", {}),
        }]

        new_incident = Incident(
            incident_id=inc_id,
            incident_type=incident_type,
            severity=severity,
            status="Open",
            first_detected_at=now,
            last_observed_at=now,
            model_version=model_version,
            evidence=initial_evidence,
            recommended_action=recommended_action,
        )
        db.add(new_incident)
        db.commit()
        db.refresh(new_incident)

        # Record action creation
        action = IncidentAction(
            action_id=f"act_{uuid.uuid4().hex[:12]}",
            incident_id=inc_id,
            timestamp=now,
            actor="monitoring_worker",
            action_type="created",
            details={"initial_metrics": evidence.get("metrics", {})},
        )
        db.add(action)
        db.commit()

        return new_incident

    @staticmethod
    def acknowledge_incident(
        db: Session,
        incident_id: str,
        actor: str = "operator",
        notes: Optional[str] = None,
    ) -> Optional[Incident]:
        """Transitions incident from Open -> Acknowledged."""
        inc = db.query(Incident).filter(Incident.incident_id == incident_id).first()
        if not inc:
            return None

        if inc.status == "Open":
            inc.status = "Acknowledged"
            now = utcnow()
            action = IncidentAction(
                action_id=f"act_{uuid.uuid4().hex[:12]}",
                incident_id=incident_id,
                timestamp=now,
                actor=actor,
                action_type="acknowledged",
                details={"notes": notes or "Acknowledged by operator"},
            )
            db.add(action)
            db.commit()
            db.refresh(inc)

        return inc

    @staticmethod
    def resolve_incident(
        db: Session,
        incident_id: str,
        actor: str = "operator",
        resolution_notes: str = "Resolved following recovery runbook",
    ) -> Optional[Incident]:
        """Transitions incident to Resolved."""
        inc = db.query(Incident).filter(Incident.incident_id == incident_id).first()
        if not inc:
            return None

        now = utcnow()
        inc.status = "Resolved"
        inc.resolved_at = now
        inc.resolution_notes = resolution_notes

        action = IncidentAction(
            action_id=f"act_{uuid.uuid4().hex[:12]}",
            incident_id=incident_id,
            timestamp=now,
            actor=actor,
            action_type="resolved",
            details={"resolution_notes": resolution_notes},
        )
        db.add(action)
        db.commit()
        db.refresh(inc)
        return inc

    @staticmethod
    def list_incidents(
        db: Session,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Incident]:
        """Lists incidents optionally filtered by status, ordered by last_observed_at desc."""
        query = db.query(Incident)
        if status:
            query = query.filter(Incident.status == status)
        return query.order_by(Incident.last_observed_at.desc()).limit(limit).all()

    @staticmethod
    def get_incident(db: Session, incident_id: str) -> Optional[Incident]:
        """Fetches incident by ID."""
        return db.query(Incident).filter(Incident.incident_id == incident_id).first()

    @staticmethod
    def get_actions_for_incident(db: Session, incident_id: str) -> List[IncidentAction]:
        """Fetches audit trail actions for an incident."""
        return (
            db.query(IncidentAction)
            .filter(IncidentAction.incident_id == incident_id)
            .order_by(IncidentAction.timestamp.asc())
            .all()
        )

    @staticmethod
    def record_deployment_event(
        db: Session,
        activated_version: str,
        previous_version: str,
        actor: str,
        reason: str,
        smoke_test_passed: bool = True,
    ) -> DeploymentEvent:
        """Records a model activation or rollback event."""
        now = utcnow()
        event = DeploymentEvent(
            event_id=f"dep_{uuid.uuid4().hex[:12]}",
            timestamp=now,
            activated_version=activated_version,
            previous_version=previous_version,
            actor=actor,
            reason=reason,
            smoke_test_passed=smoke_test_passed,
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        return event

    @staticmethod
    def list_deployment_events(db: Session, limit: int = 20) -> List[DeploymentEvent]:
        """Lists deployment audit events ordered by timestamp desc."""
        return db.query(DeploymentEvent).order_by(DeploymentEvent.timestamp.desc()).limit(limit).all()
