"""
Evidence Bundle Compiler for FaultLine Incident Investigation.

Assigns unique evidence IDs (e.g. [EVID-RULE], [EVID-METRIC-1], [EVID-DEP-1], [EVID-REQ-1])
to all facts, metrics, deployment records, and request samples so AI investigations can cite them.
"""

from __future__ import annotations

import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from src.incidents.models import Incident, DeploymentEvent, PredictionRecord, utcnow


RUNBOOKS = {
    "high_latency": {
        "title": "Service Latency Degradation Runbook",
        "diagnostic_steps": [
            "Check recent infrastructure deployments or resource contention.",
            "Verify database connection pool saturation and query execution times.",
            "Determine if traffic volume spiked or if requests contain unusually large payloads.",
        ],
        "recovery_actions": [
            "Scale API replicas or restart unhealthy pods.",
            "Disable artificial chaos latency injection if in simulated test environment.",
            "If regression followed recent model deployment, execute model rollback.",
        ],
    },
    "server_errors": {
        "title": "Internal Server Error Spike Runbook",
        "diagnostic_steps": [
            "Inspect error logs for unhandled exceptions or 500 status codes.",
            "Check database availability and connectivity status.",
            "Verify if dependent microservices or model artifact storage is unreachable.",
        ],
        "recovery_actions": [
            "Resolve unhandled code exceptions in endpoint handler.",
            "Disable chaos server error injection.",
            "Roll back recent code or model deployments.",
        ],
    },
    "validation_failures": {
        "title": "Schema Validation Rejection Spike Runbook",
        "diagnostic_steps": [
            "Inspect rejected request payloads for missing mandatory keys or invalid categorical values.",
            "Identify the upstream client service generating non-compliant payloads.",
        ],
        "recovery_actions": [
            "Update client service to comply with API schema specification.",
            "Apply backward-compatible schema migration if intentional.",
        ],
    },
    "data_drift": {
        "title": "Covariate Shift / Feature Distribution Drift Runbook",
        "diagnostic_steps": [
            "Identify which specific features failed the 2-sample KS test or exceeded PSI threshold.",
            "Check if business expansion or external seasonality occurred.",
            "Note: Drift indicates distribution change, not confirmed accuracy degradation until outcomes arrive.",
        ],
        "recovery_actions": [
            "Trigger pipeline retraining with recent representative data.",
            "Monitor outcome delay and prediction accuracy closely.",
        ],
    },
    "model_regression": {
        "title": "Delivery Duration Prediction Accuracy Regression Runbook",
        "diagnostic_steps": [
            "Compare observed window MAE against baseline threshold.",
            "Identify whether degradation is concentrated in specific traffic slices (e.g. jam traffic).",
            "Verify recent model activations in deployment audit history.",
        ],
        "recovery_actions": [
            "Execute approved rollback to previous champion model version (champion_v1).",
            "Audit candidate model training pipeline for data leakage or bias.",
        ],
    },
}


class EvidenceBundleCompiler:
    """Compiles structured evidence bundle from database records."""

    @staticmethod
    def compile_bundle(db: Session, incident: Incident) -> Dict[str, Any]:
        """Builds structured evidence bundle with unique IDs."""
        ev = incident.evidence or {}
        evidence_items: Dict[str, Any] = {}

        # 1. Triggered Rule Evidence
        rule_name = ev.get("rule", incident.incident_type)
        evidence_items["EVID-RULE"] = {
            "description": f"Triggered rule '{rule_name}' with severity '{incident.severity}' on model version '{incident.model_version}'",
            "threshold": ev.get("rule_threshold"),
            "first_detected_at": incident.first_detected_at.isoformat(),
            "last_observed_at": incident.last_observed_at.isoformat(),
        }

        # 2. Metric Evidence
        metrics = ev.get("metrics", {})
        evidence_items["EVID-METRIC-1"] = {
            "description": f"Observed metrics during violation: {metrics}",
            "metrics": metrics,
        }

        # 3. Affected Features Evidence
        affected_feats = ev.get("affected_features", [])
        if affected_feats:
            evidence_items["EVID-FEAT-1"] = {
                "description": f"Features failing statistical drift tests: {affected_feats}",
                "features": affected_feats,
            }

        # 4. Recent Deployment Events Evidence
        recent_deployments = (
            db.query(DeploymentEvent)
            .order_by(DeploymentEvent.timestamp.desc())
            .limit(3)
            .all()
        )
        dep_list = []
        for i, dep in enumerate(recent_deployments, start=1):
            dep_id = f"EVID-DEP-{i}"
            item = {
                "event_id": dep.event_id,
                "timestamp": dep.timestamp.isoformat(),
                "activated_version": dep.activated_version,
                "previous_version": dep.previous_version,
                "actor": dep.actor,
                "reason": dep.reason,
            }
            dep_list.append(item)
            evidence_items[dep_id] = {
                "description": f"Model activation: {dep.activated_version} activated by {dep.actor} (reason: {dep.reason})",
                "details": item,
            }

        # 5. Sample Request Evidence
        sample_reqs = ev.get("sample_requests", [])
        for i, req in enumerate(sample_reqs[:3], start=1):
            req_id_key = f"EVID-REQ-{i}"
            evidence_items[req_id_key] = {
                "description": f"Sample request {req.get('request_id')}: status={req.get('status_code')}, latency={req.get('latency_ms')}ms",
                "details": req,
            }

        # 6. Runbook Evidence
        runbook = RUNBOOKS.get(incident.incident_type, RUNBOOKS.get("high_latency", {}))
        evidence_items["EVID-RUNBOOK"] = {
            "title": runbook.get("title"),
            "diagnostic_steps": runbook.get("diagnostic_steps"),
            "recovery_actions": runbook.get("recovery_actions"),
        }

        return {
            "incident_id": incident.incident_id,
            "incident_type": incident.incident_type,
            "model_version": incident.model_version,
            "severity": incident.severity,
            "evidence_items": evidence_items,
            "runbook": runbook,
        }
