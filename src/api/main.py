"""
FaultLine Prediction API (FastAPI Service).

Provides:
- GET /health (Process is running)
- GET /ready (Model loaded & DB reachable)
- POST /predict (Validate input & return delivery time prediction)
- GET /model-info (Active version & training metadata)
- POST /outcomes (Delayed actual delivery outcome recording)
- POST /rollback (Audited model rollback with smoke check)
- POST /admin/activate-model (Model version switching)
- POST /admin/inject-fault (Simulated fault injection for testing)
- GET /metrics (Prometheus scrapable metrics)
- Structured DB logging for every request & latency tracking
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import datetime
import secrets
from typing import Dict, Any, Optional

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import FastAPI, Depends, HTTPException, Request, Response, status, Header
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import pandas as pd

from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelInfoResponse,
    HealthResponse,
    ReadyResponse,
    OutcomeRequest,
    OutcomeResponse,
    RollbackRequest,
    RollbackResponse,
    FaultInjectionRequest,
)
from src.incidents.database import get_db, init_db
from src.incidents.models import PredictionRecord, OutcomeRecord, utcnow
from src.incidents.manager import IncidentManager
from src.models.rollback_manager import model_manager
from src.monitoring.metrics import (
    REQUESTS_TOTAL,
    REQUEST_DURATION_SECONDS,
    SERVER_ERRORS_TOTAL,
    VALIDATION_FAILURES_TOTAL,
    PREDICTIONS_TOTAL,
    OUTCOMES_TOTAL,
    get_metrics_payload,
)


START_TIME = time.time()


def verify_admin_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    """
    Validates X-API-Key header against the required ADMIN_API_KEY environment variable.
    Uses constant-time comparison (secrets.compare_digest) to prevent timing attacks.
    """
    configured_key = os.getenv("ADMIN_API_KEY")
    if not configured_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ADMIN_API_KEY environment variable is not configured on the server",
        )
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key administrative header",
        )
    if not secrets.compare_digest(x_api_key, configured_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Invalid administrative API key",
        )
    return x_api_key


# In-memory runtime fault injection state (for controlled simulation)
FAULT_STATE: Dict[str, Any] = {
    "latency_delay_seconds": 0.0,
    "latency_probability": 0.0,
    "server_error_probability": 0.0,
    "server_error_status": 500,
    "server_error_message": "Injected chaos server error",
}

app = FastAPI(
    title="FaultLine Delivery Time Prediction Service",
    description="Monitored ML API tracking service health, data quality, drift, and performance regressions.",
    version="0.1.0",
)

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    """Initializes database schema and loads champion model artifact."""
    init_db()
    try:
        model_manager.initialize_active_model()
    except Exception as e:
        print(f"Warning: Model could not be loaded at startup: {e}. Model will load once trained.")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Custom handler for 422 validation rejections:
    Increments Prometheus VALIDATION_FAILURES_TOTAL and records rejected input to DB for monitoring.
    """
    failed_features = []
    for err in exc.errors():
        loc = err.get("loc", [])
        if len(loc) > 1:
            failed_features.append(str(loc[1]))

    feature_tag = failed_features[0] if failed_features else "general"
    VALIDATION_FAILURES_TOTAL.labels(endpoint="/predict", feature_failed=feature_tag).inc()
    REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status_code="422").inc()

    # Log rejection in database if possible
    try:
        raw_body = await request.body()
        req_id = f"rej_{uuid.uuid4().hex[:12]}"
        db: Session = next(get_db())
        record = PredictionRecord(
            request_id=req_id,
            timestamp=utcnow(),
            model_version=model_manager.active_version,
            latency_ms=0.0,
            status_code=422,
            error_message=str(exc),
        )
        db.add(record)
        db.commit()
    except Exception:
        pass

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "message": "Input validation failed"},
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    """Liveness check: process is running."""
    REQUESTS_TOTAL.labels(method="GET", endpoint="/health", status_code="200").inc()
    return HealthResponse(
        status="ok",
        service="faultline-prediction-api",
        uptime_seconds=round(time.time() - START_TIME, 2),
    )


@app.get("/ready", response_model=ReadyResponse, tags=["Health"])
def ready(db: Session = Depends(get_db)) -> ReadyResponse:
    """Readiness check: model is loaded and database is responsive."""
    is_loaded = model_manager.active_pipeline is not None
    db_ok = True
    try:
        db.execute(PredictionRecord.__table__.select().limit(1))
    except Exception:
        db_ok = False

    status_code = "200" if (is_loaded and db_ok) else "503"
    REQUESTS_TOTAL.labels(method="GET", endpoint="/ready", status_code=status_code).inc()

    if not (is_loaded and db_ok):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready (model_loaded={is_loaded}, db_connected={db_ok})",
        )

    return ReadyResponse(
        status="ready",
        model_loaded=is_loaded,
        model_version=model_manager.active_version,
        database_connected=db_ok,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info() -> ModelInfoResponse:
    """Returns active model version, type, and training evaluation metrics."""
    REQUESTS_TOTAL.labels(method="GET", endpoint="/model-info", status_code="200").inc()
    meta = model_manager.active_metadata
    val_m = meta.get("val_metrics", {}).get("overall_mae")
    test_m = meta.get("test_metrics", {}).get("overall_mae")

    return ModelInfoResponse(
        model_version=model_manager.active_version,
        model_type=meta.get("model_type", "unknown"),
        status=meta.get("status", "active"),
        features=meta.get("features", ["distance_km", "traffic_level", "weather", "hour_of_day", "restaurant_load"]),
        artifact_path=os.path.join(model_manager.artifacts_dir, model_manager.active_version),
        val_mae=val_m,
        test_mae=test_m,
        active_since=datetime.datetime.fromtimestamp(START_TIME, tz=datetime.timezone.utc).isoformat(),
        additional_metadata=meta,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(payload: PredictionRequest, db: Session = Depends(get_db)) -> PredictionResponse:
    """
    Validates delivery request, evaluates model, logs record to DB,
    and updates Prometheus metrics.
    """
    t0 = time.perf_counter()
    req_id = payload.request_id or f"req_{uuid.uuid4().hex[:12]}"
    # Prevent integrity error on request_id collision
    if db.query(PredictionRecord).filter(PredictionRecord.request_id == req_id).first():
        req_id = f"{req_id}_{uuid.uuid4().hex[:6]}"

    # --- Simulated Chaos: Latency Injection ---
    if FAULT_STATE["latency_probability"] > 0:
        import random
        if random.random() < FAULT_STATE["latency_probability"]:
            delay = FAULT_STATE["latency_delay_seconds"]
            time.sleep(delay)

    # --- Simulated Chaos: Server Error Injection ---
    if FAULT_STATE["server_error_probability"] > 0:
        import random
        if random.random() < FAULT_STATE["server_error_probability"]:
            SERVER_ERRORS_TOTAL.labels(endpoint="/predict", error_type="SimulatedServerError").inc()
            REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status_code=str(FAULT_STATE["server_error_status"])).inc()
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            # Log failure to DB
            err_record = PredictionRecord(
                request_id=req_id,
                timestamp=utcnow(),
                model_version=model_manager.active_version,
                distance_km=payload.distance_km,
                traffic_level=payload.traffic_level,
                weather=payload.weather,
                hour_of_day=payload.hour_of_day,
                restaurant_load=payload.restaurant_load,
                latency_ms=latency_ms,
                status_code=FAULT_STATE["server_error_status"],
                error_message=FAULT_STATE["server_error_message"],
            )
            db.add(err_record)
            db.commit()
            raise HTTPException(
                status_code=FAULT_STATE["server_error_status"],
                detail=FAULT_STATE["server_error_message"],
            )

    # --- Model Prediction ---
    df_in = pd.DataFrame([{
        "distance_km": payload.distance_km,
        "traffic_level": payload.traffic_level,
        "weather": payload.weather,
        "hour_of_day": payload.hour_of_day,
        "restaurant_load": payload.restaurant_load,
    }])

    try:
        predicted_minutes, active_ver = model_manager.predict(df_in)
    except Exception as e:
        SERVER_ERRORS_TOTAL.labels(endpoint="/predict", error_type=type(e).__name__).inc()
        REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status_code="500").inc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    duration_seconds = time.perf_counter() - t0
    latency_ms = round(duration_seconds * 1000, 2)

    # Prometheus metrics
    REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status_code="200").inc()
    REQUEST_DURATION_SECONDS.labels(endpoint="/predict").observe(duration_seconds)
    PREDICTIONS_TOTAL.labels(model_version=active_ver).inc()

    # Persist structured record to DB
    record = PredictionRecord(
        request_id=req_id,
        timestamp=utcnow(),
        model_version=active_ver,
        distance_km=payload.distance_km,
        traffic_level=payload.traffic_level,
        weather=payload.weather,
        hour_of_day=payload.hour_of_day,
        restaurant_load=payload.restaurant_load,
        predicted_minutes=predicted_minutes,
        latency_ms=latency_ms,
        status_code=200,
    )
    db.add(record)
    db.commit()

    return PredictionResponse(
        request_id=req_id,
        predicted_minutes=predicted_minutes,
        model_version=active_ver,
        latency_ms=latency_ms,
        timestamp=record.timestamp.isoformat(),
    )


@app.post("/outcomes", response_model=OutcomeResponse, tags=["Outcomes"])
def record_outcome(payload: OutcomeRequest, db: Session = Depends(get_db)) -> OutcomeResponse:
    """
    Ingests actual ground truth delivery time once trip completes.
    Matches outcome to original prediction for delay/accuracy monitoring.
    """
    # Check if request_id exists in prediction_requests
    pred = db.query(PredictionRecord).filter(PredictionRecord.request_id == payload.request_id).first()
    if not pred:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction request_id '{payload.request_id}' not found",
        )

    # Check for duplicate outcome
    existing_outcome = db.query(OutcomeRecord).filter(OutcomeRecord.request_id == payload.request_id).first()
    if existing_outcome:
        return OutcomeResponse(
            status="already_recorded",
            request_id=payload.request_id,
            actual_delivery_minutes=existing_outcome.actual_delivery_minutes,
            error_delta_minutes=round(abs(pred.predicted_minutes - existing_outcome.actual_delivery_minutes), 2)
            if pred.predicted_minutes else None,
        )

    outcome = OutcomeRecord(
        request_id=payload.request_id,
        actual_delivery_minutes=payload.actual_delivery_minutes,
        created_at=utcnow(),
    )
    db.add(outcome)
    db.commit()

    OUTCOMES_TOTAL.labels(model_version=pred.model_version).inc()
    REQUESTS_TOTAL.labels(method="POST", endpoint="/outcomes", status_code="200").inc()

    delta = None
    if pred.predicted_minutes is not None:
        delta = round(abs(pred.predicted_minutes - payload.actual_delivery_minutes), 2)

    return OutcomeResponse(
        status="recorded",
        request_id=payload.request_id,
        actual_delivery_minutes=payload.actual_delivery_minutes,
        error_delta_minutes=delta,
    )


@app.post(
    "/rollback",
    response_model=RollbackResponse,
    tags=["Rollback"],
    dependencies=[Depends(verify_admin_key)],
)
def rollback_model(payload: RollbackRequest, db: Session = Depends(get_db)) -> RollbackResponse:
    """
    Performs audited model rollback with mandatory pre-activation smoke test.
    Protected by X-API-Key administrative dependency.
    """
    target = payload.target_version or "champion_v1"
    try:
        res = model_manager.activate_version(
            target_version=target,
            actor=payload.actor,
            reason=payload.reason,
            db=db,
        )
        return RollbackResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Rollback failed: {str(e)}")


@app.post(
    "/admin/activate-model",
    tags=["Admin"],
    dependencies=[Depends(verify_admin_key)],
)
def admin_activate_model(
    version: str,
    actor: str = "admin",
    reason: str = "Admin manual switch",
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Activates a specific model version (e.g. champion_v1, candidate_degraded_v2)."""
    try:
        return model_manager.activate_version(
            target_version=version,
            actor=actor,
            reason=reason,
            db=db,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/admin/inject-fault",
    tags=["Admin"],
    dependencies=[Depends(verify_admin_key)],
)
def admin_inject_fault(payload: FaultInjectionRequest) -> Dict[str, Any]:
    """Configures simulated live operational faults for testing and demonstrations."""
    global FAULT_STATE
    if payload.fault_type == "latency_spike":
        FAULT_STATE["latency_delay_seconds"] = payload.delay_seconds
        FAULT_STATE["latency_probability"] = payload.probability
    elif payload.fault_type == "server_error":
        FAULT_STATE["server_error_probability"] = payload.probability
        FAULT_STATE["server_error_status"] = payload.error_status_code
        FAULT_STATE["server_error_message"] = payload.error_message
    elif payload.fault_type == "none":
        FAULT_STATE = {
            "latency_delay_seconds": 0.0,
            "latency_probability": 0.0,
            "server_error_probability": 0.0,
            "server_error_status": 500,
            "server_error_message": "Injected chaos server error",
        }

    return {"status": "updated", "current_fault_state": FAULT_STATE}


@app.get(
    "/admin/fault-status",
    tags=["Admin"],
    dependencies=[Depends(verify_admin_key)],
)
def admin_fault_status() -> Dict[str, Any]:
    """Returns current runtime fault injection state."""
    return FAULT_STATE


@app.get("/metrics", tags=["Observability"])
def metrics() -> Response:
    """Scrapable Prometheus metrics endpoint."""
    content, content_type = get_metrics_payload()
    return Response(content=content, media_type=content_type)


@app.get("/incidents", tags=["Incidents"])
def list_incidents(
    status: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Returns tracked incidents with evidence."""
    incidents = IncidentManager.list_incidents(db=db, status=status, limit=limit)
    return [
        {
            "incident_id": inc.incident_id,
            "incident_type": inc.incident_type,
            "severity": inc.severity,
            "status": inc.status,
            "model_version": inc.model_version,
            "first_detected_at": inc.first_detected_at.isoformat(),
            "last_observed_at": inc.last_observed_at.isoformat(),
            "recommended_action": inc.recommended_action,
            "resolution_notes": inc.resolution_notes,
            "resolved_at": inc.resolved_at.isoformat() if inc.resolved_at else None,
            "evidence": inc.evidence,
        }
        for inc in incidents
    ]
