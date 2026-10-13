"""
Pydantic Schemas for FaultLine Prediction Service.
"""

from __future__ import annotations

import uuid
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


TrafficLevel = Literal["low", "medium", "high", "jam"]
WeatherCondition = Literal["clear", "rain", "storm", "snow"]


class PredictionRequest(BaseModel):
    """Payload for delivery time prediction."""
    distance_km: float = Field(
        ...,
        gt=0.0,
        le=100.0,
        description="Delivery distance in kilometers (0 < d <= 100)",
        examples=[8.2],
    )
    traffic_level: TrafficLevel = Field(
        ...,
        description="Categorical traffic condition: low, medium, high, jam",
        examples=["high"],
    )
    weather: WeatherCondition = Field(
        ...,
        description="Categorical weather condition: clear, rain, storm, snow",
        examples=["rain"],
    )
    hour_of_day: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of order placement (0 to 23)",
        examples=[17],
    )
    restaurant_load: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Kitchen load factor from 0.0 (empty) to 1.0 (max capacity)",
        examples=[0.8],
    )

    request_id: Optional[str] = Field(
        default=None,
        description="Optional client request ID; generated automatically if omitted",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "distance_km": 8.2,
                "traffic_level": "high",
                "weather": "rain",
                "hour_of_day": 17,
                "restaurant_load": 0.8,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response returned by /predict endpoint."""
    request_id: str
    predicted_minutes: float
    model_version: str
    latency_ms: float
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Metadata response returned by /model-info."""
    model_version: str
    model_type: str
    status: str
    features: list[str]
    artifact_path: str
    val_mae: Optional[float] = None
    test_mae: Optional[float] = None
    active_since: str
    additional_metadata: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    """Process liveness response returned by /health."""
    status: str = "ok"
    service: str = "faultline-prediction-api"
    uptime_seconds: float


class ReadyResponse(BaseModel):
    """Readiness response returned by /ready."""
    status: str = "ready"
    model_loaded: bool
    model_version: str
    database_connected: bool


class OutcomeRequest(BaseModel):
    """Payload for delayed actual outcome submission."""
    request_id: str = Field(..., description="Unique request ID from /predict")
    actual_delivery_minutes: float = Field(
        ...,
        gt=0.0,
        le=300.0,
        description="Observed ground truth delivery duration in minutes",
    )


class OutcomeResponse(BaseModel):
    """Confirmation response for outcome recording."""
    status: str = "recorded"
    request_id: str
    actual_delivery_minutes: float
    error_delta_minutes: Optional[float] = None


class RollbackRequest(BaseModel):
    """Payload to trigger model rollback."""
    target_version: Optional[str] = Field(
        default="v1",
        description="Target model version or 'champion' / 'previous'",
    )
    actor: str = Field(..., description="Operator or automated agent triggering rollback")
    reason: str = Field(..., description="Justification or incident ID triggering rollback")


class RollbackResponse(BaseModel):
    """Result of rollback action."""
    success: bool
    previous_version: str
    active_version: str
    actor: str
    reason: str
    smoke_test_passed: bool
    message: str


class FaultInjectionRequest(BaseModel):
    """Payload to simulate live operational faults."""
    fault_type: Literal["latency_spike", "server_error", "corrupt_data", "none"]
    probability: float = Field(default=1.0, ge=0.0, le=1.0)
    delay_seconds: float = Field(default=0.8, ge=0.0, le=10.0)
    error_status_code: int = Field(default=500, ge=400, le=599)
    error_message: str = Field(default="Injected chaos error for testing")
