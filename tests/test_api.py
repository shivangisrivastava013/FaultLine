"""
Test suite for FaultLine Prediction API.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.incidents.database import init_db
from src.models.rollback_manager import model_manager


@pytest.fixture(scope="module", autouse=True)
def setup_api():
    init_db()
    model_manager.initialize_active_model()


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data


def test_ready_endpoint(client):
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert data["model_version"] == "v1"


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "v1"
    assert data["status"] in ["champion", "active"]
    assert "distance_km" in data["features"]


def test_predict_success(client):
    payload = {
        "distance_km": 8.2,
        "traffic_level": "high",
        "weather": "rain",
        "hour_of_day": 17,
        "restaurant_load": 0.8,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert "predicted_minutes" in data
    assert data["model_version"] == "v1"
    assert data["predicted_minutes"] > 15.0  # reasonable delivery duration
    assert data["latency_ms"] >= 0.0


def test_predict_validation_error(client):
    # Invalid distance (negative) and invalid traffic level
    payload = {
        "distance_km": -5.0,
        "traffic_level": "warp_speed",
        "weather": "rain",
        "hour_of_day": 17,
        "restaurant_load": 0.8,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_outcome_recording(client):
    # First make a prediction
    pred_res = client.post("/predict", json={
        "distance_km": 4.5,
        "traffic_level": "low",
        "weather": "clear",
        "hour_of_day": 14,
        "restaurant_load": 0.3,
    })
    assert pred_res.status_code == 200
    req_id = pred_res.json()["request_id"]

    # Record outcome
    outcome_res = client.post("/outcomes", json={
        "request_id": req_id,
        "actual_delivery_minutes": 22.5,
    })
    assert outcome_res.status_code == 200
    odata = outcome_res.json()
    assert odata["status"] == "recorded"
    assert odata["request_id"] == req_id
    assert odata["actual_delivery_minutes"] == 22.5
    assert odata["error_delta_minutes"] is not None


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "faultline_http_requests_total" in response.text
