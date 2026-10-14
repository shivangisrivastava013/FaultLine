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
    assert data["model_version"] in ["champion_v1", "v1"]


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] in ["champion_v1", "v1"]
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
    assert data["model_version"] in ["champion_v1", "v1"]
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


def test_rollback_unauthorized_missing_key(client, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "test-secure-admin-token")
    resp = client.post("/rollback", json={"target_version": "champion_v1"})
    assert resp.status_code == 401
    assert "Missing X-API-Key" in resp.json()["detail"]


def test_rollback_unauthorized_invalid_key(client, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "test-secure-admin-token")
    resp = client.post(
        "/rollback",
        json={"target_version": "champion_v1"},
        headers={"X-API-Key": "wrong-token"},
    )
    assert resp.status_code == 403
    assert "Invalid administrative API key" in resp.json()["detail"]


def test_rollback_authorized_success(client, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "test-secure-admin-token")
    resp = client.post(
        "/rollback",
        json={"target_version": "champion_v1", "actor": "ci_test", "reason": "testing rollback auth"},
        headers={"X-API-Key": "test-secure-admin-token"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["active_version"] in ["champion_v1", "v1"]
    assert data["smoke_test_passed"] is True


def test_admin_activate_model_unauthorized(client, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "test-secure-admin-token")
    resp = client.post("/admin/activate-model", params={"version": "candidate_degraded_v2"})
    assert resp.status_code == 401


def test_admin_inject_fault_unauthorized(client, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "test-secure-admin-token")
    resp = client.post("/admin/inject-fault", json={"fault_type": "latency_spike"})
    assert resp.status_code == 401


def test_admin_endpoints_authorized_success(client, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "test-secure-admin-token")
    headers = {"X-API-Key": "test-secure-admin-token"}

    # Inject fault authorized
    resp = client.post(
        "/admin/inject-fault",
        json={"fault_type": "latency_spike", "delay_seconds": 0.2, "probability": 0.5},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["current_fault_state"]["latency_delay_seconds"] == 0.2

    # Check status authorized
    status_resp = client.get("/admin/fault-status", headers=headers)
    assert status_resp.status_code == 200
    assert status_resp.json()["latency_delay_seconds"] == 0.2

    # Reset fault
    reset_resp = client.post("/admin/inject-fault", json={"fault_type": "none"}, headers=headers)
    assert reset_resp.status_code == 200
    assert reset_resp.json()["current_fault_state"]["latency_delay_seconds"] == 0.0
