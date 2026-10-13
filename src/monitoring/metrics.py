"""
Prometheus Metrics Instrumentation for FaultLine.

Keeps cardinality bounded by avoiding request IDs in metric labels.
"""

from __future__ import annotations

import time
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


# 1. Request Counter
REQUESTS_TOTAL = Counter(
    "faultline_http_requests_total",
    "Total HTTP requests received by the prediction service",
    ["method", "endpoint", "status_code"],
)

# 2. Request Duration Histogram
REQUEST_DURATION_SECONDS = Histogram(
    "faultline_http_request_duration_seconds",
    "HTTP request latency distribution in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# 3. Server Errors Counter
SERVER_ERRORS_TOTAL = Counter(
    "faultline_server_errors_total",
    "Total internal server errors (5xx) encountered",
    ["endpoint", "error_type"],
)

# 4. Validation Failures Counter
VALIDATION_FAILURES_TOTAL = Counter(
    "faultline_validation_failures_total",
    "Total 422 schema or data validation rejections",
    ["endpoint", "feature_failed"],
)

# 5. Model Loading Failures Counter
MODEL_LOAD_FAILURES_TOTAL = Counter(
    "faultline_model_load_failures_total",
    "Total failures encountered when attempting to load a model artifact",
    ["model_version"],
)

# 6. Active Model Version Gauge (numeric timestamp or version identifier)
ACTIVE_MODEL_INFO = Gauge(
    "faultline_active_model_info",
    "Information about currently active model version",
    ["model_version", "model_type"],
)

# 7. Predictions & Outcomes Counters
PREDICTIONS_TOTAL = Counter(
    "faultline_predictions_total",
    "Total successful predictions generated",
    ["model_version"],
)

OUTCOMES_TOTAL = Counter(
    "faultline_outcomes_total",
    "Total delayed delivery outcomes ingested",
    ["model_version"],
)

PREDICTION_ERROR_MAE_GAUGE = Gauge(
    "faultline_window_mae",
    "Mean Absolute Error observed in current monitoring window",
    ["model_version"],
)


def get_metrics_payload() -> tuple[bytes, str]:
    """Generates Prometheus scrapable payload and content type."""
    return generate_latest(), CONTENT_TYPE_LATEST
