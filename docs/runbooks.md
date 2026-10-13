# FaultLine Operational Runbooks

This document provides step-by-step diagnostic and recovery runbooks for all failure modes detected by FaultLine.

---

## 1. High Latency Incident (`high_latency`)

### Trigger Thresholds
- **Warning**: p95 request latency >= 150 ms (min 10 samples)
- **Critical**: p95 request latency >= 400 ms

### Diagnostic Procedure
1. Check Prometheus histogram `faultline_http_request_duration_seconds` to verify whether latency is uniform across all endpoints or specific to `/predict`.
2. Review recent deployments in the Dashboard **Model Rollback & Audit** tab. Check if a new model version was recently activated (`[EVID-DEP-1]`).
3. Check CPU and memory utilization on the host or API container.
4. Verify if synthetic chaos delay was injected via `/admin/fault-status`.

### Recovery Actions
- If caused by chaos injection: send `POST /admin/inject-fault` with `{"fault_type": "none"}` or click **Reset Clean** in the dashboard.
- If caused by a newly deployed candidate model: execute model rollback to `champion_v1`.
- If caused by container resource starvation: increase replica count or container CPU quota.

---

## 2. Server Errors Spike (`server_errors`)

### Trigger Thresholds
- **Warning**: 5xx HTTP error rate >= 5.0%
- **Critical**: 5xx HTTP error rate >= 15.0%

### Diagnostic Procedure
1. Inspect API logs (`docker logs faultline-api`) for unhandled tracebacks or database disconnection errors.
2. Verify database health:
   ```bash
   docker exec -it faultline-postgres pg_isready -U faultline -d faultline_db
   ```
3. Inspect `sample_requests` in the incident evidence bundle for common failure patterns.

### Recovery Actions
- If chaos error injection is active: disable fault injection.
- If database pool is exhausted: recycle API workers or scale pool limit in `DATABASE_URL`.
- If an unhandled edge case exists in preprocessing: rollback model and patch input validator.

---

## 3. Schema Validation Spike (`validation_failures`)

### Trigger Thresholds
- **Warning**: 422 Unprocessable Content rate >= 10.0%
- **Critical**: 422 Unprocessable Content rate >= 25.0%

### Diagnostic Procedure
1. Inspect rejected request payloads in `prediction_requests` table where `status_code == 422`.
2. Check `VALIDATION_FAILURES_TOTAL` Prometheus metric labels for `feature_failed`.
3. Identify upstream client service emitting missing keys or invalid categorical values.

### Recovery Actions
- Notify client application engineers with sample rejected payloads.
- If client deployment is intentional (e.g. new weather condition or feature deprecation): update API Pydantic schemas in `src/api/schemas.py`.

---

## 4. Covariate Feature Drift (`data_drift`)

### Trigger Thresholds
- **Warning**: Two-sample Kolmogorov-Smirnov test p-value < 0.01 or Population Stability Index (PSI) >= 0.25 (min 25 samples).

### Diagnostic Procedure
1. Review the **Feature Drift & Data Quality** dashboard tab. Compare incoming distributions against `data/reference_baseline.parquet`.
2. Identify drifted features (e.g. `distance_km` shifted toward extreme long-hauls, or `weather` skewed toward storms).
3. **Important Distinction**: Feature drift establishes that the input distribution has changed; it does **not** prove that prediction accuracy has deteriorated. Wait for ground-truth delivery outcomes to confirm whether accuracy has dropped.

### Recovery Actions
- Alert data science team to evaluate if retraining on the new distribution is warranted.
- Verify whether an operational change occurred (e.g. new market launched with different geography).

---

## 5. Model Accuracy Regression (`model_regression`)

### Trigger Thresholds
- **Critical**: Window Mean Absolute Error (MAE) >= 5.5 minutes over >= 12 completed outcomes.

### Diagnostic Procedure
1. Verify the currently active model version (`GET /model-info`).
2. Review the slice-level MAE breakdown in the evidence bundle across traffic categories (`low`, `medium`, `high`, `jam`).
3. Check `deployment_events` for recent model activations.

### Recovery Actions
1. Open the **Model Rollback & Audit** tab in the dashboard.
2. Select target version `champion_v1`.
3. Enter operator credentials and incident reason.
4. Click **Execute Authenticated Rollback**.
5. FaultLine will automatically execute a pre-activation smoke test before swapping the active model, record an audit event, and restore healthy predictions.
