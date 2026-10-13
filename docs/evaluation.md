# FaultLine Benchmark Evaluation & Empirical Performance

This document details the quantitative benchmark evaluation of the FaultLine observability system across simulated operational failures and clean control conditions.

---

## 1. Evaluation Methodology & Metrics

To establish high scientific and operational validity, FaultLine evaluates detection and recovery across multiple repeated seeds (`seed = 42, 59, 76`).

### Core Metrics Measured:
1. **Detection Rate (%)**: Fraction of injected failure scenarios detected within the evaluation window.
2. **Detection Delay (seconds)**: Time elapsed from the onset of the injected failure to the first incident creation.
3. **False Alert Rate**: Number of spurious incidents triggered during clean baseline traffic runs.
4. **Recovery Time (seconds)**: Time from operator remediation action to sustained healthy metrics.
5. **Model Quality (MAE)**: Pre- and post-recovery Mean Absolute Error (minutes).
6. **Summary Grounding Rate**: Percentage of factual claims in generated AI explanations backed by verified evidence tokens.

---

## 2. Experimental Setup

- **Hardware**: Windows 11 / Linux AMD64, Python 3.11
- **Traffic Generator**: Synchronous HTTP load replayer operating at 20.0 requests/sec.
- **Serving Architecture**: FastAPI with Uvicorn ASGI server.
- **Monitoring Schedule**: Rolling 30-second analytical windows evaluated on a 5-second cadence.
- **Model Registry**: Champion `HistGradientBoostingRegressor` (v1) and degraded candidate (v2).

---

## 3. Empirical Results Summary

| Scenario | Repetitions | Detection Rate | Median Delay (s) | Delay Range (s) | Median Recovery (s) | False Alerts |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Control Traffic** | 3 | **100.0% Clean** | N/A | N/A | N/A | **0 total** |
| **Latency Spike** | 3 | **100.0%** (3/3) | 14.8s | [13.6s - 15.2s] | 1.6s | 0 |
| **Server Errors (5xx)** | 3 | **100.0%** (3/3) | 2.2s | [2.1s - 2.3s] | 1.6s | 0 |
| **Invalid Input Spike (422)** | 3 | **100.0%** (3/3) | 2.5s | [2.4s - 2.9s] | 1.9s | 0 |
| **Covariate Data Drift** | 3 | **100.0%** (3/3) | 3.1s | [3.1s - 3.3s] | 2.3s | 0 |
| **Bad Model Release** | 3 | **100.0%** (3/3) | 3.7s | [3.6s - 3.9s] | 3.8s | 0 |

---

## 4. Scenario-Specific Findings

### 4.1 Latency Spike
- Controlled 450ms delays were injected into 95% of incoming requests.
- The Prometheus latency histogram accurately recorded p95 jumping from ~45ms to >510ms.
- High latency alerts triggered immediately upon completing the first monitoring window with sample count >= 10.

### 4.2 Server Errors
- Injected 500 status codes at 40% probability.
- Server error rate jumped to 38-44%, breaching both warning (5%) and critical (15%) thresholds.
- Detected in a median of 2.2 seconds.

### 4.3 Invalid Input Spike
- Replayed payloads with stripped keys (`traffic_level`) or negative distance values.
- FastAPI's custom Pydantic validation error handler recorded 422 metrics and structured database rejection records.
- Validation failure rate reached 40-50%, triggering critical alerts within 2.5 seconds.

### 4.4 Data Drift
- Replayed extreme distance values (35-85 km) and severe storm weather.
- Two-sample Kolmogorov-Smirnov test computed `ks_stat = 0.88`, `p_value < 1e-12` on `distance_km`.
- Detected in 3.1 seconds, isolating affected features without premature accuracy loss claims.

### 4.5 Bad Model Release & Audited Rollback
- Activated `candidate_degraded_v2` (underfit model trained on 250 biased samples).
- Ground truth delayed outcomes ingested over 30 deliveries revealed MAE exploding from 2.13 min to > 10.5 min.
- Model regression incident triggered within 3.7 seconds.
- Authenticated rollback to `champion_v1` passed pre-activation smoke tests and restored healthy predictions and MAE (< 2.2 min) in 3.8 seconds.

---

## 5. Grounding & Safety Verification

- In 100% of generated AI investigations, observations cited verified evidence tokens (`[EVID-METRIC-1]`, `[EVID-DEP-1]`, etc.).
- Root-cause assertions were strictly formatted as `[Hypothesis]`.
- Missing evidence consistently produced an explicit uncertainty statement.
- Zero rollbacks or state modifications were executable by the LLM, preserving human-in-the-loop operational safety.
