# FaultLine Benchmark Evaluation Report

**Generated:** 2026-10-13 19:40:39 UTC  
**Total Experiment Runs:** 18  

## Summary Metrics by Scenario

| Scenario | Runs | Detection Rate | Median Delay (s) | Delay Range (s) | Median Recovery (s) | False Alerts |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **clean_traffic** (Control) | 3 | 100.0% Clean | N/A | N/A | N/A | 0 total |
| **latency_spike** | 3 | **100.0%** (3/3) | 15.06 | [14.74 - 15.19] | 1.58 | 0 |
| **server_errors** | 3 | **100.0%** (3/3) | 2.25 | [2.16 - 2.28] | 1.54 | 0 |
| **invalid_input_spike** | 3 | **66.7%** (2/3) | 2.62 | [2.61 - 2.62] | 1.92 | 0 |
| **data_drift** | 3 | **100.0%** (3/3) | 2.98 | [2.85 - 3.06] | 2.31 | 0 |
| **bad_model_release** | 3 | **100.0%** (3/3) | 3.59 | [3.58 - 3.75] | 3.70 | 0 |

## Key Observations & Model Quality Findings

- **Clean Baseline Stability:** Zero false alerts observed during control runs.
- **Service Faults (Latency & Errors):** Rapidly caught in the first window post-onset.
- **Covariate Drift:** Correctly isolated shifted features (`distance_km`, `traffic_level`) via 2-sample KS test without premature accuracy claims.
- **Model Regression & Rollback:** The degraded candidate model degraded MAE from ~2.1 min to >10.5 min. Rollback to `champion_v1` passed pre-activation smoke tests and restored healthy latency and prediction accuracy (post-rollback MAE: 1.82–1.91 min).

## Grounding & Safety Verification

- **Audit Trail:** 100% of model rollbacks recorded operator actor, justification, and pre-activation validation in `deployment_events`.
- **Citation Enforcement:** Investigation observations strictly cited evidence tokens (`[EVID-METRIC-1]`, `[EVID-DEP-1]`).