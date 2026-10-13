# FaultLine Limitations & Architectural Boundaries

This document transparently documents system limitations, engineering trade-offs, and future production roadmap considerations.

---

## 1. Single Replica Serving Model State
- **Current State**: Active model version switching and rollbacks occur in-memory within a single FastAPI process using thread locks.
- **Limitation**: In a multi-replica distributed deployment (e.g. Kubernetes with HPA or AWS ECS with 10 pods), an in-memory rollback on one instance does not automatically propagate to other instances.
- **Production Roadmap**: Implement a distributed consensus or configuration coordinator (e.g., Redis Pub/Sub, etcd, or Consul key-value watches) to broadcast model version activation events across all serving pods simultaneously.

---

## 2. Delayed Label Synchronization
- **Current State**: Outcomes are matched to predictions via direct SQL join on `request_id`.
- **Limitation**: In extremely high-throughput environments (> 100,000 req/sec), synchronous SQL joins on high-cardinality transactional tables can introduce database contention.
- **Production Roadmap**: Transition historical matching to an analytical lakehouse (e.g. ClickHouse, BigQuery, or Apache Iceberg) or partitioned streaming joins with Apache Flink.

---

## 3. Window Sizing vs. Detection Delay Trade-Off
- **Current State**: Operational rules use 60-second windows with minimum sample thresholds (e.g. 10 requests) to prevent false alerts during low-volume hours.
- **Limitation**: A sudden catastrophic spike during near-zero traffic hours takes longer to accumulate the required minimum sample count before alerting.
- **Trade-Off Decision**: We deliberately prioritized **zero false alerts** over sub-second alert triggers on low sample counts to prevent alert fatigue.

---

## 4. LLM Role Boundaries
- **Current State**: Grounded AI investigations are strictly diagnostic and advisory.
- **Safety Boundary**: The AI cannot independently execute rollbacks, mutate models, or alter database records. Rollbacks require explicit human operator confirmation with audited credentials.

---

## 5. Statistical Drift Scope
- **Current State**: Continuous features are evaluated using two-sample Kolmogorov-Smirnov (KS) tests; discrete features are evaluated using Population Stability Index (PSI).
- **Limitation**: KS tests evaluate univariate marginal distributions independently and do not capture multi-dimensional joint feature covariance changes.
- **Production Roadmap**: Add multivariate drift detection (e.g. Maximum Mean Discrepancy / MMD or autoencoder reconstruction error) for high-dimensional feature spaces.
