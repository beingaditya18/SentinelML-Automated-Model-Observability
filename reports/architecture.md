# SentinelML — System Architecture

## 1. High-Level Architecture Overview

SentinelML is designed as a **production-grade ML observability pipeline** that cleanly separates **model inference** from **monitoring and drift detection**. This decoupled architecture ensures that observability workloads never impact real-time prediction latency.

### Architectural Flow

1. Incoming requests hit the Flask Inference API
2. Predictions are logged in real time
3. Reference vs Live data is continuously monitored
4. Evidently AI generates statistical drift reports
5. Prometheus exports drift & health metrics
6. Grafana visualizes system behavior and drift trends

```
Client
  │
  ▼
Flask Inference API
  │
  ▼
Live Inference Logs (CSV)
  │
  ▼
Evidently Drift Engine
  │
  ▼
HTML Drift Reports
  │
  ├── Prometheus Metrics
  └── Grafana Dashboards
```

---

## 2. System Design Philosophy

SentinelML follows **monitoring-first ML system design**, inspired by real-world enterprise MLOps platforms.

### Key Design Principles

* **Decoupled inference & monitoring** — zero impact on API latency
* **Asynchronous drift detection** — batch-based statistical analysis
* **Immutable inference logs** — reproducibility & auditability
* **Metrics-driven observability** — monitoring before failure

This design enables early detection of **silent model failures**, where models continue running but gradually become unreliable due to data drift.

---

## 3. Component Breakdown

### A. Inference Layer — Flask Gateway

**Role**
Synchronous entry point for model consumers.

**Responsibilities**

* Accept JSON-based prediction requests
* Perform feature validation
* Execute model inference using scikit-learn
* Return prediction and confidence scores

**Design Pattern**
Interceptor-based logging pattern.

Each request:

* Is timestamped
* Assigned a unique UUID
* Logged before response delivery

This ensures **end-to-end traceability** of every prediction.

---

### B. Production Inference Logging Layer

**Role**
Persistent storage of real-time prediction data.

**Implementation**

* Append-only CSV logs (`data/live/live_predictions.csv`)
* Mimics production-grade inference storage systems

**Why CSV?**

* Simple, transparent, and reproducible
* Easily replaceable with databases, data lakes, or feature stores

This layer acts as the **single source of truth** for monitoring and drift analysis.

---

### C. Observability Engine — Evidently AI

**Role**
Asynchronous statistical profiling and drift detection.

**Core Logic**

* Compares **Baseline (Reference)** data vs **Production (Live)** data
* Computes feature-level and dataset-level drift metrics
* Identifies distributional changes over time

**Key Capability**
Detection of **silent failures** — scenarios where:

* The model still runs
* No errors are thrown
* But input data has shifted significantly

**Output Artifacts**

* Interactive HTML drift reports
* Feature distribution comparisons
* Drift severity indicators

---

### D. Automation & Scheduling Layer

**Component**
`scheduler.py`

**Role**
Automates drift detection at fixed intervals.

**Behavior**

* Triggers `evidently_runner.py` every *X* minutes
* Enables continuous monitoring without manual intervention

This simulates **real production monitoring jobs** used in enterprise ML systems.

---

### E. Metrics Pipeline — Prometheus & Grafana

**Role**
Real-time system health and drift visibility.

**Metrics Flow**

1. `evidently_runner.py` calculates drift scores
2. `metrics_exporter.py` converts drift scores into Prometheus gauges
3. Prometheus scrapes exposed metrics
4. Grafana queries Prometheus for visualization

**Tracked Metrics**

* Drift score magnitude
* Drift detection timestamps
* Inference volume
* System health indicators

**Alerting Logic**
If:

```
Drift Score > 0.1
```

Then:

* Prometheus alert is triggered
* System flags potential model reliability risk

---

## 4. End-to-End Data Flow

1. **Request**
   Client sends JSON payload to `/predict`

2. **Inference**
   Flask API processes request using trained RandomForest model

3. **Logging**
   Request–response pair appended to live inference logs

4. **Analysis**
   Scheduler triggers Evidently drift analysis periodically

5. **Reporting**
   HTML drift reports generated

6. **Monitoring**
   Metrics exported → Prometheus → Grafana dashboards

---

## 5. Architectural Value

This architecture demonstrates how **real-world ML systems are engineered beyond notebooks** by focusing on:

* Model reliability
* Observability & monitoring
* Early drift detection
* Responsible & trustworthy AI

SentinelML is not a demo system — it is a **blueprint for production ML observability**.

---

**Author:** Aditya Mandloi
**Domain:** AI, MLOps, Model Observability & Reliability Engineering

