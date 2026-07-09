# SentinelML 🛡️ — Enterprise-Grade Automated Model Observability & Drift Engine

SentinelML is a high-performance, asynchronous Model Observability system designed to monitor real-time ML inference, ingest ground-truth labels for rolling performance tracking (Accuracy, F1-score), evaluate statistical covariate drift (Kolmogorov-Smirnov & Chi-Square tests), and orchestrate analytical dashboards. 

By decoupling inference execution from database writes and heavy analytical workloads, SentinelML guarantees sub-10ms response latencies.

---

## 🛠️ System Architecture

```
                                 ┌────────────────────────┐
                                 │  Client POST /predict  │
                                 └───────────┬────────────┘
                                             │
                                             ▼ (Execute Prediction < 5ms)
                                 ┌────────────────────────┐
                                 │   FastAPI Gateway API  │
                                 └─────┬────────────┬─────┘
                                       │            │
                      (Log Async Task) │            │ (Inference Output)
                                       ▼            ▼
                            ┌─────────────────┐   ┌────────────────────┐
                            │Background Queue │   │ XGBoost Classifier │
                            └────────┬────────┘   └────────────────────┘
                                     │
                                     ▼ (Write Telemetry)
                            ┌─────────────────┐
                            │ SQLite Database │ <─── (Post labels) ─── Feedback Loop API
                            └────────┬────────┘
                                     │
                         (Sliding Window Queries)
                                     ▼
                            ┌─────────────────┐
                            │ Drift Evaluator │ ──(Generate HTML)──> Data Drift Reports
                            └────────┬────────┘
                                     │
                                     │ (Export Stats)
                                     ▼
                            ┌──────────────────┐
                            │ Prometheus Agent │ <────(Scrape)─────  Prometheus Server
                            └──────────────────┘                            │
                                                                            │ (Visualize)
                                                                            ▼
                                                                     Grafana Dashboards
```

---

## 🌟 Key Capabilities (Elite Edition)

* **High-Performance Caching Layer**: Utilizes an in-memory caching mechanism with a 30-second Time-To-Live (TTL) for analytical summaries, keeping response latencies of dashboard-data queries under **5ms** even during concurrent client requests.
* **Inline Data Quality & Outlier Auditing**: Performs statistical boundary checking at startup on reference datasets (using Q1/Q3 Interquartile Ranges and unique categories). Flagged anomalies (e.g. out-of-bound variables, unrecognized classes) and auto-imputed columns are saved in SQLite and returned directly to API clients.
* **Dynamic Distribution Matching**: Pre-calculates 10-bin histograms (numerical) and categorical frequencies (grouped top percentages) for live comparison against the training baselines, fully plotted on interactive Chart.js widgets.
* **Chronological Performance Tracking**: Groups ground-truth labels chronologically to calculate and plot rolling Accuracy, Precision, Recall, and F1-score progression over time.
* **Model-Agnostic Schema-Driven Imputation**: Dynamically loads baseline features, data schemas, and imputation defaults from `model_config.yaml` to ensure clean recovery from missing features.
* **Interactive Glassmorphic Portal**: Located at `/dashboard` — built with tabbed sidebar layout, live Feature Drift matrices, data quality charts, evidently report manager, and a paginated, filtered SQL log stream for direct feedback injection.

---

## 📁 Project Structure

```
SentinelML-Automated-Model-Observability/
│
├── api/
│   ├── app.py              # FastAPI Main Application & Telemetry Endpoints
│   ├── database.py         # SQLAlchemy Engine & SQLite Schema Auto-Migrations
│   ├── schemas.py          # Pydantic Input/Output Schemas
│   ├── templates/
│   │   └── dashboard.html  # Premium Glassmorphic Dark UI Portal
│   └── requirements.txt    # Application dependencies
│
├── data/
│   ├── raw/                # Demographics Dataset (Adult Census)
│   ├── reference/          # Reference training distribution metrics
│   ├── live/               # SQLite database file
│   ├── drift_reports/      # Interactive HTML drift files
│   └── model_config.yaml   # Central schema config and baseline defaults
│
├── model/
│   └── trained_model.pkl   # Serialized ML Pipeline (XGBoost/Sklearn)
│
├── monitoring/
│   ├── train_model.py      # XGBoost Model & statistics training pipeline
│   ├── evidently_runner.py # Bounded Evidently AI drift report builder
│   ├── scheduler.py        # Drift Report cron trigger daemon
│   └── metrics_exporter.py # Prometheus client metrics exporter (Chi-Sq & KS tests)
│
├── prometheus/
│   └── prometheus.yml      # Scraping configurations
│
├── grafana/
│   ├── provisioning/       # Automated Datasource & Dashboard provisioning
│   └── sentinelml_dashboard.json # Grafana JSON layout with categorical drift timeseries
│
├── tests/
│   └── test_api.py         # Pytest API, Async, & Performance Verification Suite
│
├── docker-compose.yml      # Orchestrates all service containers
│   └── Dockerfile          # Container image builder
```

---

## 🚀 Quickstart

### Option A: Bare-Metal Local Launch

1. **Install Dependencies**:
   ```bash
   pip install -r api/requirements.txt
   ```
2. **Train Model & Compute Reference Statistics**:
   ```bash
   python monitoring/train_model.py
   ```
3. **Start API Gateway Server**:
   ```bash
   python -m uvicorn api.app:app --host 0.0.0.0 --port 5000
   ```
4. **Boot Prometheus Exporter Daemon**:
   ```bash
   python monitoring/metrics_exporter.py
   ```
5. **Boot Periodic Drift Scheduler Daemon**:
   ```bash
   python monitoring/scheduler.py
   ```
6. **Open Dashboard**:
   Navigate to `http://localhost:5000/dashboard` in your browser.

---

### Option B: Using Docker Compose

Launch the entire monitoring stack (FastAPI Gateway, SQLite, Prometheus Exporter, Scheduler Daemon, Prometheus Server, and Grafana Console):
```bash
docker-compose up --build
```
* **Inference Portal**: `http://localhost:5000/dashboard`
* **Inference Gateway API**: `http://localhost:5000`
* **Prometheus Server**: `http://localhost:9090`
* **Grafana Console**: `http://localhost:3000` (Default credentials: `admin`/`admin`)

---

## ⚡ API Specifications

### 1. Inference Request: `POST /predict`
Runs inference on demographic features. Fields left unprovided are auto-imputed from reference data baseline configurations.

* **Sample Command**:
  ```bash
  curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"age": 120, "education": "Bachelors", "hours-per-week": 40}'
  ```

* **Sample Response (with outlier flag and auto-imputation audit)**:
  ```json
  {
    "prediction": 0,
    "probability": 0.187421946,
    "id": "7649bbba-29ac-40d3-b26a-9e12bf48ab79",
    "status": "success",
    "is_anomaly": true,
    "anomaly_reasons": ["age: 120 is an outlier (expected range: [-4.5, 87.5])"],
    "imputed_features": ["workclass", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "native-country"]
  }
  ```

### 2. Submit Feedback: `POST /api/feedback`
Submits a ground-truth label for a past inference record to update model performance metrics (Accuracy, F1-score) dynamically.

* **Sample Command**:
  ```bash
  curl -X POST http://localhost:5000/api/feedback \
    -H "Content-Type: application/json" \
    -d '{"id": "7649bbba-29ac-40d3-b26a-9e12bf48ab79", "feedback_label": 0}'
  ```

### 3. Fetch Dashboard Summary Data: `GET /api/dashboard-data`
Returns cached drift analysis p-values, feature distribution frequencies, rolling performance stats, generated Evidently report locations, and paginated/filtered telemetry logs.

* **Query Parameters (Optional)**:
  * `page`: page number index (default: `1`)
  * `page_size`: records per page (default: `15`)
  * `search_id`: substring search on inference IDs
  * `filter_prediction`: `0` or `1` classification label
  * `filter_feedback`: `-1` (unlabeled), `0` or `1` ground-truth labels

---

## 🧪 Testing

Run the test suite to verify schema structures, database connection pools, async telemetry tasks, performance timeline aggregation, and inline anomaly calculations:
```bash
python -m pytest -vv
```

---

## 👤 Author

* **beingaditya18** (Aditya Mandloi)
