# 🚨 SentinelML — Automated Model Observability & Drift Detection

> **Production-grade ML observability system for detecting data drift, monitoring model health, and ensuring reliability of deployed machine learning models.**

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![MLOps](https://img.shields.io/badge/MLOps-Production--Ready-green)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📌 Overview

**SentinelML** is an end-to-end **Model Observability & Drift Detection system** built to simulate **real-world production ML workflows**.

It continuously monitors live inference data, compares it with reference data, and automatically generates **data drift reports** — enabling ML teams to detect silent model failures before they impact business decisions.

This project reflects **industry-grade MLOps practices** used in large-scale ML systems.

---

## 🎯 Key Features

✅ Real-time ML inference via REST API
✅ Automatic logging of live prediction data
✅ Data drift detection using statistical metrics
✅ Interactive drift reports (HTML)
✅ Scheduled monitoring jobs
✅ Prometheus-ready metrics
✅ Grafana dashboard support
✅ Clean, modular, production-ready architecture

---

## 🧠 Why This Project Matters

In production, **models fail silently** — not by crashing, but by **slowly drifting away from reality**.

SentinelML helps answer critical questions:

* Has incoming data changed?
* Is the model still reliable?
* Are predictions drifting over time?
* Can we detect issues *before* accuracy drops?

This project focuses on **model trust, reliability, and responsible AI** — not just accuracy.

---

## 🏗️ System Architecture

```
Client
  │
  ▼
Flask API (Inference)
  │
  ▼
Live Prediction Logs (CSV)
  │
  ▼
Evidently AI (Drift Detection)
  │
  ▼
HTML Drift Reports
  │
  ├── Prometheus Metrics
  └── Grafana Dashboards
```

---

## 🛠 Tech Stack

| Layer         | Technology                     |
| ------------- | ------------------------------ |
| API           | Flask                          |
| ML Model      | RandomForest (Scikit-learn)    |
| Data Drift    | Evidently AI                   |
| Monitoring    | Prometheus                     |
| Visualization | Grafana                        |
| Automation    | Python Scheduler               |
| Storage       | CSV (Production-style logging) |

---

## 📂 Project Structure

```
SentinelML-Automated-Model-Observability/
│
├── api/                    # Inference API
│   ├── app.py
│   ├── utils.py
│   └── requirements.txt
│
├── data/
│   ├── reference/          # Training / baseline data
│   ├── live/               # Real-time prediction logs
│   └── drift_reports/      # Generated drift reports
│
├── model/                  # Trained ML artifacts
│
├── monitoring/
│   ├── evidently_runner.py # Drift detection engine
│   ├── scheduler.py        # Automated monitoring
│   └── metrics_exporter.py # Prometheus metrics
│
├── grafana/
│   └── dashboards/
│
├── prometheus/
│   └── prometheus.yml
│
├── reports/
│   ├── architecture.md
│   └── technical_report.md
│
└── README.md
```

---

## 🚀 How It Works (Step-by-Step)

### 1️⃣ Model Inference

* Flask API receives prediction requests
* Model generates prediction + probability

### 2️⃣ Live Data Logging

* Each request is logged with timestamp & UUID
* Stored as **production inference data**

### 3️⃣ Drift Detection

* Live data compared with reference dataset
* Statistical drift metrics calculated
* HTML drift reports generated

### 4️⃣ Monitoring & Automation

* Scheduler runs drift detection periodically
* Prometheus exports monitoring metrics
* Grafana visualizes system health

---

## ▶️ Running the Project

### Install dependencies

```bash
pip install -r api/requirements.txt
```

### Train dummy model (for demo)

```bash
python model/train_dummy.py
```

### Start API

```bash
python api/app.py
```

### Send prediction

```bash
curl -X POST http://127.0.0.1:5000/predict \
 -H "Content-Type: application/json" \
 -d '{"age":39,"education":"Bachelors","hours_per_week":40}'
```

### Run drift detection

```bash
python monitoring/evidently_runner.py
```


## 📊 Sample Output

* **Live Predictions:** `data/live/live_predictions.csv`
* **Drift Report:** `data/drift_reports/drift_YYYYMMDD_HHMMSS.html`

Interactive HTML reports include:

* Feature distribution comparison
* Drift scores
* Statistical tests
* Data quality insights


## 🧪 Advanced Concepts Demonstrated

* Data Drift vs Concept Drift
* Production inference logging
* Model observability
* ML system reliability
* Monitoring-first ML design
* Responsible & trustworthy AI


## 💼 Use Cases

* Production ML monitoring
* Enterprise ML pipelines
* AI governance & compliance
* Model reliability engineering
* MLOps interviews & portfolios


## 👨‍💻 Author

**Aditya Mandloi**
AI & MLOps Enthusiast
🔗 GitHub: [beingaditya18](https://github.com/beingaditya18)


## ⭐ Final Note

> This project is intentionally designed to mirror **real production ML systems**, not toy demos.

If you’re a recruiter, engineer, or student exploring **modern MLOps**, SentinelML demonstrates how **robust ML systems are built, monitored, and trusted in the real world**.

⭐ **Star this repo if you find it useful!**


