import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "SentinelML API running"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "imputation_features" in data

def test_predict_full():
    payload = {
        "age": 39,
        "workclass": "Private",
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "id" in data
    assert data["status"] == "success"
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0

def test_predict_partial_imputation():
    # Sending only required inputs, others should be filled from training statistics
    payload = {
        "age": 30,
        "education": "Masters",
        "hours-per-week": 45
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["status"] == "success"

def test_predict_invalid_missing_required():
    # Missing required field 'age'
    payload = {
        "education": "Masters",
        "hours-per-week": 45
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error

def test_predict_invalid_datatype():
    # 'age' should be integer, sending string
    payload = {
        "age": "thirty",
        "hours-per-week": 40
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_dashboard():
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "SentinelML" in response.text
    assert "Prediction Playground" in response.text

def test_dashboard_data():
    response = client.get("/api/dashboard-data")
    assert response.status_code == 200
    data = response.json()
    assert "system_healthy" in data
    assert data["system_healthy"] is True
    assert "total_inferences" in data
    assert "drift_metrics" in data
    assert "recent_inferences" in data
    assert "reports" in data

def test_trigger_drift_api():
    response = client.post("/api/trigger-drift")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "report" in data

def test_feedback_submission():
    # 1. Run a prediction to create a record in DB
    payload = {
        "age": 45,
        "hours-per-week": 40,
        "education": "Bachelors"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    inf_id = res_data["id"]

    # 2. Submit feedback for that inference record
    feedback_payload = {
        "id": inf_id,
        "feedback_label": 1
    }
    response_fb = client.post("/api/feedback", json=feedback_payload)
    assert response_fb.status_code == 200
    fb_data = response_fb.json()
    assert fb_data["status"] == "success"
    assert fb_data["id"] == inf_id
    assert fb_data["feedback_label"] == 1

    # 3. Check that /api/dashboard-data includes this feedback score recalculation
    response_db = client.get("/api/dashboard-data")
    assert response_db.status_code == 200
    db_data = response_db.json()
    assert "performance_metrics" in db_data
    assert db_data["performance_metrics"]["total_labeled"] >= 1

def test_predict_anomalies_and_imputations():
    # Test record with imputed fields
    payload_impute = {
        "age": 35,
        "hours-per-week": 40
    }
    response = client.post("/predict", json=payload_impute)
    assert response.status_code == 200
    data = response.json()
    assert data["is_anomaly"] is False
    assert data["imputed_features"] is not None
    assert "workclass" in data["imputed_features"]
    assert "education" in data["imputed_features"]

    # Test record with out-of-bounds anomaly (age = 150)
    payload_anomaly = {
        "age": 150,
        "hours-per-week": 40,
        "workclass": "Private",
        "education": "Bachelors"
    }
    response_anom = client.post("/predict", json=payload_anomaly)
    assert response_anom.status_code == 200
    data_anom = response_anom.json()
    assert data_anom["is_anomaly"] is True
    assert any("age" in r for r in data_anom["anomaly_reasons"])
    
    # Test record with unseen category (workclass = "Space-work")
    payload_unseen = {
        "age": 30,
        "hours-per-week": 40,
        "workclass": "Space-work",
        "education": "Bachelors"
    }
    response_unseen = client.post("/predict", json=payload_unseen)
    assert response_unseen.status_code == 200
    data_unseen = response_unseen.json()
    assert data_unseen["is_anomaly"] is True
    assert any("workclass" in r for r in data_unseen["anomaly_reasons"])
