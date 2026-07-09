import os
import uuid
import time
from datetime import datetime, timezone
import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from api.database import init_db, get_db, InferenceRecord
from api.schemas import PredictionRequest, PredictionResponse

MODEL_PATH = "model/trained_model.pkl"
REFERENCE_DATA_PATH = "data/reference/reference_data.csv"
CONFIG_PATH = "data/model_config.yaml"

app = FastAPI(title="SentinelML API", description="Automated Model Observability & Drift Detection API")

# In-memory caching for dashboard data to optimize UI refresh latencies
class DashboardCache:
    def __init__(self, ttl_seconds=30):
        self.ttl = ttl_seconds
        self.last_updated = 0
        self.data = None

    def get(self):
        now = time.time()
        if self.data is not None and (now - self.last_updated) < self.ttl:
            return self.data
        return None

    def set(self, data):
        self.data = data
        self.last_updated = time.time()

dashboard_cache = DashboardCache(ttl_seconds=30)

# Load model pipeline
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Please run training first.")
model = joblib.load(MODEL_PATH)

# Load reference statistics for robust imputation
IMPUTATION_DEFAULTS = {}
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            config_data = yaml.safe_load(f)
            for feat_type in ["numerical", "categorical"]:
                features_list = config_data.get("features", {}).get(feat_type, [])
                for feat in features_list:
                    IMPUTATION_DEFAULTS[feat["name"]] = feat["impute_default"]
    except Exception as e:
        print(f"Failed to load YAML config: {e}")
        
if not IMPUTATION_DEFAULTS:
    if os.path.exists(REFERENCE_DATA_PATH) and os.path.getsize(REFERENCE_DATA_PATH) > 0:
        ref_df = pd.read_csv(REFERENCE_DATA_PATH)
        # Drop target columns if they exist
        for target in ["income", "target"]:
            if target in ref_df.columns:
                ref_df = ref_df.drop(target, axis=1)
        
        # Store defaults
        for col in ref_df.columns:
            if pd.api.types.is_numeric_dtype(ref_df[col]):
                IMPUTATION_DEFAULTS[col] = float(ref_df[col].median())
            else:
                mode_series = ref_df[col].mode()
                IMPUTATION_DEFAULTS[col] = mode_series[0] if not mode_series.empty else ""
    else:
        # Fail-safe static defaults mapping to adult dataset
        IMPUTATION_DEFAULTS = {
            "age": 38,
            "workclass": "Private",
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Married-civ-spouse",
            "occupation": "Craft-repair",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
        }

# Pre-calculate IQR bounds and categories for rapid outlier & anomaly detection
REFERENCE_STATS = {}
if os.path.exists(REFERENCE_DATA_PATH) and os.path.getsize(REFERENCE_DATA_PATH) > 0:
    try:
        ref_df_stats = pd.read_csv(REFERENCE_DATA_PATH)
        for col in ref_df_stats.columns:
            if col in ["income", "target"]:
                continue
            if pd.api.types.is_numeric_dtype(ref_df_stats[col]):
                q1 = float(ref_df_stats[col].quantile(0.25))
                q3 = float(ref_df_stats[col].quantile(0.75))
                iqr = q3 - q1
                REFERENCE_STATS[col] = {
                    "type": "numerical",
                    "mean": float(ref_df_stats[col].mean()),
                    "std": float(ref_df_stats[col].std()),
                    "min": float(ref_df_stats[col].min()),
                    "max": float(ref_df_stats[col].max()),
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": q1 - 1.5 * iqr,
                    "upper_bound": q3 + 1.5 * iqr
                }
            else:
                REFERENCE_STATS[col] = {
                    "type": "categorical",
                    "categories": [str(c) for c in ref_df_stats[col].dropna().unique()]
                }
    except Exception as e:
        print(f"Failed to calculate reference stats: {e}")

# Ensure Database is initialized globally
init_db()

@app.get("/")
def home():
    return {"status": "SentinelML API running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "imputation_features": len(IMPUTATION_DEFAULTS)
    }

def log_inference_task(record_dict: dict):
    from api.database import SessionLocal
    db = SessionLocal()
    try:
        db_record = InferenceRecord(**record_dict)
        db.add(db_record)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Background Logging Error: {e}")
    finally:
        db.close()

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    try:
        # 1. Dump Pydantic payload using field aliases (hyphenated column names)
        payload_dict = request.model_dump(by_alias=True)
        
        # 2. Impute missing/None fields using baseline defaults
        imputed_list = []
        for field, default_val in IMPUTATION_DEFAULTS.items():
            if payload_dict.get(field) is None:
                payload_dict[field] = default_val
                imputed_list.append(field)
        
        # 3. Represent as DataFrame matching the exact feature order of baseline
        feature_cols = list(IMPUTATION_DEFAULTS.keys())
        df = pd.DataFrame([payload_dict])[feature_cols]
        
        # 4. Run prediction directly on raw DataFrame (Pipeline handles encoding internally)
        prediction = int(model.predict(df)[0])
        proba = float(model.predict_proba(df)[0][1])
        record_id = str(uuid.uuid4())
        
        # 5. Outlier & Anomaly Detection using Pre-calculated reference stats
        is_anomaly = False
        anomaly_reasons = []
        for field, val in payload_dict.items():
            # Skip fields that were imputed, since default imputation values are valid
            if field in imputed_list:
                continue
            
            stats = REFERENCE_STATS.get(field)
            if stats:
                if stats["type"] == "numerical":
                    try:
                        f_val = float(val)
                        if f_val < stats["lower_bound"] or f_val > stats["upper_bound"]:
                            is_anomaly = True
                            anomaly_reasons.append(f"{field}: {f_val} is an outlier (expected range: [{stats['lower_bound']:.1f}, {stats['upper_bound']:.1f}])")
                    except (ValueError, TypeError):
                        pass
                elif stats["type"] == "categorical":
                    if str(val) not in stats["categories"]:
                        is_anomaly = True
                        anomaly_reasons.append(f"{field}: '{val}' is an unseen category")
        
        # 6. Save telemetry asynchronously/thread-safely to SQLite
        record_dict = {
            "id": record_id,
            "timestamp": datetime.now(timezone.utc),
            "age": payload_dict.get("age"),
            "workclass": payload_dict.get("workclass"),
            "education": payload_dict.get("education"),
            "education_num": payload_dict.get("education-num"),
            "marital_status": payload_dict.get("marital-status"),
            "occupation": payload_dict.get("occupation"),
            "relationship": payload_dict.get("relationship"),
            "race": payload_dict.get("race"),
            "sex": payload_dict.get("sex"),
            "capital_gain": payload_dict.get("capital-gain"),
            "capital_loss": payload_dict.get("capital-loss"),
            "hours_per_week": payload_dict.get("hours-per-week"),
            "native_country": payload_dict.get("native-country"),
            "prediction": prediction,
            "probability": proba,
            "imputed_features": ",".join(imputed_list) if imputed_list else None,
            "is_anomaly": 1 if is_anomaly else 0,
            "anomaly_reasons": "; ".join(anomaly_reasons) if anomaly_reasons else None
        }
        
        background_tasks.add_task(log_inference_task, record_dict)
        
        return PredictionResponse(
            prediction=prediction,
            probability=proba,
            id=record_id,
            status="success",
            is_anomaly=is_anomaly,
            anomaly_reasons=anomaly_reasons if anomaly_reasons else None,
            imputed_features=imputed_list if imputed_list else None
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference Failure: {str(e)}"
        )
# --- Observability Dashboard & Telemetry Enhancements ---
import glob
from typing import Optional
import numpy as np
from scipy.stats import ks_2samp, chisquare
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from monitoring.evidently_runner import run_drift_analysis

REPORTS_DIR = "data/drift_reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

def calculate_categorical_drift(ref_series, curr_series):
    categories = list(set(ref_series.dropna().unique()) | set(curr_series.dropna().unique()))
    if len(categories) <= 1:
        return 1.0
    N_ref = len(ref_series)
    N_curr = len(curr_series)
    if N_ref == 0 or N_curr == 0:
        return 1.0
    ref_counts = ref_series.value_counts()
    curr_counts = curr_series.value_counts()
    O, E = [], []
    for cat in categories:
        o_count = curr_counts.get(cat, 0)
        p_ref = ref_counts.get(cat, 0) / N_ref
        e_count = p_ref * N_curr
        O.append(o_count)
        E.append(e_count)
    O = np.array(O, dtype=float)
    E = np.array(E, dtype=float)
    E = np.where(E == 0, 1e-5, E)
    E = E * (O.sum() / E.sum())
    try:
        stat, p_val = chisquare(f_obs=O, f_exp=E)
        return float(p_val)
    except:
        return 1.0

def calculate_feature_distributions(ref_df_local, current_df, numerical_features, categorical_features):
    distributions = {}
    
    # 1. Numerical features (histogram comparison)
    for feat in numerical_features:
        if feat in ref_df_local.columns and feat in current_df.columns:
            ref_vals = ref_df_local[feat].dropna()
            curr_vals = current_df[feat].dropna()
            if len(ref_vals) > 0:
                min_val = float(ref_vals.min())
                max_val = float(ref_vals.max())
                if min_val == max_val:
                    max_val += 1.0
                
                # Reference histogram
                ref_counts, bin_edges = np.histogram(ref_vals, bins=10, range=(min_val, max_val))
                ref_pct = (ref_counts / len(ref_vals)).tolist()
                
                # Current histogram using same bins
                if len(curr_vals) > 0:
                    curr_counts, _ = np.histogram(curr_vals, bins=bin_edges)
                    curr_pct = (curr_counts / len(curr_vals)).tolist()
                else:
                    curr_pct = [0.0] * 10
                
                labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
                distributions[feat] = {
                    "type": "numerical",
                    "labels": labels,
                    "reference": ref_pct,
                    "current": curr_pct
                }
                
    # 2. Categorical features (percentage comparison)
    for feat in categorical_features:
        if feat in ref_df_local.columns and feat in current_df.columns:
            ref_vals = ref_df_local[feat].dropna()
            curr_vals = current_df[feat].dropna()
            
            # Combine categories
            all_cats = sorted(list(set(ref_vals.unique()) | set(curr_vals.unique())))
            if len(all_cats) > 12:
                ref_counts = ref_vals.value_counts()
                top_cats = list(ref_counts.head(11).index)
                other_cats = [c for c in all_cats if c not in top_cats]
                
                ref_pct = []
                curr_pct = []
                for cat in top_cats:
                    ref_pct.append(float((ref_vals == cat).sum() / len(ref_vals)) if len(ref_vals) > 0 else 0.0)
                    curr_pct.append(float((curr_vals == cat).sum() / len(curr_vals)) if len(curr_vals) > 0 else 0.0)
                
                ref_pct.append(float((ref_vals.isin(other_cats)).sum() / len(ref_vals)) if len(ref_vals) > 0 else 0.0)
                curr_pct.append(float((curr_vals.isin(other_cats)).sum() / len(curr_vals)) if len(curr_vals) > 0 else 0.0)
                labels = top_cats + ["Other"]
            else:
                ref_pct = []
                curr_pct = []
                for cat in all_cats:
                    ref_pct.append(float((ref_vals == cat).sum() / len(ref_vals)) if len(ref_vals) > 0 else 0.0)
                    curr_pct.append(float((curr_vals == cat).sum() / len(curr_vals)) if len(curr_vals) > 0 else 0.0)
                labels = all_cats
                
            distributions[feat] = {
                "type": "categorical",
                "labels": labels,
                "reference": ref_pct,
                "current": curr_pct
            }
            
    return distributions

@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Dashboard template not found")
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

from pydantic import BaseModel

class FeedbackRequest(BaseModel):
    id: str
    feedback_label: int

@app.post("/api/feedback")
def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    record = db.query(InferenceRecord).filter(InferenceRecord.id == request.id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Inference record not found")
    record.feedback_label = request.feedback_label
    db.commit()
    return {"status": "success", "id": request.id, "feedback_label": request.feedback_label}

@app.get("/api/dashboard-data")
def get_dashboard_data(
    page: int = 1,
    page_size: int = 15,
    search_id: Optional[str] = None,
    filter_prediction: Optional[int] = None,
    filter_feedback: Optional[int] = None,
    db: Session = Depends(get_db)
):
    try:
        # Check cache first for heavy statistical calculations
        cached_analytics = dashboard_cache.get()
        if cached_analytics is None:
            # Query stats, counts, drift, performance metrics and timeline
            total_inferences = db.query(InferenceRecord).count()
            
            # Query last 100 predictions to get positive ratio
            last_100 = db.query(InferenceRecord).order_by(InferenceRecord.timestamp.desc()).limit(100).all()
            positive_ratio = 0.0
            if last_100:
                positive_ratio = sum([r.prediction for r in last_100]) / len(last_100)

            # Performance Metrics on last 500 labeled inferences
            labeled_records = db.query(InferenceRecord).filter(InferenceRecord.feedback_label.isnot(None)).order_by(InferenceRecord.timestamp.desc()).limit(500).all()
            perf_metrics = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "total_labeled": len(labeled_records)
            }
            if len(labeled_records) > 0:
                tp = sum([1 for r in labeled_records if r.prediction == 1 and r.feedback_label == 1])
                fp = sum([1 for r in labeled_records if r.prediction == 1 and r.feedback_label == 0])
                tn = sum([1 for r in labeled_records if r.prediction == 0 and r.feedback_label == 0])
                fn = sum([1 for r in labeled_records if r.prediction == 0 and r.feedback_label == 1])
                
                total = tp + fp + tn + fn
                if total > 0:
                    perf_metrics["accuracy"] = float((tp + tn) / total)
                if (tp + fp) > 0:
                    perf_metrics["precision"] = float(tp / (tp + fp))
                if (tp + fn) > 0:
                    perf_metrics["recall"] = float(tp / (tp + fn))
                p = perf_metrics["precision"]
                r = perf_metrics["recall"]
                if (p + r) > 0:
                    perf_metrics["f1_score"] = float(2 * (p * r) / (p + r))

            # Performance Timeline (Group labeled records chronologically)
            all_labeled = db.query(InferenceRecord).filter(InferenceRecord.feedback_label.isnot(None)).order_by(InferenceRecord.timestamp.asc()).all()
            performance_timeline = []
            if len(all_labeled) >= 5:
                num_records = len(all_labeled)
                chunk_size = max(3, num_records // 8) # Target around 8 points
                for idx in range(0, num_records, chunk_size):
                    chunk = all_labeled[idx : idx + chunk_size]
                    if len(chunk) < 1:
                        continue
                    c_tp = sum([1 for r in chunk if r.prediction == 1 and r.feedback_label == 1])
                    c_fp = sum([1 for r in chunk if r.prediction == 1 and r.feedback_label == 0])
                    c_tn = sum([1 for r in chunk if r.prediction == 0 and r.feedback_label == 0])
                    c_fn = sum([1 for r in chunk if r.prediction == 0 and r.feedback_label == 1])
                    c_total = c_tp + c_fp + c_tn + c_fn
                    if c_total > 0:
                        c_acc = (c_tp + c_tn) / c_total
                        c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
                        c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
                        c_f1 = 2 * (c_prec * c_rec) / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0
                        performance_timeline.append({
                            "label": chunk[0].timestamp.strftime("%m/%d %H:%M") + " - " + chunk[-1].timestamp.strftime("%m/%d %H:%M"),
                            "accuracy": float(c_acc),
                            "f1_score": float(c_f1)
                        })

            # Calculate Live Statistical Drift & Distributions
            drift_metrics = {}
            feature_distributions = {}
            if total_inferences > 0 and os.path.exists(REFERENCE_DATA_PATH) and os.path.getsize(REFERENCE_DATA_PATH) > 0:
                all_inferences_objs = db.query(InferenceRecord).order_by(InferenceRecord.timestamp.desc()).limit(5000).all()
                inferences_data = []
                for r in all_inferences_objs:
                    inferences_data.append({
                        "age": r.age,
                        "workclass": r.workclass,
                        "education": r.education,
                        "education-num": r.education_num,
                        "marital-status": r.marital_status,
                        "occupation": r.occupation,
                        "relationship": r.relationship,
                        "race": r.race,
                        "sex": r.sex,
                        "capital-gain": r.capital_gain,
                        "capital-loss": r.capital_loss,
                        "hours-per-week": r.hours_per_week,
                        "native-country": r.native_country
                    })
                current_df = pd.DataFrame(inferences_data)
                ref_df_local = pd.read_csv(REFERENCE_DATA_PATH)
                
                # Compute Kolmogorov-Smirnov Test
                numerical_features = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
                for feat in numerical_features:
                    if feat in ref_df_local.columns and feat in current_df.columns:
                        ref_vals = ref_df_local[feat].dropna().values
                        curr_vals = current_df[feat].dropna().values
                        if len(ref_vals) > 0 and len(curr_vals) > 0:
                            stat, p_val = ks_2samp(ref_vals, curr_vals)
                            drift_metrics[feat] = float(p_val)
                        else:
                            drift_metrics[feat] = 1.0

                # Compute Chi-Square Test
                categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
                for feat in categorical_features:
                    if feat in ref_df_local.columns and feat in current_df.columns:
                        p_val = calculate_categorical_drift(ref_df_local[feat], current_df[feat])
                        drift_metrics[feat] = p_val
                
                # Precompute distribution graphs reference vs current
                feature_distributions = calculate_feature_distributions(ref_df_local, current_df, numerical_features, categorical_features)
            else:
                for feat in ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]:
                    drift_metrics[feat] = None
                for feat in ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]:
                    drift_metrics[feat] = None

            # Calculate Null / Imputation Rates in recent 1000 inferences
            recent_1000 = db.query(InferenceRecord).order_by(InferenceRecord.timestamp.desc()).limit(1000).all()
            imputation_counts = {feat: 0 for feat in IMPUTATION_DEFAULTS.keys()}
            total_recent = len(recent_1000)
            for r in recent_1000:
                if r.imputed_features:
                    imputed = r.imputed_features.split(",")
                    for f_imp in imputed:
                        if f_imp in imputation_counts:
                            imputation_counts[f_imp] += 1
            imputation_rates = {}
            for f_imp, val_count in imputation_counts.items():
                imputation_rates[f_imp] = float(val_count / total_recent) if total_recent > 0 else 0.0

            # Scan Evidently Reports Directory
            report_files = glob.glob(os.path.join(REPORTS_DIR, "drift_*.html"))
            reports = []
            for r_file in sorted(report_files, reverse=True):
                filename = os.path.basename(r_file)
                parts = filename.replace("drift_", "").replace(".html", "").split("_")
                date_str = ""
                if len(parts) >= 2:
                    try:
                        dt = datetime.strptime(f"{parts[0]}_{parts[1]}", "%Y%m%d_%H%M%S")
                        date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        date_str = "Unknown Date"
                else:
                    date_str = "Unknown Date"
                reports.append({
                    "filename": filename,
                    "date": date_str,
                    "url": f"/reports/{filename}"
                })

            # Save in cache
            cached_analytics = {
                "total_inferences": total_inferences,
                "positive_ratio": positive_ratio,
                "drift_metrics": drift_metrics,
                "performance_metrics": perf_metrics,
                "performance_timeline": performance_timeline,
                "feature_distributions": feature_distributions,
                "imputation_rates": imputation_rates,
                "reports": reports
            }
            dashboard_cache.set(cached_analytics)

        # Dynamic Querying of Logs & Recent Anomalies (independent of cache)
        query = db.query(InferenceRecord)
        if search_id:
            query = query.filter(InferenceRecord.id.like(f"%{search_id}%"))
        if filter_prediction is not None:
            query = query.filter(InferenceRecord.prediction == filter_prediction)
        if filter_feedback is not None:
            if filter_feedback == -1:
                query = query.filter(InferenceRecord.feedback_label.is_(None))
            else:
                query = query.filter(InferenceRecord.feedback_label == filter_feedback)
                
        total_filtered = query.count()
        logs_objs = query.order_by(InferenceRecord.timestamp.desc()).offset((page - 1) * page_size).limit(page_size).all()
        
        recent_inferences = []
        for r in logs_objs:
            recent_inferences.append({
                "id": r.id,
                "timestamp": r.timestamp.isoformat(),
                "age": r.age,
                "workclass": r.workclass,
                "education": r.education,
                "education_num": r.education_num,
                "marital_status": r.marital_status,
                "occupation": r.occupation,
                "relationship": r.relationship,
                "race": r.race,
                "sex": r.sex,
                "capital_gain": r.capital_gain,
                "capital_loss": r.capital_loss,
                "hours_per_week": r.hours_per_week,
                "native_country": r.native_country,
                "prediction": r.prediction,
                "probability": r.probability,
                "feedback_label": r.feedback_label,
                "imputed_features": r.imputed_features.split(",") if r.imputed_features else [],
                "is_anomaly": bool(r.is_anomaly),
                "anomaly_reasons": r.anomaly_reasons.split("; ") if r.anomaly_reasons else []
            })

        # Query recent 10 anomalies
        anomaly_objs = db.query(InferenceRecord).filter(InferenceRecord.is_anomaly == 1).order_by(InferenceRecord.timestamp.desc()).limit(10).all()
        recent_anomalies = []
        for r in anomaly_objs:
            recent_anomalies.append({
                "id": r.id,
                "timestamp": r.timestamp.isoformat(),
                "prediction": r.prediction,
                "probability": r.probability,
                "anomaly_reasons": r.anomaly_reasons.split("; ") if r.anomaly_reasons else []
            })

        # Prepare final response payload
        payload = {
            "system_healthy": True,
            "total_inferences": cached_analytics["total_inferences"],
            "positive_ratio": cached_analytics["positive_ratio"],
            "drift_metrics": cached_analytics["drift_metrics"],
            "performance_metrics": cached_analytics["performance_metrics"],
            "performance_timeline": cached_analytics["performance_timeline"],
            "feature_distributions": cached_analytics["feature_distributions"],
            "imputation_rates": cached_analytics["imputation_rates"],
            "reports": cached_analytics["reports"],
            "recent_inferences": recent_inferences,
            "recent_anomalies": recent_anomalies,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_records": total_filtered,
                "total_pages": (total_filtered + page_size - 1) // page_size if total_filtered > 0 else 1
            }
        }
        return JSONResponse(payload)

    except Exception as e:
        return JSONResponse({
            "system_healthy": False,
            "error": str(e),
            "total_inferences": 0,
            "positive_ratio": 0.0,
            "drift_metrics": {},
            "performance_metrics": {},
            "performance_timeline": [],
            "feature_distributions": {},
            "imputation_rates": {},
            "reports": [],
            "recent_inferences": [],
            "recent_anomalies": [],
            "pagination": {"page": 1, "page_size": 15, "total_records": 0, "total_pages": 1}
        }, status_code=500)

@app.post("/api/trigger-drift")
def trigger_drift():
    try:
        run_drift_analysis()
        report_files = glob.glob(os.path.join(REPORTS_DIR, "drift_*.html"))
        if report_files:
            latest = os.path.basename(sorted(report_files, reverse=True)[0])
        else:
            latest = "None"
        # Reset cache on manual drift trigger so UI updates immediately
        dashboard_cache.last_updated = 0
        return {"status": "success", "report": latest}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run drift evaluation: {str(e)}")
