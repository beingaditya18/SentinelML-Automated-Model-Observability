import os
import time
import sqlite3
import pandas as pd
import numpy as np
import yaml
from prometheus_client import start_http_server, Gauge
from scipy.stats import ks_2samp, chisquare

# Port to expose Prometheus metrics
EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", 8000))
REFERENCE_PATH = "data/reference/reference_data.csv"
DB_PATH = "data/live/sentinelml.db"
CONFIG_PATH = "data/model_config.yaml"

# Declare Prometheus Metrics
INFERENCE_VOLUME = Gauge("sentinelml_inferences_total", "Total inference request count")
PREDICTION_RATE = Gauge("sentinelml_prediction_positive_ratio", "Ratio of positive income predictions (>50K) in last 100 predictions")
DRIFT_P_VALUE = Gauge("sentinelml_drift_ks_p_value", "Kolmogorov-Smirnov p-value for numerical features", ["feature"])
DRIFT_CHI_SQUARE_P_VALUE = Gauge("sentinelml_drift_chisquare_p_value", "Chi-Square goodness-of-fit p-value for categorical features", ["feature"])

def calculate_categorical_drift(ref_series, curr_series):
    # Get all unique categories across both reference and current series
    categories = list(set(ref_series.dropna().unique()) | set(curr_series.dropna().unique()))
    if len(categories) <= 1:
        return 1.0  # No drift if there is 0 or 1 category
    
    N_ref = len(ref_series)
    N_curr = len(curr_series)
    if N_ref == 0 or N_curr == 0:
        return 1.0
        
    ref_counts = ref_series.value_counts()
    curr_counts = curr_series.value_counts()
    
    O = []
    E = []
    for cat in categories:
        o_count = curr_counts.get(cat, 0)
        # Expected count based on reference proportion
        p_ref = ref_counts.get(cat, 0) / N_ref
        e_count = p_ref * N_curr
        
        O.append(o_count)
        E.append(e_count)
        
    O = np.array(O, dtype=float)
    E = np.array(E, dtype=float)
    
    # Avoid zero expectation to prevent division by zero in Chi-Square
    E = np.where(E == 0, 1e-5, E)
    # Re-normalize expected to sum to observed count
    E = E * (O.sum() / E.sum())
    
    try:
        stat, p_val = chisquare(f_obs=O, f_exp=E)
        return float(p_val)
    except Exception:
        return 1.0

def calculate_metrics():
    # 1. Read Reference Data
    if not os.path.exists(REFERENCE_PATH):
        return
    ref_df = pd.read_csv(REFERENCE_PATH)
    
    # 2. Read SQLite Inferences
    if not os.path.exists(DB_PATH):
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        # Limit SQL query window to the last 5000 records to prevent OOM
        current_df = pd.read_sql_query("SELECT * FROM inferences ORDER BY timestamp DESC LIMIT 5000", conn)
        # Total inferences counts in the DB
        total_inferences_df = pd.read_sql_query("SELECT COUNT(*) as count FROM inferences", conn)
        total_count = int(total_inferences_df["count"].iloc[0])
        conn.close()
    except Exception as e:
        print(f"Database query error: {e}")
        return

    if len(current_df) == 0:
        return

    # Update Inferences count
    INFERENCE_VOLUME.set(total_count)

    # Calculate rolling positive prediction ratio
    last_100 = current_df.head(100)
    pos_ratio = float(last_100["prediction"].mean())
    PREDICTION_RATE.set(pos_ratio)

    # Align columns from SQLite database underscores to reference hyphens
    rename_map = {
        "education_num": "education-num",
        "marital_status": "marital-status",
        "hours_per_week": "hours-per-week",
        "native_country": "native-country",
        "capital_gain": "capital-gain",
        "capital_loss": "capital-loss"
    }
    current_df = current_df.rename(columns=rename_map)

    # Load dynamic configurations
    numerical_features = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                config_data = yaml.safe_load(f)
                numerical_features = [feat["name"] for feat in config_data.get("features", {}).get("numerical", [])]
                categorical_features = [feat["name"] for feat in config_data.get("features", {}).get("categorical", [])]
        except Exception as e:
            print(f"Failed to load dynamic model config in exporter: {e}")

    # Compute Kolmogorov-Smirnov Test for all numerical features
    for feat in numerical_features:
        if feat in ref_df.columns and feat in current_df.columns:
            ref_vals = ref_df[feat].dropna().values
            curr_vals = current_df[feat].dropna().values
            if len(ref_vals) > 0 and len(curr_vals) > 0:
                # ks_2samp returns statistic, pvalue
                stat, p_val = ks_2samp(ref_vals, curr_vals)
                DRIFT_P_VALUE.labels(feature=feat).set(p_val)

    # Compute Chi-Square Test for all categorical features
    for feat in categorical_features:
        if feat in ref_df.columns and feat in current_df.columns:
            p_val = calculate_categorical_drift(ref_df[feat], current_df[feat])
            if p_val is not None:
                DRIFT_CHI_SQUARE_P_VALUE.labels(feature=feat).set(p_val)

def main():
    print(f"Starting Prometheus exporter HTTP server on port {EXPORTER_PORT}...")
    start_http_server(EXPORTER_PORT)
    
    while True:
        try:
            calculate_metrics()
        except Exception as e:
            print(f"Error calculating metrics: {e}")
        time.sleep(15)

if __name__ == "__main__":
    main()
