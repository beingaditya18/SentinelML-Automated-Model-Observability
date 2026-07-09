import os
import sqlite3
from datetime import datetime
import pandas as pd
from evidently.legacy.report.report import Report
from evidently.legacy.metric_preset import DataDriftPreset

REFERENCE = "data/reference/reference_data.csv"
DB_PATH = "data/live/sentinelml.db"
REPORT_DIR = "data/drift_reports"

def run_drift_analysis():
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    if not os.path.exists(REFERENCE):
        print(f"Reference data not found at {REFERENCE}. Please run training first.")
        return
        
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}. No live inferences recorded yet.")
        return
        
    # Read reference data
    reference = pd.read_csv(REFERENCE)
    for target in ["income", "target"]:
        if target in reference.columns:
            reference = reference.drop(target, axis=1)
            
    # Read live data from SQLite
    try:
        conn = sqlite3.connect(DB_PATH)
        current = pd.read_sql_query("SELECT * FROM inferences ORDER BY timestamp DESC LIMIT 5000", conn)
        conn.close()
    except Exception as e:
        print(f"Failed to query database: {e}")
        return
        
    if len(current) == 0:
        print("No live prediction logs found. Skipping drift generation.")
        return
        
    # Align columns from SQLite database underscores to reference hyphens
    rename_map = {
        "education_num": "education-num",
        "marital_status": "marital-status",
        "hours_per_week": "hours-per-week",
        "native_country": "native-country",
        "capital_gain": "capital-gain",
        "capital_loss": "capital-loss"
    }
    current = current.rename(columns=rename_map)
    
    # Filter only relevant columns
    cols_to_compare = [c for c in reference.columns if c in current.columns]
    reference_aligned = reference[cols_to_compare]
    current_aligned = current[cols_to_compare]
    
    # Run Evidently report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_aligned, current_data=current_aligned)
    
    path = f"{REPORT_DIR}/drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report.save_html(path)
    print(f"Drift report generated successfully -> {path}")

if __name__ == "__main__":
    run_drift_analysis()
