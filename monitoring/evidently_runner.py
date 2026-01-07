import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os
from datetime import datetime

REFERENCE = "data/reference/reference_data.csv"
CURRENT = "data/live/live_predictions.csv"
REPORT_DIR = "data/drift_reports"

os.makedirs(REPORT_DIR, exist_ok=True)

reference = pd.read_csv(REFERENCE)
current = pd.read_csv(CURRENT)

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)

path = f"{REPORT_DIR}/drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
report.save_html(path)

print(f"Drift report saved → {path}")
