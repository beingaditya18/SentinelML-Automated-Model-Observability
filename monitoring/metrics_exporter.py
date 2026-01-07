from prometheus_client import start_http_server, Gauge
import pandas as pd
import time

DRIFT_SCORE = Gauge("data_drift_score", "Current drift score")

while True:
    try:
        df = pd.read_csv("data/live/live_predictions.csv")
        DRIFT_SCORE.set(len(df))
    except:
        pass

    time.sleep(30)
