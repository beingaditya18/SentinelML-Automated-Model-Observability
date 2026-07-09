import time
import os
import sys

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from monitoring.evidently_runner import run_drift_analysis

INTERVAL_SECONDS = int(os.getenv("DRIFT_INTERVAL_SECONDS", 60))

def main():
    print(f"Starting automated monitoring scheduler (Interval: {INTERVAL_SECONDS}s)...")
    while True:
        try:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Triggering periodic drift analysis...")
            run_drift_analysis()
        except Exception as e:
            print(f"Scheduler encountered an error: {e}")
        
        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
