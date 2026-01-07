import time
import os

while True:
    print("Running scheduled drift check...")
    os.system("python monitoring/evidently_runner.py")
    time.sleep(3600)  # every 1 hour
