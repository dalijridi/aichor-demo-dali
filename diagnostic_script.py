# debug_script.py
import time
import sys
from datetime import datetime

print(f"--- Debug script started at {datetime.now()} ---", flush=True)
sys.stdout.flush() # Be extra sure it flushes

# Keep the container alive for 5 minutes (300 seconds)
print("Now sleeping for 300 seconds...", flush=True)
sys.stdout.flush()

time.sleep(300)

print(f"--- Debug script finished sleeping at {datetime.now()} ---", flush=True)
sys.stdout.flush()
