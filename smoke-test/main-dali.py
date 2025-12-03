import time
import sys

def main():
    for i in range(1, 101):
        print(f"[Job] Step {i}/100 running...", flush=True)
        time.sleep(1)  # simulate work
    print("[Job] Completed!", flush=True)

if __name__ == "__main__":
    main()
