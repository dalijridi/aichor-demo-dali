import os
import sys
import time

def main():
    print("=== START ===", flush=True)

    tb_path = os.environ.get("AICHOR_LOGS_PATH", "NOT_SET")
    print(f"AICHOR_LOGS_PATH={tb_path}", flush=True)

    if tb_path == "NOT_SET":
        print("ERROR: AICHOR_LOGS_PATH not set", flush=True)
        sys.exit(1)

    # Wait for GCS Fuse mount to be ready
    mount = "/mnt/tensorboard"
    for i in range(30):
        try:
            os.listdir(mount)
            print(f"Mount ready after {i}s", flush=True)
            break
        except Exception as e:
            print(f"Waiting for mount ({i}s): {e}", flush=True)
            time.sleep(1)

    # Write a plain file first to confirm the mount works
    test_file = os.path.join(tb_path, "test.txt")
    try:
        os.makedirs(tb_path, exist_ok=True)
        print(f"makedirs OK: {tb_path}", flush=True)
        with open(test_file, "w") as f:
            f.write("hello\n")
        print(f"plain file write OK: {test_file}", flush=True)
    except Exception as e:
        print(f"ERROR writing plain file: {e}", flush=True)
        sys.exit(1)

    # Write tensorboard events
    try:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(tb_path)
        for step, val in enumerate([0.31, 0.28, 0.24, 0.20, 0.18], start=1):
            writer.add_scalar("demo/loss", val, step)
            print(f"wrote step {step}", flush=True)
        writer.flush()
        writer.close()
        print("tensorboard write OK", flush=True)
    except Exception as e:
        print(f"ERROR writing tensorboard: {e}", flush=True)

    print("=== sleeping 1800s ===", flush=True)
    time.sleep(1800)

if __name__ == "__main__":
    main()
