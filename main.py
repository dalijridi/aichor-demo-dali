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

    mount = "/mnt/tensorboard"
    for i in range(30):
        try:
            os.listdir(mount)
            print(f"Mount ready after {i}s", flush=True)
            break
        except Exception as e:
            print(f"Waiting for mount ({i}s): {e}", flush=True)
            time.sleep(1)

    os.makedirs(tb_path, exist_ok=True)

    # Write a 2Gi file in 256 MB chunks to stay within memory
    target_bytes = 2 * 1024 ** 3
    chunk = b"x" * (256 * 1024 ** 2)
    written = 0
    large_file = os.path.join(tb_path, "large_test_file.bin")

    print(f"Writing {target_bytes / 1024**3:.1f} GiB to {large_file} ...", flush=True)
    with open(large_file, "wb") as f:
        while written < target_bytes:
            f.write(chunk)
            written += len(chunk)
            print(f"  {written / 1024**3:.2f} GiB written", flush=True)

    actual = os.path.getsize(large_file)
    print(f"Done. File size on disk: {actual / 1024**3:.2f} GiB", flush=True)

    # Also write tensorboard events
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(tb_path)
    for step in range(5):
        writer.add_scalar("demo/loss", 1.0 / (step + 1), step)
    writer.flush()
    writer.close()
    print("Tensorboard events written", flush=True)

    print("=== sleeping 600s ===", flush=True)
    time.sleep(600)


if __name__ == "__main__":
    main()
