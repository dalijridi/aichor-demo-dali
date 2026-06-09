import os
import sys
import time


def write_large_file_local(path, target_bytes):
    chunk = b"x" * (256 * 1024 ** 2)
    written = 0
    print(f"Writing {target_bytes / 1024**3:.1f} GiB to {path} ...", flush=True)
    os.makedirs(path, exist_ok=True)
    dest = os.path.join(path, "large_test_file.bin")
    with open(dest, "wb") as f:
        while written < target_bytes:
            n = min(len(chunk), target_bytes - written)
            f.write(chunk[:n])
            written += n
            print(f"  {written / 1024**3:.2f} GiB written", flush=True)
    print(f"Done. {os.path.getsize(dest) / 1024**3:.2f} GiB at {dest}", flush=True)


def write_large_file_s3(s3_path, target_bytes):
    import boto3

    without_prefix = s3_path[len("s3://"):]
    bucket, _, prefix = without_prefix.partition("/")
    key = prefix.rstrip("/") + "/large_test_file.bin"

    endpoint = os.environ.get("S3_ENDPOINT")
    print(f"S3_ENDPOINT={endpoint}", flush=True)

    client = boto3.client("s3", endpoint_url=endpoint)

    chunk_size = 256 * 1024 ** 2
    print(f"Starting multipart upload: {target_bytes / 1024**3:.1f} GiB → s3://{bucket}/{key}", flush=True)

    resp = client.create_multipart_upload(Bucket=bucket, Key=key)
    upload_id = resp["UploadId"]
    parts = []
    uploaded = 0
    part_num = 1
    chunk = b"x" * chunk_size

    try:
        while uploaded < target_bytes:
            n = min(chunk_size, target_bytes - uploaded)
            part_resp = client.upload_part(
                Bucket=bucket, Key=key,
                PartNumber=part_num, UploadId=upload_id,
                Body=chunk[:n],
            )
            parts.append({"PartNumber": part_num, "ETag": part_resp["ETag"]})
            uploaded += n
            print(f"  {uploaded / 1024**3:.2f} GiB uploaded (part {part_num})", flush=True)
            part_num += 1

        client.complete_multipart_upload(
            Bucket=bucket, Key=key, UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
        print(f"Done. Uploaded {uploaded / 1024**3:.2f} GiB to s3://{bucket}/{key}", flush=True)

    except Exception:
        client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        raise


def main():
    print("=== START ===", flush=True)

    tb_path = os.environ.get("AICHOR_LOGS_PATH", "NOT_SET")
    print(f"AICHOR_LOGS_PATH={tb_path}", flush=True)

    if tb_path == "NOT_SET":
        print("ERROR: AICHOR_LOGS_PATH not set", flush=True)
        sys.exit(1)

    target_bytes = 2 * 1024 ** 3

    if tb_path.startswith("s3://"):
        write_large_file_s3(tb_path, target_bytes)
    else:
        mount = "/mnt/tensorboard"
        for i in range(30):
            try:
                os.listdir(mount)
                print(f"Mount ready after {i}s", flush=True)
                break
            except Exception as e:
                print(f"Waiting for mount ({i}s): {e}", flush=True)
                time.sleep(1)

        write_large_file_local(tb_path, target_bytes)

    try:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(tb_path)
        for step in range(5):
            writer.add_scalar("demo/loss", 1.0 / (step + 1), step)
        writer.flush()
        writer.close()
        print("Tensorboard events written", flush=True)
    except Exception as e:
        print(f"Tensorboard events write failed: {e}", flush=True)

    print("=== sleeping 60s ===", flush=True)
    time.sleep(60)


if __name__ == "__main__":
    main()
