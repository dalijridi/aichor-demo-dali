import os
import time
import glob as globmod
from urllib.parse import urlparse

import boto3
from tensorboardX import SummaryWriter

LOCAL_TB_DIR = "/tmp/tensorboard_logs"


def upload_dir_to_s3(local_dir, s3_url):
    parsed = urlparse(s3_url)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    endpoint_url = os.getenv("AWS_ENDPOINT_URL") or os.getenv("CEPH_AWS_ENDPOINT_URL")
    s3 = boto3.client("s3", endpoint_url=endpoint_url)
    for filepath in globmod.glob(os.path.join(local_dir, "**", "*"), recursive=True):
        if os.path.isfile(filepath):
            key = prefix + os.path.relpath(filepath, local_dir)
            s3.upload_file(filepath, bucket, key)
            print(f"Uploaded {filepath} -> s3://{bucket}/{key}")


def aichor_write_tensorboard():
    tb_remote_path = os.getenv("AICHOR_TENSORBOARD_PATH") or os.getenv("AICHOR_LOGS_PATH")
    print(f"### Remote path: {tb_remote_path}")

    writer = SummaryWriter(LOCAL_TB_DIR)
    for step, val in enumerate([0.31, 0.28, 0.24, 0.20, 0.18], start=5):
        writer.add_scalar("demo/loss", val, step)
        time.sleep(1)
    writer.flush()
    writer.close()

    upload_dir_to_s3(LOCAL_TB_DIR, tb_remote_path)
    print("### TensorBoard logs uploaded to", tb_remote_path)


def print_test():
    # do math multiplications and then print test

    for i in range(10):
        a = i * 2
        b = i * 3
        c = a + b
        print(f"Test {i}: {c}")

    

def main():
    aichor_write_tensorboard()

    print_test()

    time.sleep(1800)

if __name__ == "__main__":
    main()
