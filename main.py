import os
import time
from tensorboardX import SummaryWriter


def aichor_write_tensorboard():
    tb_path = os.environ["AICHOR_LOGS_PATH"]
    print(f"### Writing TensorBoard logs to {tb_path}")
    os.makedirs(tb_path, exist_ok=True)

    writer = SummaryWriter(tb_path)
    for step, val in enumerate([0.31, 0.28, 0.24, 0.20, 0.18], start=5):
        writer.add_scalar("demo/loss", val, step)
        time.sleep(1)
    writer.flush()
    writer.close()

    print("### Wrote TensorBoard event files to", tb_path)


def main():
    aichor_write_tensorboard()
    time.sleep(1800)
