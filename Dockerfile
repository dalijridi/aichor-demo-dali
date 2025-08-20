# Use a more recent PyTorch image with better CUDA compatibility
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all

# Set the working directory in the container
WORKDIR /app

# Install any additional dependencies if needed
RUN pip install --no-cache-dir argparse

# Copy the Python script into the container
COPY ./src ./src
COPY main.gpu.py .

# Make sure the script is executable
RUN chmod +x main.gpu.py

# Optional: Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python3 -c "import torch; print('Health check passed')" || exit 1
