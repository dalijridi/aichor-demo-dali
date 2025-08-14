# Use TensorFlow GPU base image which includes CUDA and cuDNN
FROM tensorflow/tensorflow:2.13.0-gpu

# Set environment variables for CUDA and Python output
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV TF_CPP_MIN_LOG_LEVEL=0

# Update system packages
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages
RUN pip install --no-cache-dir \
    tensorboardX \
    boto3 \
    google-cloud-storage \
    numpy

# Set working directory
WORKDIR /app

# Copy source code
COPY ./src ./src
COPY main.py .

# Create a startup script to ensure proper logging
RUN echo '#!/bin/bash\nexec python3 -u "$@" 2>&1' > /app/run.sh && chmod +x /app/run.sh

# Set default command (can be overridden in manifest)
CMD ["/app/run.sh", "main.py"]
