# Use TensorFlow GPU base image which includes CUDA and cuDNN
FROM tensorflow/tensorflow:2.13.0-gpu

# Set environment variables for CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

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

# Ensure Python outputs are not buffered (important for Vertex AI logs)
ENV PYTHONUNBUFFERED=1

# Set default command (can be overridden in manifest)
CMD ["python3", "-u", "main.py"]
