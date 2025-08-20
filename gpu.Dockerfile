# Use the same approach as the working TensorFlow setup
FROM python:3.9-slim

# Install system dependencies for CUDA (if needed) and PyTorch
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support (similar to how the TF script installs TF)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install numpy argparse

# Set working directory
WORKDIR /app

# Copy training script
COPY main.gpu.py .

# Set environment variables for proper output
ENV PYTHONUNBUFFERED=1
