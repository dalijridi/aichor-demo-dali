# Use official PyTorch image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system utilities for logging
RUN apt-get update && apt-get install -y \
    procps \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for logging
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_LAUNCH_BLOCKING=1

# Set working directory
WORKDIR /app

# Copy the training script
COPY main_gpu.py .

# Make script executable
RUN chmod +x cloud_logging_script.py

# Create comprehensive startup wrapper
RUN echo '#!/bin/bash' > /app/start.sh && \
    echo 'set -e' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Multi-channel logging function' >> /app/start.sh && \
    echo 'log_multi() {' >> /app/start.sh && \
    echo '    echo "[WRAPPER-STDOUT] $1"' >> /app/start.sh && \
    echo '    echo "[WRAPPER-STDERR] $1" >&2' >> /app/start.sh && \
    echo '    echo "[WRAPPER-LOG] $(date): $1" >> /tmp/wrapper.log' >> /app/start.sh && \
    echo '}' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo 'log_multi "=== CONTAINER STARTUP ==="' >> /app/start.sh && \
    echo 'log_multi "Timestamp: $(date)"' >> /app/start.sh && \
    echo 'log_multi "Hostname: $(hostname)"' >> /app/start.sh && \
    echo 'log_multi "User: $(whoami)"' >> /app/start.sh && \
    echo 'log_multi "Working dir: $(pwd)"' >> /app/start.sh && \
    echo 'log_multi "Environment variables:"' >> /app/start.sh && \
    echo 'env | grep -E "(PYTHON|CUDA|NVIDIA)" | while read line; do log_multi "  $line"; done' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo 'log_multi "=== NVIDIA-SMI OUTPUT ==="' >> /app/start.sh && \
    echo 'if nvidia-smi 2>/dev/null; then' >> /app/start.sh && \
    echo '    log_multi "✅ nvidia-smi successful"' >> /app/start.sh && \
    echo 'else' >> /app/start.sh && \
    echo '    log_multi "⚠️ nvidia-smi not available"' >> /app/start.sh && \
    echo 'fi' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo 'log_multi "=== STARTING PYTHON SCRIPT ==="' >> /app/start.sh && \
    echo 'exec python3 -u cloud_logging_script.py 2>&1 | tee -a /tmp/python_output.log' >> /app/start.sh && \
    chmod +x /app/start.sh

# Test build
RUN python3 -c "print('✅ Docker build test successful')"

# Default command
CMD ["/app/start.sh"]
