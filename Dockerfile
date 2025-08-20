# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set environment variables for better logging
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_LAUNCH_BLOCKING=1

# Set working directory
WORKDIR /app

# Copy the training script
COPY main_gpu.py .

# Make script executable
RUN chmod +x main_gpu.py

# Test that PyTorch is working (CUDA might not be available during build, that's normal)
RUN python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available during build:', torch.cuda.is_available()); print('Build completed successfully')"

# Create a startup wrapper script
RUN echo '#!/bin/bash' > /app/run.sh && \
    echo 'set -e' >> /app/run.sh && \
    echo 'echo "=== CONTAINER STARTUP ==="' >> /app/run.sh && \
    echo 'echo "Timestamp: $(date)"' >> /app/run.sh && \
    echo 'echo "Running as user: $(whoami)"' >> /app/run.sh && \
    echo 'echo "Working directory: $(pwd)"' >> /app/run.sh && \
    echo 'echo "Python path: $(which python3)"' >> /app/run.sh && \
    echo 'echo "NVIDIA SMI output:"' >> /app/run.sh && \
    echo 'nvidia-smi || echo "nvidia-smi not available"' >> /app/run.sh && \
    echo 'echo "=== STARTING TRAINING ==="' >> /app/run.sh && \
    echo 'exec python3 -u main_gpu.py' >> /app/run.sh && \
    chmod +x /app/run.sh

# Default command
CMD ["/app/run.sh"]
