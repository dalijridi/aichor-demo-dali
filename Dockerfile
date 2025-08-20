# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set environment variables for better logging
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV NVIDIA_VISIBLE_DEVICES=all

# Set working directory
WORKDIR /app

# Copy the training script
COPY main_gpu.py .

# Make script executable
RUN chmod +x main_gpu.py

# Test that PyTorch and CUDA are working
RUN python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Default command
CMD ["python3", "main_gpu.py"]
