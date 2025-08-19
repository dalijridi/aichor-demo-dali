# Use an official PyTorch image with CUDA 12.1 support
# This ensures that CUDA/cuDNN versions are compatible
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the Python script into the container
COPY main.gpu.py .

# Install any additional dependencies if you have them (in a requirements.txt)
# RUN pip install -r requirements.txt

# Set the entrypoint to run the training script when the container starts
ENTRYPOINT ["python", "main.gpu.py"]
