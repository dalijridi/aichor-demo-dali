# Start from NVIDIA CUDA runtime image with cuDNN
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Python + pip setup
RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip3 install --upgrade pip

# Install dependencies
RUN pip3 install tensorboardX boto3 tensorflow[and-cuda]

# Set Python as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Workdir and copy code
WORKDIR /app
COPY ./src ./src
COPY main.py .

# Ensure we can see GPUs when running
RUN python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
