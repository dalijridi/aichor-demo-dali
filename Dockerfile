# Use an official TensorFlow image with GPU support
# This image includes Python, CUDA, and cuDNN, all pre-configured.
#FROM tensorflow/tensorflow:2.13.0-gpu
FROM europe-docker.pkg.dev/vertex-ai/training/tf-gpu.2.13-py310

# Set the working directory in the container
WORKDIR /app

# Copy the Python script into the container
#COPY main_gpu.py .
#debug script to test logging
COPY diagnostic_script.py .

# No ENTRYPOINT is needed, as the command will be provided by the manifest
