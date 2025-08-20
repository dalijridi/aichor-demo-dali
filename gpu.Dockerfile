# Use EXACTLY the same pattern as your working TensorFlow Dockerfile
FROM python:3.8-slim

# Install only the basic dependencies (like your working TF setup)
RUN pip install tensorboardX boto3

# Set working directory  
WORKDIR /app

# Copy the script (your TF setup copies to root, let's match that)
COPY main.gpu.py .
