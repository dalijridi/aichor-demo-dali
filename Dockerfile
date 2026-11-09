FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install "delta-sharing==1.3.3"

COPY delta_sharing_check.py .
