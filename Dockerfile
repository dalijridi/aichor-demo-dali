FROM python:3.8-slim

RUN pip install tensorboardX boto3

WORKDIR /app
COPY ./smoke-test/src ./src
COPY ./smoke-test/main.py .
COPY ./smoke-test/main-dali.py .
