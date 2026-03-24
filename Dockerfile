FROM python:3.12-slim

RUN pip install tensorboardX boto3 tensorflow tensorflow_io

WORKDIR /app
COPY main.py . 


# RUN --mount=type=cache,target=/root/.cache/uv \
#     echo "CACHE TEST $(date)" > /root/.cache/uv/test.txt && \
#     cat /root/.cache/uv/test.txt
 
# RUN --mount=type=cache,target=/root/.cache/uv \
#     cat /root/.cache/uv/test.txt
