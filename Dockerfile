FROM rayproject/ray:2.23.0-cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY ./src ./src
COPY main.py .
