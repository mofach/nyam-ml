FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
  libgl1 \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PORT=8080

COPY requirements.txt .
RUN pip install --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt

COPY . .

CMD exec gunicorn \
  --workers 1 \
  --threads 4 \
  --bind :$PORT \
  --timeout 300 \
  app:app
