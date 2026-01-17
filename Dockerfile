FROM python:3.12-slim

# Install library OS yang dibutuhkan
RUN apt-get update && apt-get install -y \
  libgl1 \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set TensorFlow Legacy Mode
ENV TF_USE_LEGACY_KERAS=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Install Dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt

# Copy Kode
COPY . .

ENV PORT=8080

# PENTING: Kurangi workers untuk save memory
# 1 worker cukup untuk startup, autoscaling Cloud Run yang handle traffic
CMD exec gunicorn --workers 1 --threads 4 --bind :$PORT --timeout 300 --preload app:app