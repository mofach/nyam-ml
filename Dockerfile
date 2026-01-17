# Gunakan Python 3.12 (Sesuai Laptop Kamu)
FROM python:3.12-slim

# Install library OS
RUN apt-get update && apt-get install -y \
  libgl1 \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# SOLUSI: Paksa mode Legacy biar dua model itu bisa dibaca tanpa error Flatten
ENV TF_USE_LEGACY_KERAS=1

# Install Dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy Kode
COPY . .

ENV PORT=8080

# Jalankan
CMD exec gunicorn --workers 3 --bind :$PORT --timeout 120 app:app