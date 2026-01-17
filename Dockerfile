# Gunakan Python 3.10 (Paling stabil untuk TensorFlow 2.15)
FROM python:3.10-slim

# Install library OS wajib untuk OpenCV
RUN apt-get update && apt-get install -y \
  libgl1 \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements dan install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Port Cloud Run
ENV PORT=8080

# Jalankan Gunicorn (Timeout 120s biar gak mati pas download model)
CMD exec gunicorn --workers 3 --bind :$PORT --timeout 120 app:app