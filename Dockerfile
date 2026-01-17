# Gunakan Python Slim
FROM python:3.10-slim

# Install library OS untuk OpenCV
RUN apt-get update && apt-get install -y \
  libgl1 \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy & Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Kodingan
COPY . .

# Port Cloud Run
ENV PORT=8080

# Jalankan Gunicorn (Timeout 120s karena TensorFlow berat)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 120 app:app