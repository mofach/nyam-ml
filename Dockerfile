# Gunakan Python 3.12 Slim
FROM python:3.12-slim

# Install library sistem
# PERBAIKAN: Ganti libgl1-mesa-glx jadi libgl1
RUN apt-get update && apt-get install -y \
  libgl1 \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements & install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua kode
COPY . .

# Expose port
EXPOSE 8080

# Jalankan pakai Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]