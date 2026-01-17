# Gunakan Python 3.12 (Sesuai referensi laptop kamu)
FROM python:3.12-slim

# Install library OS untuk OpenCV (Wajib ada di Linux server)
RUN apt-get update && apt-get install -y \
  libgl1 \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV TF_USE_LEGACY_KERAS=1
# Copy requirements & Install
COPY requirements.txt .
# Upgrade pip dulu biar aman
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy Kodingan
COPY . .

# Port
ENV PORT=8080

# Jalankan (Sesuai referensi kamu, workers=3 oke)
CMD exec gunicorn --workers 3 --bind :$PORT --timeout 120 app:app