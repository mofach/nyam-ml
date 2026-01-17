# Gunakan Python base image
FROM python:3.12.7

# Atur working directory di dalam container
WORKDIR /app

# Copy requirements file secara terpisah untuk caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file proyek ke container
COPY . /app

# Expose port Flask
EXPOSE 8080

# Jalankan aplikasi dengan Gunicorn
CMD ["gunicorn", "--workers=3", "--bind=0.0.0.0:8080", "app:app"]