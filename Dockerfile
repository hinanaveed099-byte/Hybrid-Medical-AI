# Hugging Face Spaces — Hybrid Medical AI + HVF-Net
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# OpenCV headless runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt

COPY hybrid_api.py model_loader.py ./
COPY hvf_net ./hvf_net
COPY models ./models

EXPOSE 7860

# One worker: models are heavy and load once at startup
CMD gunicorn --bind 0.0.0.0:7860 --workers 1 --threads 2 --timeout 300 hybrid_api:app
