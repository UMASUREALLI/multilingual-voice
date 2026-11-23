# Dockerfile for Multilingual Voice AI Call Translation Backend
FROM python:3.10-slim

# Install system audio and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt elevenlabs

# Copy backend code
COPY backend/ ./backend/

# Expose FastAPI port
EXPOSE 8000

WORKDIR /app/backend

# Healthcheck to ensure container is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn production server
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
