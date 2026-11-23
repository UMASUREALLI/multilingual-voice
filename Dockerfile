# Dockerfile for Multilingual Voice AI Call Translation Backend
# Compatible with Hugging Face Spaces, Render, AWS, and Docker Compose
FROM python:3.10-slim

# Install system audio and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up user with UID 1000 (Required for Hugging Face Spaces security)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Copy requirements and install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt elevenlabs

# Copy backend code
COPY --chown=user backend/ ./backend/

# Ensure correct permissions for model cache downloads
RUN mkdir -p $HOME/.cache && chown -R user:user $HOME/.cache

USER user
WORKDIR $HOME/app/backend

# Expose port (Hugging Face default is 7860)
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Start Uvicorn production server
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
