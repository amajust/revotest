FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_CACHE_DIR=/models \
    MODEL_SIZE=small \
    DEVICE=cpu \
    COMPUTE_TYPE=float32 \
    MAX_WORKERS=2

WORKDIR /app

RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='float32', download_root='/models')"

COPY . .

RUN chown -R app:app /app /models

USER app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
