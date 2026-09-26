FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Persisted data (generated API key) lives here — mount a volume in
# production so it survives container replacement.
RUN mkdir -p /app/data
VOLUME ["/app/data"]

# PaddleOCR downloads its model weights to ~/.paddlex (i.e. /root/.paddlex,
# since this container runs as root) on first use and caches them there.
# Mount a volume over it in production too, or every container restart
# re-downloads the models on first OCR request.
VOLUME ["/root/.paddlex"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
