FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./
COPY yolo-person-ball-v1.pt ./
COPY tactical_models.pkl ./

RUN pip install --upgrade pip \
    && pip install .

RUN useradd \
        --create-home \
        --uid 10001 \
        --shell /usr/sbin/nologin \
        appuser \
    && mkdir -p \
        /tmp/football_platform/uploads \
        /tmp/football_platform/outputs \
        /tmp/Ultralytics \
    && chown -R appuser:appuser \
        /app \
        /tmp/football_platform \
        /tmp/Ultralytics

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
