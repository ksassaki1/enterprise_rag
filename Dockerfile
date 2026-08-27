# Imagem única para API e dashboard: o que muda é o comando.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# tesseract e opcional: so entra se voce for indexar PDF digitalizado
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        tesseract-ocr \
        tesseract-ocr-por \
    && rm -rf /var/lib/apt/lists/*

# camada de dependencias separada do codigo: rebuild de codigo nao reinstala torch
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --upgrade pip && pip install ".[ml,vectordb,parsers,dashboard]"

COPY apps/ ./apps/
COPY scripts/ ./scripts/
COPY datasets/ ./datasets/

RUN mkdir -p /app/data /models

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
