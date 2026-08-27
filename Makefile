# Atalhos do dia a dia. Tudo funciona sem Docker usando os backends locais.

PY ?= python

.PHONY: help install install-dev lint format typecheck test test-unit ingest evaluate baseline benchmark api dashboard up down logs clean

help:
	@echo "install       instala o pacote (nucleo)"
	@echo "install-dev   instala com extras de desenvolvimento e ML"
	@echo "lint          ruff check"
	@echo "format        ruff format"
	@echo "typecheck     mypy"
	@echo "test          pytest (unit + integration)"
	@echo "ingest        indexa datasets/sample_documents"
	@echo "evaluate      roda o golden dataset"
	@echo "baseline      regrava reports/baseline.json"
	@echo "benchmark     compara estrategias de chunking/retrieval"
	@echo "api           sobe a API local"
	@echo "dashboard     sobe o Streamlit"
	@echo "up / down     docker compose"

install:
	$(PY) -m pip install -e .

install-dev:
	$(PY) -m pip install -e ".[dev,ml,vectordb,parsers,dashboard]"

lint:
	ruff check src apps scripts tests

format:
	ruff format src apps scripts tests

typecheck:
	mypy src

test:
	pytest

test-unit:
	pytest tests/unit

ingest:
	$(PY) scripts/ingest.py ./datasets/sample_documents

evaluate:
	$(PY) scripts/evaluate.py --baseline reports/baseline.json

baseline:
	$(PY) scripts/evaluate.py --update-baseline

benchmark:
	$(PY) scripts/benchmark.py

api:
	uvicorn apps.api.main:app --reload --port 8000

dashboard:
	streamlit run apps/dashboard/app.py

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api

clean:
	rm -rf data/*.json data/*.sqlite3 data/uploads .pytest_cache .ruff_cache .mypy_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
