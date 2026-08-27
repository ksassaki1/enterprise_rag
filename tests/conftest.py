"""Fixtures compartilhadas.

Os testes rodam com os backends determinísticos (`memory`, `hashing`,
`lexical`, `echo`). É essa combinação que permite ter um portão de regressão
de verdade em CI gratuito: sem GPU, sem download de modelo, sem serviço
externo -- e com o mesmo código de produção.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from enterprise_rag.config.settings import Settings
from enterprise_rag.container import Container, build_container

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCUMENTS = REPO_ROOT / "datasets" / "sample_documents"
GOLDEN_DATASET = REPO_ROOT / "datasets" / "golden_dataset.jsonl"


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "environment": "test",
        "log_level": "WARNING",
        "data_dir": tmp_path,
        "catalog_path": tmp_path / "catalog.sqlite3",
        "vector_backend": "memory",
        "embedding_backend": "hashing",
        "dense_dim": 256,
        "reranker_backend": "lexical",
        "llm_backend": "echo",
        "chunk_strategy": "structural",
        "chunk_size": 300,
        "chunk_overlap": 40,
        "top_k_candidates": 30,
        "top_k_final": 6,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(scope="session")
def sample_documents() -> Path:
    return SAMPLE_DOCUMENTS


@pytest.fixture(scope="session")
def golden_dataset() -> Path:
    return GOLDEN_DATASET


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture()
def container(settings: Settings) -> Iterator[Container]:
    instance = build_container(settings, configure_logging=False)
    yield instance
    instance.close()


@pytest.fixture()
def indexed_container(container: Container, sample_documents: Path) -> Container:
    container.pipeline.ingest_path(sample_documents)
    return container
