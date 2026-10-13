"""O workflow ingere e avalia em processos separados, sem serviços externos."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests.conftest import REPO_ROOT, SAMPLE_DOCUMENTS


def test_ingestao_e_avaliacao_compartilham_indice_entre_processos(tmp_path: Path):
    env = {
        **os.environ,
        "RAG_VECTOR_BACKEND": "memory",
        "RAG_EMBEDDING_BACKEND": "hashing",
        "RAG_RERANKER_BACKEND": "lexical",
        "RAG_LLM_BACKEND": "echo",
        "RAG_LOG_LEVEL": "WARNING",
        "RAG_DATA_DIR": str(tmp_path / "data"),
        "RAG_CATALOG_PATH": str(tmp_path / "catalog.sqlite3"),
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "PYTHONIOENCODING": "utf-8",
    }

    def run(*args: str) -> None:
        result = subprocess.run(
            [sys.executable, *args],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    run("scripts/ingest.py", str(SAMPLE_DOCUMENTS))
    index_path = tmp_path / "data" / "memory_store.json"
    original_index = index_path.read_bytes()
    run("scripts/ingest.py", str(SAMPLE_DOCUMENTS))
    assert index_path.read_bytes() == original_index

    output = tmp_path / "evaluation.json"
    run(
        "scripts/evaluate.py",
        "--output",
        str(output),
        "--markdown",
        str(tmp_path / "regression.md"),
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["dataset_size"] == 16
    assert report["retrieval"]["recall@10"] >= 0.6
    assert (tmp_path / "regression.md").is_file()
