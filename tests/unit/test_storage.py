"""Contratos de armazenamento e persistência dos backends restaurados."""

from __future__ import annotations

import importlib.abc
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

from enterprise_rag.models.chunk import Chunk, SparseVector
from enterprise_rag.models.document import Document, DocumentStatus
from enterprise_rag.retrieval.filters import FilterSet
from enterprise_rag.storage.base import VectorRecord, VectorStore
from enterprise_rag.storage.catalog import Catalog
from enterprise_rag.storage.factory import build_vector_store
from enterprise_rag.storage.memory import MemoryVectorStore
from tests.conftest import make_settings


def record(
    chunk_id: str = "br#0000",
    document_id: str = "br",
    dense: list[float] | None = None,
    country: str = "Brazil",
    sparse_weight: float = 2.0,
) -> VectorRecord:
    return VectorRecord(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content=f"Conteudo {chunk_id}",
            source="nota.md",
            filename="nota.md",
            file_type="md",
            metadata={"country": country},
        ),
        dense=dense if dense is not None else [2.0, 0.0],
        sparse=SparseVector(indices=[7], values=[sparse_weight]),
    )


@pytest.fixture(params=["memory", "qdrant"])
def store(request: pytest.FixtureRequest) -> Iterator[VectorStore]:
    instance: VectorStore
    if request.param == "memory":
        instance = MemoryVectorStore()
    else:
        qdrant = pytest.importorskip("qdrant_client")
        from enterprise_rag.storage.qdrant import QdrantVectorStore

        instance = QdrantVectorStore(
            url="http://localhost:6333", collection="test", client=qdrant.QdrantClient(":memory:")
        )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Payload indexes have no effect.*")
        instance.ensure_ready(2)
    try:
        yield instance
    finally:
        instance.close()


def test_buscas_densa_e_esparsa_preservam_scores_e_filtros(store: VectorStore):
    store.upsert(
        [
            record(),
            record("cl#0000", "cl", dense=[1.0, 1.0], country="Chile", sparse_weight=3.0),
        ]
    )
    dense = store.search_dense([1.0, 0.0], limit=2)
    assert [hit.chunk.document_id for hit in dense] == ["br", "cl"]
    assert dense[0].score == pytest.approx(1.0)
    assert dense[1].score == pytest.approx(2**-0.5)

    sparse = store.search_sparse(SparseVector(indices=[7], values=[4.0]), limit=2)
    assert [hit.chunk.document_id for hit in sparse] == ["cl", "br"]
    assert [hit.score for hit in sparse] == pytest.approx([12.0, 8.0])

    filters = FilterSet.from_dict({"country": "Chile"})
    assert store.search_dense([1.0, 0.0], limit=1, filters=filters)[0].chunk.document_id == "cl"
    filters = FilterSet.from_dict({"country": "Brazil"})
    assert (
        store.search_sparse(SparseVector(indices=[7], values=[4.0]), limit=1, filters=filters)[
            0
        ].chunk.document_id
        == "br"
    )


def test_upsert_substitui_chunk_e_remocao_preserva_outro_documento(store: VectorStore):
    store.upsert([record(), record("cl#0000", "cl", country="Chile")])
    updated = record(dense=[0.0, 1.0])
    updated.chunk.content = "Conteudo atualizado"
    store.upsert([updated])
    assert store.count() == 2
    assert store.get_document_chunks("br")[0].content == "Conteudo atualizado"
    assert store.delete_document("br") == 1
    assert store.delete_document("br") == 0
    assert store.get_document_chunks("br") == []
    assert [chunk.document_id for chunk in store.iter_chunks()] == ["cl"]
    store.clear()
    assert store.count() == 0


def test_consulta_sem_termos_em_comum_nao_gera_hit_esparso(store: VectorStore):
    store.upsert([record()])
    assert store.search_sparse(SparseVector(), limit=2) == []
    assert store.search_sparse(SparseVector(indices=[8], values=[1.0]), limit=2) == []
    assert store.search_dense([1.0, 0.0], limit=0) == []
    assert store.search_sparse(SparseVector(indices=[7], values=[1.0]), limit=0) == []


def test_dimensao_incompativel_nao_substitui_indice(store: VectorStore):
    store.upsert([record()])
    with pytest.raises(ValueError):
        store.ensure_ready(3)
    with pytest.raises(ValueError):
        store.upsert([record(dense=[1.0, 0.0, 0.0])])
    assert store.count() == 1


def test_iteracao_percorre_todos_os_chunks(store: VectorStore):
    store.upsert([record(f"br#{index:04d}") for index in range(270)])
    assert len(list(store.iter_chunks())) == 270
    assert len(store.get_document_chunks("br")) == 270
    assert store.delete_document("br") == 270


def test_indice_em_memoria_sobrevive_a_reabertura_e_remocoes(tmp_path: Path):
    path = tmp_path / "nested" / "index.json"
    original = MemoryVectorStore(path)
    original.upsert([record(), record("cl#0000", "cl", country="Chile")])
    restored = MemoryVectorStore(path)
    assert restored.count() == 2
    assert restored.search_dense([1.0, 0.0], limit=1)[0].chunk.chunk_id == "br#0000"
    assert restored.search_sparse(SparseVector(indices=[7], values=[1.0]), limit=1)
    assert restored.get_document_chunks("br")[0].metadata == {"country": "Brazil"}
    restored.delete_document("br")
    assert MemoryVectorStore(path).count() == 1
    restored.clear()
    assert MemoryVectorStore(path).count() == 0


def test_backend_memory_nao_precisa_do_cliente_qdrant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class BlockQdrant(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "qdrant_client" or fullname.startswith("qdrant_client."):
                raise ModuleNotFoundError("qdrant-client indisponivel")
            return None

    for name in list(sys.modules):
        if name == "qdrant_client" or name.startswith(
            ("qdrant_client.", "enterprise_rag.storage.qdrant")
        ):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [BlockQdrant(), *sys.meta_path])
    instance = build_vector_store(make_settings(tmp_path))
    instance.upsert([record()])
    assert isinstance(instance, MemoryVectorStore)
    assert instance.count() == 1
    instance.close()


def document(status: DocumentStatus = DocumentStatus.INDEXED) -> Document:
    return Document(
        document_id="br/nota",
        filename="nota.md",
        source="br/nota.md",
        file_type="md",
        content_sha256="abc",
        status=status,
        chunk_count=2,
        metadata={"country": "Brazil"},
    )


def test_catalogo_persiste_documentos_versoes_e_exclusoes(tmp_path: Path):
    path = tmp_path / "catalog.sqlite3"
    first = Catalog(path)
    try:
        first.upsert_document(document())
        assert first.is_unchanged("br/nota", "abc")
        assert not first.is_unchanged("br/nota", "novo")
        updated = document().model_copy(update={"version": 2, "content_sha256": "novo"})
        first.upsert_document(updated)
    finally:
        first.close()
    restored = Catalog(path)
    try:
        assert restored.get_document("br/nota") == updated
        assert restored.stats()["documents"] == 1
        assert restored.stats()["chunks"] == 2
        assert len(restored.list_documents(limit=1)) == 1
        assert not restored.list_documents(offset=1)
        restored.delete_document("br/nota")
        assert restored.get_document("br/nota") is None
        assert restored.stats()["documents"] == 0
    finally:
        restored.close()


def test_documento_com_falha_nao_e_tratado_como_inalterado(tmp_path: Path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    try:
        catalog.upsert_document(document(DocumentStatus.FAILED))
        assert not catalog.is_unchanged("br/nota", "abc")
        assert catalog.stats()["failed"] == 1
        assert catalog.stats()["chunks"] == 0
    finally:
        catalog.close()


def test_estatisticas_de_consultas_separam_busca_de_resposta_nao_encontrada(tmp_path: Path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    try:
        assert catalog.query_stats() == {"total": 0, "avg_latency_ms": 0.0, "not_found_rate": 0.0}
        for index, found in enumerate([None, True, False]):
            catalog.log_query(
                trace_id=str(index),
                endpoint="search" if found is None else "ask",
                query="Qual a altura?",
                answer_found=found,
                chunks_used=2,
                latency_ms=10.0 * (index + 1),
                detail={"filters": {"country": "Brazil"}},
            )
        assert catalog.query_stats() == {"total": 3, "avg_latency_ms": 20.0, "not_found_rate": 0.5}
        recent = catalog.recent_queries(limit=1)
        assert recent[0]["trace_id"] == "2"
        assert recent[0]["answer_found"] is False
        assert recent[0]["detail"] == {"filters": {"country": "Brazil"}}
    finally:
        catalog.close()
