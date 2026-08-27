from __future__ import annotations

import pytest

from enterprise_rag.models.chunk import Chunk, ScoredChunk
from enterprise_rag.retrieval.hybrid import reciprocal_rank_fusion


def chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        content=f"conteudo {chunk_id}",
        source="doc.md",
        filename="doc.md",
        file_type="md",
    )


def dense(ids: list[str]) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=chunk(item), dense_score=1.0 - index * 0.1, dense_rank=index + 1)
        for index, item in enumerate(ids)
    ]


def sparse(ids: list[str]) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=chunk(item), sparse_score=10.0 - index, sparse_rank=index + 1)
        for index, item in enumerate(ids)
    ]


def test_rrf_favorece_quem_aparece_nas_duas_listas():
    fused = reciprocal_rank_fusion([dense(["A", "B", "C"]), sparse(["C", "A", "D"])], k=60)
    order = [item.chunk.chunk_id for item in fused]
    assert order[:2] == ["A", "C"]
    assert set(order) == {"A", "B", "C", "D"}


def test_rrf_preserva_scores_de_cada_estagio():
    fused = reciprocal_rank_fusion([dense(["A"]), sparse(["A"])], k=60)
    item = fused[0]
    assert item.dense_score is not None
    assert item.sparse_score is not None
    assert item.fusion_score is not None


def test_rrf_respeita_pesos():
    apenas_denso = reciprocal_rank_fusion(
        [dense(["A", "B"]), sparse(["B", "A"])], weights=[10.0, 0.01], k=60
    )
    assert apenas_denso[0].chunk.chunk_id == "A"


def test_rrf_com_lista_vazia():
    fused = reciprocal_rank_fusion([dense(["A", "B"]), []], k=60)
    assert [item.chunk.chunk_id for item in fused] == ["A", "B"]


def test_rrf_valida_tamanho_dos_pesos():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([dense(["A"])], weights=[1.0, 1.0])
