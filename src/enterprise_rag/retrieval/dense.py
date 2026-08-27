"""Recuperação densa (semântica)."""

from __future__ import annotations

from enterprise_rag.embeddings.dense import DenseEmbedder
from enterprise_rag.models.chunk import ScoredChunk
from enterprise_rag.retrieval.filters import FilterSet
from enterprise_rag.storage.base import VectorStore


class DenseRetriever:
    """Captura paráfrase: "potência instalada" recupera "capacidade de geração".

    O rank é guardado junto com o score porque o RRF usa posição, não valor
    absoluto -- comparar cosseno com BM25 diretamente não faria sentido.
    """

    name = "dense"

    def __init__(self, store: VectorStore, embedder: DenseEmbedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(
        self, query: str, limit: int, filters: FilterSet | None = None
    ) -> list[ScoredChunk]:
        vector = self.embedder.encode_query(query)
        hits = self.store.search_dense(vector, limit=limit, filters=filters)
        return [
            ScoredChunk(chunk=hit.chunk, dense_score=hit.score, dense_rank=rank)
            for rank, hit in enumerate(hits, start=1)
        ]
