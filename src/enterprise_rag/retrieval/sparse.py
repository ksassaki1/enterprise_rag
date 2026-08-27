"""Recuperação esparsa (lexical, BM25)."""

from __future__ import annotations

from enterprise_rag.embeddings.sparse import BM25SparseEncoder
from enterprise_rag.models.chunk import ScoredChunk
from enterprise_rag.retrieval.filters import FilterSet
from enterprise_rag.storage.base import VectorStore


class SparseRetriever:
    """Captura o que o denso perde: códigos, siglas, números de norma.

    Um embedding de 384 dimensões não tem por que separar `UHE-XPTO-01` de
    `UHE-XPTO-02`; o BM25 tem.
    """

    name = "sparse"

    def __init__(self, store: VectorStore, encoder: BM25SparseEncoder) -> None:
        self.store = store
        self.encoder = encoder

    def retrieve(
        self, query: str, limit: int, filters: FilterSet | None = None
    ) -> list[ScoredChunk]:
        vector = self.encoder.encode_query(query)
        hits = self.store.search_sparse(vector, limit=limit, filters=filters)
        return [
            ScoredChunk(chunk=hit.chunk, sparse_score=hit.score, sparse_rank=rank)
            for rank, hit in enumerate(hits, start=1)
        ]
