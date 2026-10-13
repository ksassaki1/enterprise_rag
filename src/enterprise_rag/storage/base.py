"""Contratos compartilhados pelos backends vetoriais."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from enterprise_rag.models.chunk import Chunk, SparseVector
from enterprise_rag.retrieval.filters import FilterSet


@dataclass
class VectorRecord:
    chunk: Chunk
    dense: list[float]
    sparse: SparseVector


@dataclass
class VectorHit:
    chunk: Chunk
    score: float


@runtime_checkable
class VectorStore(Protocol):
    def ensure_ready(self, dim: int) -> None: ...

    def upsert(self, records: list[VectorRecord]) -> None: ...

    def search_dense(
        self, vector: list[float], limit: int, filters: FilterSet | None = None
    ) -> list[VectorHit]: ...

    def search_sparse(
        self, vector: SparseVector, limit: int, filters: FilterSet | None = None
    ) -> list[VectorHit]: ...

    def get_document_chunks(self, document_id: str) -> list[Chunk]: ...

    def delete_document(self, document_id: str) -> int: ...

    def iter_chunks(self) -> Iterator[Chunk]: ...

    def count(self) -> int: ...

    def clear(self) -> None: ...

    def close(self) -> None: ...
