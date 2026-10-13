"""Busca exata em memória, com persistência JSON entre processos do CLI."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from enterprise_rag.models.chunk import Chunk, SparseVector
from enterprise_rag.retrieval.filters import FilterSet
from enterprise_rag.storage.base import VectorHit, VectorRecord


class MemoryVectorStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._dim: int | None = None
        self._records: dict[str, VectorRecord] = {}
        self._lock = threading.RLock()
        if path is not None and path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            self._dim = payload["dim"]
            for item in payload["records"]:
                record = VectorRecord(
                    chunk=Chunk.from_payload(item["chunk"]),
                    dense=item["dense"],
                    sparse=SparseVector.model_validate(item["sparse"]),
                )
                if len(record.dense) != self._dim:
                    raise ValueError("dimensao inconsistente no indice persistido")
                self._records[record.chunk.chunk_id] = record

    def ensure_ready(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dimensao do vetor deve ser positiva")
        with self._lock:
            if self._records and self._dim != dim:
                raise ValueError(
                    f"indice usa dimensao {self._dim}, mas o embedder usa {dim}; reindexe a base"
                )
            self._dim = dim

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        with self._lock:
            dim = self._dim if self._dim is not None else len(records[0].dense)
            if dim <= 0 or any(len(record.dense) != dim for record in records):
                raise ValueError(f"todos os vetores densos devem ter dimensao {dim}")
            self.ensure_ready(dim)
            for record in records:
                self._records[record.chunk.chunk_id] = record
            self._save()

    def search_dense(
        self, vector: list[float], limit: int, filters: FilterSet | None = None
    ) -> list[VectorHit]:
        if limit <= 0:
            return []
        with self._lock:
            if self._dim is not None and len(vector) != self._dim:
                raise ValueError(f"consulta deve ter dimensao {self._dim}")
            query = np.asarray(vector, dtype=np.float32)
            query_norm = float(np.linalg.norm(query))
            hits = []
            for record in self._records.values():
                if filters and not filters.matches(record.chunk.payload()):
                    continue
                dense = np.asarray(record.dense, dtype=np.float32)
                denominator = query_norm * float(np.linalg.norm(dense))
                score = float(np.dot(query, dense)) / denominator if denominator else 0.0
                hits.append(VectorHit(chunk=record.chunk, score=score))
            return self._top(hits, limit)

    def search_sparse(
        self, vector: SparseVector, limit: int, filters: FilterSet | None = None
    ) -> list[VectorHit]:
        if limit <= 0 or not vector.indices:
            return []
        with self._lock:
            hits = []
            for record in self._records.values():
                if filters and not filters.matches(record.chunk.payload()):
                    continue
                score = vector.dot(record.sparse)
                if score > 0:
                    hits.append(VectorHit(chunk=record.chunk, score=score))
            return self._top(hits, limit)

    @staticmethod
    def _top(hits: list[VectorHit], limit: int) -> list[VectorHit]:
        return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.chunk_id))[:limit]

    def get_document_chunks(self, document_id: str) -> list[Chunk]:
        return sorted(
            (chunk for chunk in self.iter_chunks() if chunk.document_id == document_id),
            key=lambda chunk: chunk.position,
        )

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            ids = [
                key
                for key, record in self._records.items()
                if record.chunk.document_id == document_id
            ]
            for key in ids:
                del self._records[key]
            if ids:
                self._save()
            return len(ids)

    def iter_chunks(self) -> Iterator[Chunk]:
        with self._lock:
            return iter([record.chunk for record in self._records.values()])

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._save()

    def close(self) -> None:
        pass

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self._dim,
            "records": [
                {
                    "chunk": record.chunk.payload(),
                    "dense": record.dense,
                    "sparse": record.sparse.model_dump(),
                }
                for record in self._records.values()
            ],
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.path)
