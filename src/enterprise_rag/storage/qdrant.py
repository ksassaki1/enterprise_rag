"""Qdrant com vetores nomeados dense/bm25 e IDs estáveis por chunk."""

from __future__ import annotations

from collections.abc import Iterator
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from enterprise_rag.models.chunk import Chunk, SparseVector
from enterprise_rag.retrieval.filters import FilterSet
from enterprise_rag.storage.base import VectorHit, VectorRecord


class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        collection: str,
        api_key: str | None = None,
        *,
        client: QdrantClient | None = None,
    ) -> None:
        self._client = client if client is not None else QdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self._dim: int | None = None

    def ensure_ready(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dimensao do vetor deve ser positiva")
        if self._dim == dim:
            return
        if self._client.collection_exists(self.collection):
            config = self._client.get_collection(self.collection).config.params
            vectors = config.vectors
            if (
                not isinstance(vectors, dict)
                or "dense" not in vectors
                or vectors["dense"].size != dim
                or vectors["dense"].distance != models.Distance.COSINE
                or "bm25" not in (config.sparse_vectors or {})
            ):
                raise ValueError("colecao Qdrant incompativel com os vetores dense/bm25; reindexe")
        else:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={"bm25": models.SparseVectorParams()},
            )
            for key in (
                "document_id",
                "file_type",
                "metadata.country",
                "metadata.project",
                "metadata.document_type",
            ):
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name=key,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
        self._dim = dim

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        dim = self._dim if self._dim is not None else len(records[0].dense)
        if dim <= 0 or any(len(record.dense) != dim for record in records):
            raise ValueError(f"todos os vetores densos devem ter dimensao {dim}")
        self.ensure_ready(dim)
        self._client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, record.chunk.chunk_id)),
                    payload=record.chunk.payload(),
                    vector={
                        "dense": record.dense,
                        "bm25": models.SparseVector(
                            indices=record.sparse.indices, values=record.sparse.values
                        ),
                    },
                )
                for record in records
            ],
            wait=True,
        )

    def search_dense(
        self, vector: list[float], limit: int, filters: FilterSet | None = None
    ) -> list[VectorHit]:
        if limit <= 0:
            return []
        response = self._client.query_points(
            collection_name=self.collection,
            query=vector,
            using="dense",
            limit=limit,
            query_filter=filters.to_qdrant() if filters else None,
            with_payload=True,
            with_vectors=False,
        )
        return [
            VectorHit(chunk=Chunk.from_payload(point.payload or {}), score=point.score)
            for point in response.points
        ]

    def search_sparse(
        self, vector: SparseVector, limit: int, filters: FilterSet | None = None
    ) -> list[VectorHit]:
        if limit <= 0 or not vector.indices:
            return []
        response = self._client.query_points(
            collection_name=self.collection,
            query=models.SparseVector(indices=vector.indices, values=vector.values),
            using="bm25",
            limit=limit,
            query_filter=filters.to_qdrant() if filters else None,
            with_payload=True,
            with_vectors=False,
        )
        return [
            VectorHit(chunk=Chunk.from_payload(point.payload or {}), score=point.score)
            for point in response.points
        ]

    @staticmethod
    def _document_filter(document_id: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))
            ]
        )

    def get_document_chunks(self, document_id: str) -> list[Chunk]:
        return sorted(
            self._scroll(self._document_filter(document_id)), key=lambda chunk: chunk.position
        )

    def delete_document(self, document_id: str) -> int:
        query_filter = self._document_filter(document_id)
        removed = self._client.count(
            collection_name=self.collection, count_filter=query_filter, exact=True
        ).count
        if removed:
            self._client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(filter=query_filter),
                wait=True,
            )
        return removed

    def iter_chunks(self) -> Iterator[Chunk]:
        return self._scroll()

    def _scroll(self, query_filter: models.Filter | None = None) -> Iterator[Chunk]:
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self.collection,
                scroll_filter=query_filter,
                offset=offset,
                limit=256,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                yield Chunk.from_payload(point.payload or {})
            if offset is None:
                return

    def count(self) -> int:
        return self._client.count(collection_name=self.collection, exact=True).count

    def clear(self) -> None:
        self._client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=models.Filter()),
            wait=True,
        )

    def close(self) -> None:
        self._client.close()
