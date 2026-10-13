"""Seleção do backend sem importar dependências opcionais no caminho memory."""

from __future__ import annotations

from enterprise_rag.config.settings import Settings
from enterprise_rag.storage.base import VectorStore
from enterprise_rag.storage.memory import MemoryVectorStore


def build_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_backend == "memory":
        return MemoryVectorStore(settings.memory_store_path)

    from enterprise_rag.storage.qdrant import QdrantVectorStore

    return QdrantVectorStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        api_key=settings.qdrant_api_key,
    )
