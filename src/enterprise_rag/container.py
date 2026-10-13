"""Composition root.

Um único lugar onde os componentes são construídos e ligados. A API, o CLI de
ingestão, o de avaliação e o dashboard usam exatamente o mesmo objeto -- é o
que garante que o número medido pelo `scripts/evaluate.py` descreve o sistema
que roda em produção, e não uma montagem paralela feita para o teste passar.

Todas as escolhas de backend (Qdrant vs memória, cross-encoder vs léxico,
Ollama vs extrativo) acontecem aqui, a partir do `Settings`.
"""

from __future__ import annotations

from dataclasses import dataclass

from enterprise_rag.config.logging import get_logger, setup_logging
from enterprise_rag.config.settings import Settings, get_settings
from enterprise_rag.embeddings.dense import DenseEmbedder, build_dense_embedder
from enterprise_rag.embeddings.sparse import BM25SparseEncoder, build_sparse_encoder
from enterprise_rag.generation.ollama import LLM, build_llm
from enterprise_rag.ingestion.chunker import Chunker, build_chunker
from enterprise_rag.ingestion.parsers import ParserRegistry, default_registry
from enterprise_rag.ingestion.pipeline import IngestionPipeline
from enterprise_rag.observability.tracing import init_otel
from enterprise_rag.orchestration.context_builder import ContextBuilder
from enterprise_rag.orchestration.query_analyzer import RuleBasedQueryAnalyzer
from enterprise_rag.orchestration.rag import RAGOrchestrator
from enterprise_rag.retrieval.dense import DenseRetriever
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.reranker import Reranker, build_reranker
from enterprise_rag.retrieval.sparse import SparseRetriever
from enterprise_rag.storage.base import VectorStore
from enterprise_rag.storage.catalog import Catalog
from enterprise_rag.storage.factory import build_vector_store

logger = get_logger(__name__)


@dataclass
class Container:
    settings: Settings
    store: VectorStore
    catalog: Catalog
    embedder: DenseEmbedder
    sparse_encoder: BM25SparseEncoder
    registry: ParserRegistry
    chunker: Chunker
    reranker: Reranker
    llm: LLM
    pipeline: IngestionPipeline
    orchestrator: RAGOrchestrator

    def close(self) -> None:
        try:
            self.catalog.close()
        finally:
            self.store.close()

    def describe(self) -> dict[str, object]:
        """Configuração efetiva. Vai para `/health` e para o cabeçalho dos
        relatórios de avaliação: sem isso, uma métrica não diz de qual sistema
        ela fala."""
        return {
            "vector_backend": type(self.store).__name__,
            "embedding_model": self.embedder.name,
            "embedding_dim": self.embedder.dim,
            "sparse": "bm25",
            "chunk_strategy": self.chunker.name,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "reranker": self.reranker.name,
            "llm": self.llm.name,
            "top_k_candidates": self.settings.top_k_candidates,
            "top_k_final": self.settings.top_k_final,
            "hybrid": self.settings.use_hybrid,
        }


def build_container(settings: Settings | None = None, configure_logging: bool = True) -> Container:
    settings = settings or get_settings()
    if configure_logging:
        setup_logging(settings.log_level, settings.log_json)
    if settings.otel_enabled and init_otel(settings.app_name, settings.otel_endpoint):
        logger.info("OpenTelemetry ativo", extra={"endpoint": settings.otel_endpoint})

    settings.ensure_dirs()

    store = build_vector_store(settings)
    catalog = Catalog(settings.catalog_path)
    embedder = build_dense_embedder(settings)
    sparse_encoder = build_sparse_encoder(settings)
    registry = default_registry()
    chunker = build_chunker(settings)
    reranker = build_reranker(settings)
    llm = build_llm(settings)

    store.ensure_ready(embedder.dim)

    pipeline = IngestionPipeline(
        settings=settings,
        store=store,
        catalog=catalog,
        embedder=embedder,
        sparse_encoder=sparse_encoder,
        registry=registry,
        chunker=chunker,
    )

    dense_retriever = DenseRetriever(store, embedder)
    sparse_retriever = SparseRetriever(store, sparse_encoder) if settings.use_hybrid else None
    retriever = HybridRetriever(
        dense=dense_retriever,
        sparse=sparse_retriever,
        rrf_k=settings.rrf_k,
        dense_weight=settings.dense_weight,
        sparse_weight=settings.sparse_weight,
    )

    orchestrator = RAGOrchestrator(
        settings=settings,
        retriever=retriever,
        reranker=reranker,
        context_builder=ContextBuilder(
            max_context_tokens=settings.max_context_tokens,
            max_chunks_per_document=settings.max_chunks_per_document,
            dedup_similarity=settings.dedup_similarity,
        ),
        llm=llm,
        analyzer=RuleBasedQueryAnalyzer(),
    )

    container = Container(
        settings=settings,
        store=store,
        catalog=catalog,
        embedder=embedder,
        sparse_encoder=sparse_encoder,
        registry=registry,
        chunker=chunker,
        reranker=reranker,
        llm=llm,
        pipeline=pipeline,
        orchestrator=orchestrator,
    )
    logger.info("plataforma inicializada", extra=container.describe())
    return container
