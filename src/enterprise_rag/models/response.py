"""Contratos de saída da API.

Regra do projeto: nenhuma resposta é apenas um campo `answer`.
Toda resposta carrega proveniência (citações) e observabilidade (latência por
estágio), porque é isso que separa um sistema depurável de uma demo.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    document: str
    document_id: str
    chunk_id: str
    page: int | None = None
    section: str | None = None
    score: float = 0.0
    excerpt: str = ""
    used_in_answer: bool = False


class RetrievalStats(BaseModel):
    dense_candidates: int = 0
    sparse_candidates: int = 0
    fused_candidates: int = 0
    reranked: int = 0
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    strategy: str = "hybrid"


class ContextStats(BaseModel):
    chunks_in_context: int = 0
    context_tokens: int = 0
    documents_in_context: int = 0
    dropped_duplicates: int = 0
    dropped_by_budget: int = 0
    dropped_by_document_cap: int = 0


class MetricsBlock(BaseModel):
    total_latency_ms: float = 0.0
    query_processing_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    context_build_ms: float = 0.0
    generation_latency_ms: float = 0.0


class QueryPlan(BaseModel):
    """O que o sistema entendeu da pergunta antes de buscar qualquer coisa."""

    original_query: str
    dense_query: str
    sparse_query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    intent: str = "qa"
    detected_entities: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    document: str
    content: str
    page: int | None = None
    section: str | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    trace_id: str
    query: str
    plan: QueryPlan
    results: list[SearchResult] = Field(default_factory=list)
    retrieval: RetrievalStats = Field(default_factory=RetrievalStats)
    metrics: MetricsBlock = Field(default_factory=MetricsBlock)


class AskResponse(BaseModel):
    trace_id: str
    question: str
    answer: str
    answer_found: bool = True
    sources: list[Citation] = Field(default_factory=list)
    plan: QueryPlan | None = None
    retrieval: RetrievalStats = Field(default_factory=RetrievalStats)
    context: ContextStats = Field(default_factory=ContextStats)
    metrics: MetricsBlock = Field(default_factory=MetricsBlock)
    model: str = ""
