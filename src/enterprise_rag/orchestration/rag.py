"""O orquestrador.

Aqui os estágios viram um fluxo único e observável:

    query -> plano -> retrieval hibrido -> rerank -> contexto -> LLM -> resposta

Duas decisões de projeto merecem destaque:

* **`search()` e `ask()` são caminhos separados.** `search()` para no
  reranking. Isso permite responder, diante de uma resposta ruim, a única
  pergunta que importa no diagnóstico: o chunk certo foi recuperado? Se sim,
  o problema é do gerador; se não, é do retrieval. Sem essa separação, todo
  bug vira palpite.
* **cada estágio é um span.** O trace sai junto com a resposta, então o custo
  de latência é atribuível sem instrumentação extra.
"""

from __future__ import annotations

from enterprise_rag.config.logging import get_logger
from enterprise_rag.config.settings import Settings
from enterprise_rag.generation.ollama import LLM, LLMError
from enterprise_rag.generation.prompts import (
    NOT_FOUND_MARKER,
    SYSTEM_PROMPT,
    build_user_prompt,
    extract_citations,
    is_not_found,
    strip_marker,
)
from enterprise_rag.models.chunk import ScoredChunk
from enterprise_rag.models.response import (
    AskResponse,
    Citation,
    MetricsBlock,
    QueryPlan,
    RetrievalStats,
    SearchResponse,
    SearchResult,
)
from enterprise_rag.observability.metrics import METRICS
from enterprise_rag.observability.tracing import Trace, new_trace
from enterprise_rag.orchestration.context_builder import ContextBuilder
from enterprise_rag.orchestration.query_analyzer import RuleBasedQueryAnalyzer
from enterprise_rag.retrieval.filters import FilterSet
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.reranker import Reranker

logger = get_logger(__name__)

EXCERPT_CHARS = 320


class RAGOrchestrator:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        reranker: Reranker,
        context_builder: ContextBuilder,
        llm: LLM,
        analyzer: RuleBasedQueryAnalyzer | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.llm = llm
        self.analyzer = analyzer or RuleBasedQueryAnalyzer()

    # ── recuperação pura ────────────────────────────────────────────────
    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
        rerank: bool = True,
    ) -> SearchResponse:
        top_k = top_k or self.settings.top_k_final
        with new_trace("search", query=query) as trace:
            plan, candidates, stats = self._retrieve(trace, query, filters)

            if rerank:
                with trace.span("reranking", candidatos=len(candidates)):
                    candidates = self.reranker.rerank(plan.original_query, candidates, top_k)
                stats.reranked = len(candidates)
            else:
                candidates = candidates[:top_k]

            METRICS.requests_total.inc(endpoint="search", status="ok")
            METRICS.latency_seconds.observe(trace.total_ms / 1000, endpoint="search")
            METRICS.retrieved_chunks.observe(len(candidates), endpoint="search")
            METRICS.observe_trace("search", _span_durations(trace))

            return SearchResponse(
                trace_id=trace.trace_id,
                query=query,
                plan=plan,
                results=[_to_result(candidate) for candidate in candidates],
                retrieval=stats,
                metrics=_metrics_from(trace),
            )

    # ── RAG completo ────────────────────────────────────────────────────
    def ask(
        self,
        question: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> AskResponse:
        top_k = top_k or self.settings.top_k_final
        with new_trace("ask", query=question) as trace:
            plan, candidates, stats = self._retrieve(trace, question, filters)

            with trace.span("reranking", candidatos=len(candidates)):
                candidates = self.reranker.rerank(plan.original_query, candidates, top_k)
            stats.reranked = len(candidates)

            with trace.span("context_build"):
                context = self.context_builder.build(candidates)

            with trace.span("generation", model=self.llm.name) as span:
                if context.is_empty:
                    # nada recuperado: gerar aqui so gastaria tokens para produzir
                    # uma recusa que ja sabemos ser a resposta correta
                    span.attributes["skipped"] = True
                    answer = NOT_FOUND_MARKER
                else:
                    try:
                        answer = self.llm.chat(
                            SYSTEM_PROMPT, build_user_prompt(question, context.text)
                        )
                    except LLMError as exc:
                        METRICS.errors_total.inc(stage="generation")
                        METRICS.requests_total.inc(endpoint="ask", status="error")
                        span.error = str(exc)
                        raise

            not_found = context.is_empty or is_not_found(answer)
            cited = set(extract_citations(answer))
            sources = [
                _to_citation(index, candidate, used=index in cited)
                for index, candidate in enumerate(context.chunks, start=1)
            ]

            if not_found:
                answer = strip_marker(answer)
                METRICS.answer_not_found_total.inc()
            METRICS.requests_total.inc(endpoint="ask", status="ok")
            METRICS.latency_seconds.observe(trace.total_ms / 1000, endpoint="ask")
            METRICS.retrieved_chunks.observe(len(context.chunks), endpoint="ask")
            METRICS.context_tokens.observe(context.stats.context_tokens)
            METRICS.observe_trace("ask", _span_durations(trace))

            logger.info(
                "pergunta respondida",
                extra={
                    "chunks": len(context.chunks),
                    "encontrado": not not_found,
                    "latencia_ms": round(trace.total_ms, 1),
                },
            )

            return AskResponse(
                trace_id=trace.trace_id,
                question=question,
                answer=answer,
                answer_found=not not_found,
                sources=sources,
                plan=plan,
                retrieval=stats,
                context=context.stats,
                metrics=_metrics_from(trace),
                model=self.llm.name,
            )

    # ── estágios compartilhados ─────────────────────────────────────────
    def _retrieve(
        self, trace: Trace, query: str, filters: dict | None
    ) -> tuple[QueryPlan, list[ScoredChunk], RetrievalStats]:
        with trace.span("query_processing"):
            plan = self.analyzer.analyze(query)
            # filtro vindo da API é explícito: sempre vence a inferência
            merged = {**plan.filters, **(filters or {})}
            plan.filters = merged
            filter_set = FilterSet.from_dict(merged)

        with trace.span("retrieval", top_k=self.settings.top_k_candidates):
            candidates, counts = self.retriever.retrieve(
                dense_query=plan.dense_query,
                sparse_query=plan.sparse_query,
                limit=self.settings.top_k_candidates,
                filters=filter_set or None,
            )

        stats = RetrievalStats(
            dense_candidates=counts["dense_candidates"],
            sparse_candidates=counts["sparse_candidates"],
            fused_candidates=counts["fused_candidates"],
            filters_applied=merged,
            strategy="hybrid" if self.settings.use_hybrid else "dense",
        )
        return plan, candidates, stats


def _to_result(candidate: ScoredChunk) -> SearchResult:
    chunk = candidate.chunk
    return SearchResult(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document=chunk.filename,
        content=chunk.content,
        page=chunk.page,
        section=chunk.section,
        dense_score=candidate.dense_score,
        sparse_score=candidate.sparse_score,
        fusion_score=candidate.fusion_score,
        rerank_score=candidate.rerank_score,
        metadata=chunk.metadata,
    )


def _to_citation(index: int, candidate: ScoredChunk, used: bool) -> Citation:
    chunk = candidate.chunk
    excerpt = chunk.content[:EXCERPT_CHARS]
    if len(chunk.content) > EXCERPT_CHARS:
        excerpt += "..."
    return Citation(
        document=chunk.filename,
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        page=chunk.page,
        section=chunk.section,
        score=round(candidate.score, 6),
        excerpt=excerpt,
        used_in_answer=used,
    )


def _span_durations(trace: Trace) -> dict[str, float]:
    durations: dict[str, float] = {}
    for span in trace.spans:
        durations[span.name] = durations.get(span.name, 0.0) + span.duration_ms
    return durations


def _metrics_from(trace: Trace) -> MetricsBlock:
    return MetricsBlock(
        total_latency_ms=round(trace.total_ms, 2),
        query_processing_ms=round(trace.duration_of("query_processing"), 2),
        retrieval_latency_ms=round(trace.duration_of("retrieval"), 2),
        rerank_latency_ms=round(trace.duration_of("reranking"), 2),
        context_build_ms=round(trace.duration_of("context_build"), 2),
        generation_latency_ms=round(trace.duration_of("generation"), 2),
    )
