"""Reranking.

O funil é `top 30 -> reranker -> top 6`. Faz sentido porque os dois estágios
resolvem problemas diferentes:

* recuperação é **rápida e recall-oriented**: compara a consulta com vetores
  pré-computados, sem nunca ver os dois textos juntos;
* reranking é **caro e precision-oriented**: um cross-encoder lê consulta e
  chunk na mesma passagem e produz relevância de verdade.

Rodar o cross-encoder sobre a coleção inteira seria inviável; sobre 30
candidatos, custa ~200 ms de CPU e costuma ser a maior melhoria de qualidade
por linha de código do projeto inteiro. O preço é latência: por isso o
`scripts/benchmark.py` mede recall e latência lado a lado.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from enterprise_rag.config.logging import get_logger
from enterprise_rag.config.settings import Settings
from enterprise_rag.embeddings.sparse import tokenize
from enterprise_rag.models.chunk import ScoredChunk

logger = get_logger(__name__)


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]: ...


class NoopReranker:
    """Mantém a ordem da fusão. Usado como baseline no benchmark."""

    name = "none"

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        return candidates[:top_k]


class LexicalReranker:
    """Fallback determinístico: cobertura dos termos da consulta no chunk.

    Não substitui um cross-encoder, mas é reprodutível, não baixa modelo e
    dá um sinal melhor que a ordem crua da fusão -- o suficiente para CI e
    para máquinas sem torch.
    """

    name = "lexical"

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return candidates[:top_k]
        for candidate in candidates:
            chunk_tokens = tokenize(candidate.chunk.content)
            if not chunk_tokens:
                candidate.rerank_score = 0.0
                continue
            unique = set(chunk_tokens)
            coverage = len(query_tokens & unique) / len(query_tokens)
            density = sum(1 for token in chunk_tokens if token in query_tokens) / len(chunk_tokens)
            # cobertura domina; densidade só desempata textos igualmente cobertos
            candidate.rerank_score = round(coverage + 0.15 * density, 6)
        ordered = sorted(candidates, key=lambda item: item.rerank_score or 0.0, reverse=True)
        return ordered[:top_k]


class CrossEncoderReranker:
    """Cross-encoder de verdade (`sentence-transformers`), carregado sob demanda."""

    def __init__(self, model_name: str, device: str = "cpu", min_score: float = -1e9) -> None:
        from sentence_transformers import CrossEncoder

        logger.info("carregando cross-encoder", extra={"model": model_name})
        self._model = CrossEncoder(model_name, device=device)
        self.name = model_name
        self.min_score = min_score

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, candidate.chunk.content) for candidate in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.rerank_score = float(score)
        ordered = sorted(candidates, key=lambda item: item.rerank_score or 0.0, reverse=True)
        # descartar candidatos abaixo do piso é o que habilita o "nao encontrado"
        kept = [item for item in ordered if (item.rerank_score or 0.0) >= self.min_score]
        return kept[:top_k]


def build_reranker(settings: Settings) -> Reranker:
    if settings.reranker_backend == "none":
        return NoopReranker()
    if settings.reranker_backend == "cross-encoder":
        try:
            return CrossEncoderReranker(
                model_name=settings.reranker_model,
                device=settings.embedding_device,
                min_score=settings.rerank_min_score,
            )
        except ImportError:
            logger.warning(
                "sentence-transformers nao instalado; usando reranker lexical "
                "(instale com: pip install '.[ml]')"
            )
    return LexicalReranker()
