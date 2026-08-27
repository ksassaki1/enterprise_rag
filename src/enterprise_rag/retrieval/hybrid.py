"""Busca híbrida com Reciprocal Rank Fusion.

Por que RRF e não soma ponderada de scores? Porque cosseno (0..1) e BM25
(0..+inf, dependente do corpus) não são comparáveis: normalizá-los exigiria
calibração por consulta e continuaria instável quando uma das listas voltasse
vazia. O RRF só usa **posição**:

    score(d) = soma_sobre_listas( peso / (k + rank(d)) )

O `k` (padrão 60, o valor do artigo original de Cormack et al.) achata a cauda,
de forma que aparecer em ambas as listas vale mais do que ser o primeiro em
apenas uma. Isso é exatamente o comportamento desejado: concordância entre
dois recuperadores independentes é o sinal mais forte que existe aqui.
"""

from __future__ import annotations

from enterprise_rag.config.logging import get_logger
from enterprise_rag.models.chunk import ScoredChunk
from enterprise_rag.retrieval.dense import DenseRetriever
from enterprise_rag.retrieval.filters import FilterSet
from enterprise_rag.retrieval.sparse import SparseRetriever

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    rankings: list[list[ScoredChunk]],
    weights: list[float] | None = None,
    k: int = 60,
    limit: int | None = None,
) -> list[ScoredChunk]:
    """Funde N rankings preservando a trilha de scores de cada estágio."""
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights e rankings precisam ter o mesmo tamanho")

    merged: dict[str, ScoredChunk] = {}
    fused: dict[str, float] = {}

    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, scored in enumerate(ranking, start=1):
            chunk_id = scored.chunk.chunk_id
            target = merged.get(chunk_id)
            if target is None:
                target = ScoredChunk(chunk=scored.chunk)
                merged[chunk_id] = target
                fused[chunk_id] = 0.0
            # cada lista contribui apenas com os campos que ela conhece
            if scored.dense_score is not None:
                target.dense_score = scored.dense_score
                target.dense_rank = scored.dense_rank or rank
            if scored.sparse_score is not None:
                target.sparse_score = scored.sparse_score
                target.sparse_rank = scored.sparse_rank or rank
            fused[chunk_id] += weight / (k + rank)

    for chunk_id, score in fused.items():
        merged[chunk_id].fusion_score = round(score, 8)

    ordered = sorted(merged.values(), key=lambda item: item.fusion_score or 0.0, reverse=True)
    return ordered[:limit] if limit else ordered


class HybridRetriever:
    """Orquestra denso + esparso e devolve uma lista única de candidatos.

    Roda os dois recuperadores mesmo quando um deles volta vazio: uma consulta
    puramente lexical ("NBR 6118") ainda se beneficia do denso para o contexto
    ao redor, e vice-versa.
    """

    name = "hybrid"

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever | None = None,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def retrieve(
        self,
        dense_query: str,
        sparse_query: str | None = None,
        limit: int = 30,
        filters: FilterSet | None = None,
    ) -> tuple[list[ScoredChunk], dict[str, int]]:
        dense_results = self.dense.retrieve(dense_query, limit=limit, filters=filters)
        sparse_results: list[ScoredChunk] = []
        if self.sparse is not None:
            sparse_results = self.sparse.retrieve(
                sparse_query or dense_query, limit=limit, filters=filters
            )

        if not sparse_results:
            fused = dense_results[:limit]
            for scored in fused:
                scored.fusion_score = scored.dense_score
        elif not dense_results:
            fused = sparse_results[:limit]
            for scored in fused:
                scored.fusion_score = scored.sparse_score
        else:
            fused = reciprocal_rank_fusion(
                [dense_results, sparse_results],
                weights=[self.dense_weight, self.sparse_weight],
                k=self.rrf_k,
                limit=limit,
            )

        stats = {
            "dense_candidates": len(dense_results),
            "sparse_candidates": len(sparse_results),
            "fused_candidates": len(fused),
        }
        return fused, stats
