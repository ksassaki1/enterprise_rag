"""Métricas de recuperação.

Separadas das métricas de geração de propósito. Recall@K responde "o chunk
certo chegou até o LLM?"; se ele não chegou, nenhum prompt salva a resposta, e
mexer no prompt é perder tempo. É a primeira coisa a medir e a última a
regredir sem que alguém perceba.

* **Recall@K** -- fração dos documentos esperados presentes no top-K. É a
  métrica que governa o `top_k_candidates`.
* **Precision@K** -- fração do top-K que era relevante. Governa quanto lixo
  entra no contexto.
* **MRR** -- posição do primeiro acerto. Importa porque o reranker e o
  orçamento de contexto favorecem o topo.
* **nDCG@K** -- ganho descontado por posição, para quando há mais de um
  documento correto.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _normalize(value: str) -> str:
    return value.strip().lower()


def matches(retrieved: str, expected: set[str]) -> bool:
    """Casamento tolerante: id completo, nome de arquivo ou sufixo de caminho.

    O golden dataset é escrito à mão, com nomes de arquivo; o sistema devolve
    `document_id` derivado do caminho. Exigir igualdade exata só produziria
    falso negativo de avaliação.
    """
    candidate = _normalize(retrieved)
    for item in expected:
        target = _normalize(item)
        if candidate == target or candidate.endswith(target) or target.endswith(candidate):
            return True
    return False


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 1.0
    top = retrieved[:k]
    hits = sum(1 for item in expected if any(matches(entry, {item}) for entry in top))
    return hits / len(expected)


def precision_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    expected_set = set(expected)
    hits = sum(1 for entry in top if matches(entry, expected_set))
    return hits / len(top)


def reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    expected_set = set(expected)
    for position, entry in enumerate(retrieved, start=1):
        if matches(entry, expected_set):
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 1.0
    expected_set = set(expected)
    gains = [1.0 if matches(entry, expected_set) else 0.0 for entry in retrieved[:k]]
    dcg = sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, start=1))
    ideal = sum(
        1.0 / math.log2(position + 1)
        for position in range(1, min(len(expected), k) + 1)
    )
    return dcg / ideal if ideal else 0.0


@dataclass
class RetrievalMetrics:
    questions: int = 0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    misses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "questions": self.questions,
            "recall@1": round(self.recall_at_1, 4),
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "precision@5": round(self.precision_at_5, 4),
            "mrr": round(self.mrr, 4),
            "ndcg@10": round(self.ndcg_at_10, 4),
            "misses": self.misses,
        }


def aggregate_retrieval(
    per_question: list[tuple[str, list[str], list[str]]],
) -> RetrievalMetrics:
    """Recebe (id_da_pergunta, documentos_recuperados, documentos_esperados)."""
    if not per_question:
        return RetrievalMetrics()

    total = len(per_question)
    metrics = RetrievalMetrics(questions=total)
    for question_id, retrieved, expected in per_question:
        metrics.recall_at_1 += recall_at_k(retrieved, expected, 1)
        metrics.recall_at_5 += recall_at_k(retrieved, expected, 5)
        metrics.recall_at_10 += recall_at_k(retrieved, expected, 10)
        metrics.precision_at_5 += precision_at_k(retrieved, expected, 5)
        metrics.mrr += reciprocal_rank(retrieved, expected)
        metrics.ndcg_at_10 += ndcg_at_k(retrieved, expected, 10)
        if recall_at_k(retrieved, expected, 10) == 0.0:
            metrics.misses.append(question_id)

    for attribute in (
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "precision_at_5",
        "mrr",
        "ndcg_at_10",
    ):
        setattr(metrics, attribute, getattr(metrics, attribute) / total)
    return metrics
