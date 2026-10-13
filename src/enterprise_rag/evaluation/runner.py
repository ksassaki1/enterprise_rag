"""Execução do golden dataset.

O runner usa o mesmo `Container` da API. Isso não é detalhe: um harness de
avaliação que monta seu próprio pipeline mede um sistema que ninguém usa.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from enterprise_rag.config.logging import get_logger
from enterprise_rag.container import Container
from enterprise_rag.evaluation.generation import (
    GenerationCase,
    GenerationMetrics,
    aggregate_generation,
)
from enterprise_rag.evaluation.retrieval import RetrievalMetrics, aggregate_retrieval

logger = get_logger(__name__)


@dataclass
class GoldenItem:
    id: str
    question: str
    expected_answer: str = ""
    expected_documents: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    unanswerable: bool = False

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> GoldenItem:
        return cls(
            id=str(raw.get("id") or raw.get("question_id") or ""),
            question=str(raw["question"]),
            expected_answer=str(raw.get("expected_answer", "")),
            expected_documents=list(raw.get("expected_documents", [])),
            filters=dict(raw.get("filters", {})),
            tags=list(raw.get("tags", [])),
            unanswerable=bool(raw.get("unanswerable", False)),
        )


def load_golden_dataset(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            items.append(GoldenItem.from_json(json.loads(line)))
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"{path}:{line_number} invalido: {exc}") from exc
    return items


@dataclass
class EvaluationReport:
    created_at: str
    configuration: dict[str, Any]
    retrieval: dict[str, Any]
    generation: dict[str, Any]
    latency: dict[str, Any]
    dataset: str = ""
    dataset_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def run_evaluation(
    container: Container,
    items: list[GoldenItem],
    dataset_name: str = "golden_dataset.jsonl",
    skip_generation: bool = False,
    top_k: int | None = None,
) -> EvaluationReport:
    retrieval_rows: list[tuple[str, list[str], list[str]]] = []
    generation_cases: list[GenerationCase] = []
    latencies: list[float] = []
    retrieval_latencies: list[float] = []
    generation_latencies: list[float] = []

    # Recall e Recall@10 exigem pelo menos 10 resultados; o top_k final do
    # sistema costuma ser menor, entao a busca de avaliacao pede mais.
    search_k = max(top_k or container.settings.top_k_final, 10)

    for index, item in enumerate(items, start=1):
        search = container.orchestrator.search(
            item.question, top_k=search_k, filters=item.filters or None
        )
        retrieval_latencies.append(search.metrics.total_latency_ms)
        retrieved = [result.document_id for result in search.results]
        retrieval_rows.append((item.id, retrieved, item.expected_documents))

        if not skip_generation:
            answer = container.orchestrator.ask(item.question, filters=item.filters or None)
            generation_latencies.append(answer.metrics.generation_latency_ms)
            latencies.append(answer.metrics.total_latency_ms)
            generation_cases.append(
                GenerationCase(
                    question_id=item.id,
                    answer=answer.answer,
                    expected_answer=item.expected_answer,
                    # pergunta sem resposta na base: acertar e justamente recusar
                    answer_found=answer.answer_found if not item.unanswerable else (
                        not answer.answer_found
                    ),
                    cited_documents=[
                        source.document_id for source in answer.sources if source.used_in_answer
                    ],
                    expected_documents=item.expected_documents,
                )
            )
        else:
            latencies.append(search.metrics.total_latency_ms)

        if index % 10 == 0:
            logger.info("avaliacao em andamento", extra={"concluidas": index, "total": len(items)})

    retrieval_metrics: RetrievalMetrics = aggregate_retrieval(retrieval_rows)
    generation_metrics: GenerationMetrics = aggregate_generation(generation_cases)

    return EvaluationReport(
        created_at=datetime.now(UTC).isoformat(),
        configuration=dict(container.describe()),
        retrieval=retrieval_metrics.to_dict(),
        generation=generation_metrics.to_dict(),
        latency={
            "p50_ms": round(_percentile(latencies, 0.50), 1),
            "p95_ms": round(_percentile(latencies, 0.95), 1),
            "mean_ms": round(statistics.fmean(latencies), 1) if latencies else 0.0,
            "retrieval_p50_ms": round(_percentile(retrieval_latencies, 0.50), 1),
            "generation_p50_ms": round(_percentile(generation_latencies, 0.50), 1),
        },
        dataset=dataset_name,
        dataset_size=len(items),
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]
