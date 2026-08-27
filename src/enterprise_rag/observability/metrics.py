"""Métricas no formato Prometheus, sem dependência externa.

Um registry de ~100 linhas evita arrastar `prometheus_client` só para expor
sete séries. O endpoint `GET /metrics` renderiza exatamente o text format que o
Prometheus faz scrape, então a migração para um stack real é trocar o import.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

LabelKey = tuple[tuple[str, str], ...]

DEFAULT_BUCKETS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)


def _labels_to_key(labels: dict[str, str] | None) -> LabelKey:
    if not labels:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


def _render_labels(key: LabelKey, extra: tuple[str, str] | None = None) -> str:
    items = list(key)
    if extra:
        items.append(extra)
    if not items:
        return ""
    inner = ",".join(f'{name}="{_escape(value)}"' for name, value in items)
    return "{" + inner + "}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\n")


@dataclass
class Counter:
    name: str
    help: str
    values: dict[LabelKey, float] = field(default_factory=lambda: defaultdict(float))

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        self.values[_labels_to_key(labels)] += amount

    def render(self) -> Iterable[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} counter"
        if not self.values:
            yield f"{self.name} 0"
        for key, value in sorted(self.values.items()):
            yield f"{self.name}{_render_labels(key)} {value:g}"


@dataclass
class Gauge:
    name: str
    help: str
    values: dict[LabelKey, float] = field(default_factory=lambda: defaultdict(float))

    def set(self, value: float, **labels: str) -> None:
        self.values[_labels_to_key(labels)] = value

    def render(self) -> Iterable[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} gauge"
        for key, value in sorted(self.values.items()):
            yield f"{self.name}{_render_labels(key)} {value:g}"


@dataclass
class Histogram:
    name: str
    help: str
    buckets: tuple[float, ...] = DEFAULT_BUCKETS
    counts: dict[LabelKey, list[int]] = field(default_factory=dict)
    sums: dict[LabelKey, float] = field(default_factory=lambda: defaultdict(float))
    totals: dict[LabelKey, int] = field(default_factory=lambda: defaultdict(int))
    observations: dict[LabelKey, list[float]] = field(default_factory=lambda: defaultdict(list))

    def observe(self, value: float, **labels: str) -> None:
        key = _labels_to_key(labels)
        bucket_counts = self.counts.setdefault(key, [0] * len(self.buckets))
        for index, upper in enumerate(self.buckets):
            if value <= upper:
                bucket_counts[index] += 1
        self.sums[key] += value
        self.totals[key] += 1
        series = self.observations[key]
        series.append(value)
        if len(series) > 2000:
            del series[: len(series) - 2000]

    def quantile(self, q: float, **labels: str) -> float:
        series = sorted(self.observations.get(_labels_to_key(labels), []))
        if not series:
            return 0.0
        index = min(len(series) - 1, max(0, int(round(q * (len(series) - 1)))))
        return series[index]

    def render(self) -> Iterable[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} histogram"
        for key, bucket_counts in sorted(self.counts.items()):
            cumulative = 0
            for upper, count in zip(self.buckets, bucket_counts, strict=True):
                cumulative = count
                yield f"{self.name}_bucket{_render_labels(key, ('le', str(upper)))} {cumulative}"
            yield f"{self.name}_bucket{_render_labels(key, ('le', '+Inf'))} {self.totals[key]}"
            yield f"{self.name}_sum{_render_labels(key)} {self.sums[key]:g}"
            yield f"{self.name}_count{_render_labels(key)} {self.totals[key]}"


class MetricsRegistry:
    """Registry thread-safe com as séries que este projeto realmente usa."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests_total = Counter("rag_requests_total", "Requisicoes por endpoint e status.")
        self.answer_not_found_total = Counter(
            "rag_answer_not_found_total", "Respostas em que o contexto foi insuficiente."
        )
        self.errors_total = Counter("rag_errors_total", "Erros por estagio do pipeline.")
        self.documents_indexed_total = Counter(
            "rag_documents_indexed_total", "Documentos indexados pelo pipeline de ingestao."
        )
        self.chunks_indexed_total = Counter(
            "rag_chunks_indexed_total", "Chunks gravados no vector store."
        )
        self.latency_seconds = Histogram("rag_latency_seconds", "Latencia fim a fim por endpoint.")
        self.stage_latency_seconds = Histogram(
            "rag_stage_latency_seconds", "Latencia por estagio do pipeline."
        )
        self.retrieved_chunks = Histogram(
            "rag_retrieved_chunks",
            "Chunks recuperados por consulta.",
            buckets=(1, 3, 5, 10, 20, 30, 50, 100),
        )
        self.context_tokens = Histogram(
            "rag_context_tokens",
            "Tokens montados no contexto.",
            buckets=(256, 512, 1024, 2048, 4096, 8192, 16384),
        )
        self.indexed_chunks = Gauge("rag_indexed_chunks", "Chunks atualmente no indice.")
        self.indexed_documents = Gauge("rag_indexed_documents", "Documentos atualmente no indice.")

    def _all(self) -> list[Counter | Gauge | Histogram]:
        return [
            self.requests_total,
            self.answer_not_found_total,
            self.errors_total,
            self.documents_indexed_total,
            self.chunks_indexed_total,
            self.latency_seconds,
            self.stage_latency_seconds,
            self.retrieved_chunks,
            self.context_tokens,
            self.indexed_chunks,
            self.indexed_documents,
        ]

    def observe_trace(self, endpoint: str, spans: dict[str, float]) -> None:
        """Alimenta o histograma por estagio a partir de um trace concluido."""
        with self._lock:
            for stage, duration_ms in spans.items():
                self.stage_latency_seconds.observe(
                    duration_ms / 1000, stage=stage, endpoint=endpoint
                )

    def render(self) -> str:
        with self._lock:
            lines: list[str] = []
            for metric in self._all():
                lines.extend(metric.render())
            return "\n".join(lines) + "\n"


METRICS = MetricsRegistry()


def render_prometheus() -> str:
    return METRICS.render()
