"""Tracing mínimo, sem dependência obrigatória.

Cada pergunta ganha um `trace_id` e uma árvore de spans com duração por
estágio. Isso é o suficiente para responder, olhando um único JSON, a pergunta
que mais importa em RAG: o tempo foi embora no retrieval, no reranker ou no LLM?

Se `RAG_OTEL_ENABLED=true` e o SDK do OpenTelemetry estiver instalado, os mesmos
spans são exportados via OTLP. O código de negócio não muda.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from enterprise_rag.config.logging import current_trace_id

_current_trace: ContextVar["Trace | None"] = ContextVar("_current_trace", default=None)


@dataclass
class Span:
    name: str
    start: float
    duration_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "duration_ms": round(self.duration_ms, 2)}
        if self.attributes:
            data["attributes"] = self.attributes
        if self.error:
            data["error"] = self.error
        return data


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = "request"
    start: float = field(default_factory=time.perf_counter)
    spans: list[Span] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        span = Span(name=name, start=time.perf_counter(), attributes=dict(attributes))
        self.spans.append(span)
        otel_span = _start_otel_span(name)
        try:
            yield span
        except Exception as exc:
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.duration_ms = (time.perf_counter() - span.start) * 1000
            _end_otel_span(otel_span, span)

    def duration_of(self, name: str) -> float:
        return sum(span.duration_ms for span in self.spans if span.name == name)

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "total_ms": round(self.total_ms, 2),
            "attributes": self.attributes,
            "spans": [span.to_dict() for span in self.spans],
        }


@contextmanager
def new_trace(name: str = "request", **attributes: Any) -> Iterator[Trace]:
    trace = Trace(name=name, attributes=dict(attributes))
    token = _current_trace.set(trace)
    log_token = current_trace_id.set(trace.trace_id)
    try:
        yield trace
    finally:
        _current_trace.reset(token)
        current_trace_id.reset(log_token)


def current_trace() -> Trace | None:
    return _current_trace.get()


# ── ponte opcional com OpenTelemetry ────────────────────────────────────────

_otel_tracer: Any | None = None
_otel_ready = False


def init_otel(service_name: str, endpoint: str | None) -> bool:
    """Liga o exportador OTLP se as libs estiverem instaladas. Nunca quebra o app."""
    global _otel_tracer, _otel_ready
    _otel_ready = True
    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return False

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if endpoint:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    otel_trace.set_tracer_provider(provider)
    _otel_tracer = otel_trace.get_tracer(service_name)
    return True


def _start_otel_span(name: str) -> Any | None:
    if _otel_tracer is None:
        return None
    return _otel_tracer.start_span(name)


def _end_otel_span(otel_span: Any | None, span: Span) -> None:
    if otel_span is None:
        return
    for key, value in span.attributes.items():
        try:
            otel_span.set_attribute(key, value)
        except Exception:
            pass
    otel_span.end()
