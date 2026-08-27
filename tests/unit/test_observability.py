from __future__ import annotations

import time

import pytest

from enterprise_rag.observability.metrics import MetricsRegistry
from enterprise_rag.observability.tracing import current_trace, new_trace


def test_trace_registra_spans_por_estagio():
    with new_trace("ask", query="teste") as trace:
        with trace.span("retrieval"):
            time.sleep(0.005)
        with trace.span("generation"):
            time.sleep(0.005)

    assert {span.name for span in trace.spans} == {"retrieval", "generation"}
    assert trace.duration_of("retrieval") > 0
    assert trace.to_dict()["trace_id"] == trace.trace_id


def test_span_captura_excecao_sem_engolir():
    with new_trace("ask") as trace:
        with pytest.raises(ValueError), trace.span("generation"):
            raise ValueError("falhou")
    assert trace.spans[0].error is not None


def test_trace_e_isolado_por_contexto():
    assert current_trace() is None
    with new_trace("x") as trace:
        assert current_trace() is trace
    assert current_trace() is None


def test_render_prometheus_tem_formato_valido():
    registry = MetricsRegistry()
    registry.requests_total.inc(endpoint="ask", status="ok")
    registry.latency_seconds.observe(0.42, endpoint="ask")
    output = registry.render()

    assert "# TYPE rag_requests_total counter" in output
    assert 'rag_requests_total{endpoint="ask",status="ok"} 1' in output
    assert 'rag_latency_seconds_bucket{endpoint="ask",le="+Inf"} 1' in output
    assert "rag_latency_seconds_count" in output


def test_quantis_do_histograma():
    registry = MetricsRegistry()
    for value in range(1, 101):
        registry.latency_seconds.observe(value / 100, endpoint="ask")
    assert 0.4 < registry.latency_seconds.quantile(0.5, endpoint="ask") < 0.6
