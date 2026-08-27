"""Contrato da API HTTP."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from enterprise_rag.container import build_container
from tests.conftest import SAMPLE_DOCUMENTS, make_settings


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import apps.api.main as api

    settings = make_settings(tmp_path)
    container = build_container(settings, configure_logging=False)
    container.pipeline.ingest_path(SAMPLE_DOCUMENTS)

    monkeypatch.setattr(api, "build_container", lambda *args, **kwargs: container)
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_expoe_configuracao_e_indice(client: TestClient):
    payload = client.get("/health").json()

    assert payload["status"] == "ok"
    assert payload["index"]["documents"] >= 5
    assert payload["index"]["chunks"] > 0
    assert payload["configuration"]["reranker"] == "lexical"


def test_metrics_no_formato_prometheus(client: TestClient):
    client.post("/ask", json={"question": "Qual a altura maxima da barragem?"})
    body = client.get("/metrics").text

    assert "# TYPE rag_requests_total counter" in body
    assert "rag_latency_seconds_bucket" in body
    assert "rag_indexed_chunks" in body


def test_listagem_e_detalhe_de_documentos(client: TestClient):
    documentos = client.get("/documents").json()
    assert documentos

    document_id = documentos[0]["document_id"]
    detalhe = client.get(f"/documents/{document_id}").json()
    assert detalhe["document_id"] == document_id
    assert detalhe["chunk_count"] > 0


def test_documento_inexistente_retorna_404(client: TestClient):
    assert client.get("/documents/nao/existe").status_code == 404


def test_search_devolve_scores_sem_gerar_resposta(client: TestClient):
    payload = client.post(
        "/search", json={"query": "potencia instalada da usina", "top_k": 5}
    ).json()

    assert payload["results"]
    assert "answer" not in payload
    assert payload["results"][0]["rerank_score"] is not None
    assert payload["metrics"]["retrieval_latency_ms"] > 0
    assert payload["trace_id"]


def test_ask_devolve_resposta_fontes_e_metricas(client: TestClient):
    payload = client.post("/ask", json={"question": "Qual a altura maxima da barragem?"}).json()

    assert payload["answer"]
    assert payload["sources"]
    assert payload["sources"][0]["chunk_id"]
    assert payload["retrieval"]["fused_candidates"] > 0
    assert payload["metrics"]["total_latency_ms"] > 0
    assert payload["model"] == "echo"


def test_ask_aceita_filtro_explicito(client: TestClient):
    payload = client.post(
        "/ask",
        json={"question": "Qual a altura maxima da barragem?", "filters": {"country": "Chile"}},
    ).json()

    assert payload["retrieval"]["filters_applied"] == {"country": "Chile"}
    assert all(source["document_id"].startswith("chile/") for source in payload["sources"])


def test_upload_e_remocao_de_documento(client: TestClient):
    conteudo = b"# Nota\n\nA potencia auxiliar do sistema e de 12 MW.\n"
    resposta = client.post(
        "/documents",
        files={"file": ("nota-teste.md", conteudo, "text/markdown")},
        data={"metadata_json": '{"country": "Peru"}'},
    )
    assert resposta.status_code == 201
    document_id = resposta.json()["document"]["document_id"]
    assert resposta.json()["chunks"] > 0

    busca = client.post("/search", json={"query": "potencia auxiliar do sistema"}).json()
    assert any(item["document_id"] == document_id for item in busca["results"])

    assert client.delete(f"/documents/{document_id}").status_code == 200
    assert client.get(f"/documents/{document_id}").status_code == 404


def test_stats_registra_consultas(client: TestClient):
    client.post("/search", json={"query": "barragem"})
    payload = client.get("/stats").json()

    assert payload["queries"]["total"] >= 1
    assert payload["recent_queries"]


def test_evaluate_roda_o_golden_dataset(client: TestClient, golden_dataset: Path):
    payload = client.post(
        "/evaluate", json={"dataset": str(golden_dataset), "skip_generation": True}
    ).json()

    assert payload["dataset_size"] == 16
    assert 0.0 <= payload["retrieval"]["recall@10"] <= 1.0


def test_validacao_de_entrada(client: TestClient):
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422
