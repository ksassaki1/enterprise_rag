"""Fluxo RAG completo sobre os documentos de exemplo."""

from __future__ import annotations

from enterprise_rag.container import Container


def documentos(response) -> list[str]:
    return [result.document_id for result in response.results]


def test_busca_encontra_o_documento_certo(indexed_container: Container):
    response = indexed_container.orchestrator.search(
        "Qual e a potencia instalada da Usina Serra Azul?", top_k=5
    )
    assert any("relatorio-tecnico-2024" in item for item in documentos(response))


def test_busca_lexical_recupera_codigo_de_ativo(indexed_container: Container):
    response = indexed_container.orchestrator.search("UHE-SRA-01-UG-04", top_k=5)
    assert response.results
    assert any("serra-azul" in item for item in documentos(response))


def test_busca_devolve_scores_de_cada_estagio(indexed_container: Container):
    response = indexed_container.orchestrator.search("altura maxima da barragem", top_k=5)
    primeiro = response.results[0]
    assert primeiro.fusion_score is not None
    assert primeiro.rerank_score is not None
    assert response.retrieval.dense_candidates > 0
    assert response.retrieval.sparse_candidates > 0


def test_filtro_de_metadados_restringe_o_resultado(indexed_container: Container):
    response = indexed_container.orchestrator.search(
        "altura maxima da barragem", top_k=10, filters={"country": "Chile"}
    )
    assert response.results
    assert all(item.startswith("chile/") for item in documentos(response))
    assert response.retrieval.filters_applied == {"country": "Chile"}


def test_filtro_inferido_da_pergunta(indexed_container: Container):
    response = indexed_container.orchestrator.search(
        "Qual a altura das barragens brasileiras?", top_k=10
    )
    assert response.plan.filters.get("country") == "Brazil"
    assert all(item.startswith("brazil/") for item in documentos(response))


def test_ask_responde_com_citacoes_e_metricas(indexed_container: Container):
    response = indexed_container.orchestrator.ask("Qual a altura maxima da barragem?")

    assert response.answer_found
    assert response.sources
    assert response.trace_id
    assert response.metrics.total_latency_ms > 0
    assert response.metrics.retrieval_latency_ms > 0
    assert response.context.chunks_in_context > 0
    assert any(source.used_in_answer for source in response.sources)


def test_ask_respeita_teto_de_chunks_por_documento(indexed_container: Container):
    indexed_container.orchestrator.context_builder.max_chunks_per_document = 1
    response = indexed_container.orchestrator.ask("Qual a potencia instalada?")

    documentos_no_contexto = [source.document_id for source in response.sources]
    assert len(documentos_no_contexto) == len(set(documentos_no_contexto))


def test_search_nao_chama_o_gerador(indexed_container: Container):
    class Explode:
        name = "explode"

        def chat(self, system: str, user: str) -> str:
            raise AssertionError("search nao deve gerar texto")

        def complete(self, prompt: str, max_tokens: int = 512) -> str:
            raise AssertionError("search nao deve gerar texto")

        def health(self) -> bool:
            return True

    indexed_container.orchestrator.llm = Explode()
    response = indexed_container.orchestrator.search("potencia instalada", top_k=3)
    assert response.results


def test_pergunta_sem_resposta_na_base_e_recusada(indexed_container: Container):
    response = indexed_container.orchestrator.ask(
        "Qual e o valor do seguro contratado para o parque de maquinas?"
    )
    assert not response.answer_found


def test_base_vazia_responde_nao_encontrado(container: Container):
    response = container.orchestrator.ask("Qual a altura da barragem?")
    assert not response.answer_found
    assert response.context.chunks_in_context == 0
