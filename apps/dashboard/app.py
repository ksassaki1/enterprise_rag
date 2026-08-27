"""Dashboard operacional (Streamlit).

Não é o produto -- é a janela para o que o sistema está fazendo: o que está
indexado, quanto tempo cada estágio custa, quantas perguntas ficam sem
resposta e como o retrieval se comporta antes do LLM entrar.

    streamlit run apps/dashboard/app.py
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Enterprise RAG", page_icon="", layout="wide")


def api_get(path: str) -> dict | list | None:
    try:
        response = httpx.get(f"{API_URL}{path}", timeout=15)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        st.error(f"API indisponivel em {API_URL}: {exc}")
        return None


def api_post(path: str, payload: dict) -> dict | None:
    try:
        response = httpx.post(f"{API_URL}{path}", json=payload, timeout=180)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        st.error(f"Falha na chamada {path}: {exc}")
        return None


st.title("Enterprise RAG Platform")
st.caption(f"API: {API_URL}")

health = api_get("/health")
stats = api_get("/stats")

if health:
    index = health["index"]
    queries = (stats or {}).get("queries", {})
    latency = (stats or {}).get("latency", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documentos indexados", index["documents"])
    col2.metric("Chunks", index["chunks"])
    col3.metric("Consultas", queries.get("total", 0))
    col4.metric("Latencia media", f"{queries.get('avg_latency_ms', 0) / 1000:.2f}s")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("P50", f"{latency.get('p50_ms', 0) / 1000:.2f}s")
    col6.metric("P95", f"{latency.get('p95_ms', 0) / 1000:.2f}s")
    col7.metric("Nao encontrado", f"{queries.get('not_found_rate', 0) * 100:.1f}%")
    col8.metric("Documentos com erro", index["failed_documents"])

    with st.expander("Configuracao efetiva"):
        st.json(health["configuration"])
        st.json(health["dependencies"])

tab_ask, tab_search, tab_docs = st.tabs(["Perguntar", "Buscar (retrieval)", "Documentos"])

with tab_ask:
    question = st.text_input("Pergunta", placeholder="Qual a altura maxima da barragem?")
    if st.button("Perguntar", type="primary") and question:
        with st.spinner("Consultando..."):
            answer = api_post("/ask", {"question": question})
        if answer:
            if not answer["answer_found"]:
                st.warning("O sistema nao encontrou a resposta nos documentos indexados.")
            st.markdown(answer["answer"])

            st.subheader("Fontes")
            for position, source in enumerate(answer["sources"], start=1):
                used = "usada na resposta" if source["used_in_answer"] else "recuperada"
                label = f"[FONTE {position}] {source['document']}"
                if source["page"] is not None:
                    label += f" - p. {source['page']}"
                with st.expander(f"{label}  ({used}, score {source['score']:.3f})"):
                    st.write(source["excerpt"])

            st.subheader("Latencia por estagio")
            st.bar_chart(
                {
                    "ms": {
                        "query": answer["metrics"]["query_processing_ms"],
                        "retrieval": answer["metrics"]["retrieval_latency_ms"],
                        "rerank": answer["metrics"]["rerank_latency_ms"],
                        "context": answer["metrics"]["context_build_ms"],
                        "generation": answer["metrics"]["generation_latency_ms"],
                    }
                }
            )
            with st.expander("Trace completo"):
                st.json(
                    {
                        "trace_id": answer["trace_id"],
                        "plan": answer["plan"],
                        "retrieval": answer["retrieval"],
                        "context": answer["context"],
                        "metrics": answer["metrics"],
                    }
                )

with tab_search:
    st.caption("Retrieval puro, sem LLM. Use para separar erro de busca de erro de geracao.")
    query = st.text_input("Consulta", key="search_query")
    top_k = st.slider("top_k", 1, 20, 5)
    if st.button("Buscar") and query:
        results = api_post("/search", {"query": query, "top_k": top_k})
        if results:
            st.json(results["plan"])
            for result in results["results"]:
                scores = " | ".join(
                    f"{label}={result[key]:.3f}"
                    for label, key in (
                        ("dense", "dense_score"),
                        ("sparse", "sparse_score"),
                        ("fusion", "fusion_score"),
                        ("rerank", "rerank_score"),
                    )
                    if result.get(key) is not None
                )
                header = f"{result['document']}"
                if result["page"] is not None:
                    header += f" - p. {result['page']}"
                with st.expander(f"{header}   ({scores})"):
                    st.write(result["content"])

with tab_docs:
    documents = api_get("/documents")
    if documents:
        st.dataframe(
            [
                {
                    "documento": document["filename"],
                    "tipo": document["file_type"],
                    "chunks": document["chunk_count"],
                    "versao": document["version"],
                    "status": document["status"],
                    "indexado_em": document["ingested_at"],
                }
                for document in documents
            ],
            width="stretch",
        )
    else:
        st.info("Nenhum documento indexado. Rode: python scripts/ingest.py ./datasets/sample_documents")
