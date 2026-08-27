"""API HTTP.

`/search` e `/ask` são endpoints distintos de propósito: um devolve o que foi
recuperado, o outro devolve a resposta gerada. Quando uma resposta sai errada,
essa separação responde em uma requisição de qual metade veio o erro -- e é a
diferença entre depurar e adivinhar.

Toda resposta carrega `trace_id`, latência por estágio e as fontes usadas.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field

from enterprise_rag import __version__
from enterprise_rag.config.logging import get_logger
from enterprise_rag.config.settings import get_settings
from enterprise_rag.container import Container, build_container
from enterprise_rag.evaluation.runner import load_golden_dataset, run_evaluation
from enterprise_rag.generation.ollama import LLMError
from enterprise_rag.models.document import Document
from enterprise_rag.models.response import AskResponse, SearchResponse
from enterprise_rag.observability.metrics import METRICS, render_prometheus

logger = get_logger(__name__)

STATE: dict[str, Container] = {}


def container() -> Container:
    instance = STATE.get("container")
    if instance is None:  # pragma: no cover - so acontece fora do lifespan
        raise HTTPException(status_code=503, detail="plataforma nao inicializada")
    return instance


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    STATE["container"] = build_container()
    try:
        yield
    finally:
        STATE.pop("container").close()


app = FastAPI(
    title="Enterprise RAG Platform",
    version=__version__,
    description=(
        "RAG local-first com retrieval hibrido, reranking, geracao ancorada, "
        "avaliacao e observabilidade."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Any:
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
    if request.url.path not in {"/metrics", "/health"}:
        logger.info(
            "requisicao",
            extra={
                "metodo": request.method,
                "rota": request.url.path,
                "status": response.status_code,
                "ms": round(elapsed_ms, 1),
            },
        )
    return response


# ── contratos de entrada ────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, examples=["potencia instalada da usina"])
    top_k: int | None = Field(default=None, ge=1, le=100)
    filters: dict[str, Any] | None = Field(default=None, examples=[{"country": "Brazil"}])
    rerank: bool = True


class AskRequest(BaseModel):
    question: str = Field(min_length=1, examples=["Qual a altura maxima da barragem?"])
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: dict[str, Any] | None = None


class EvaluateRequest(BaseModel):
    dataset: str = "datasets/golden_dataset.jsonl"
    skip_generation: bool = False
    output: str | None = None


# ── operação ────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["operacao"])
def health() -> dict[str, Any]:
    """Healthcheck com dependências.

    Devolve 200 mesmo com o LLM fora do ar: a busca continua funcionando, e
    marcar o serviço inteiro como morto por causa do gerador esconde
    exatamente a informação que o operador precisa.
    """
    instance = container()
    catalog_stats = instance.catalog.stats()
    return {
        "status": "ok",
        "version": __version__,
        "configuration": instance.describe(),
        "dependencies": {
            "vector_store": _safe(lambda: instance.store.count() >= 0),
            "catalog": True,
            "llm": _safe(instance.llm.health),
        },
        "index": {
            "documents": catalog_stats["documents"],
            "chunks": instance.store.count(),
            "failed_documents": catalog_stats["failed"],
        },
    }


@app.get("/metrics", tags=["operacao"], response_class=PlainTextResponse)
def metrics() -> str:
    """Exposicao no text format do Prometheus."""
    return render_prometheus()


@app.get("/stats", tags=["operacao"])
def stats() -> dict[str, Any]:
    instance = container()
    return {
        "index": instance.catalog.stats(),
        "queries": instance.catalog.query_stats(),
        "latency": {
            "p50_ms": METRICS.latency_seconds.quantile(0.5, endpoint="ask") * 1000,
            "p95_ms": METRICS.latency_seconds.quantile(0.95, endpoint="ask") * 1000,
        },
        "recent_queries": instance.catalog.recent_queries(limit=10),
    }


# ── documentos ──────────────────────────────────────────────────────────


@app.get("/documents", tags=["documentos"], response_model=list[Document])
def list_documents(limit: int = 100, offset: int = 0) -> list[Document]:
    return container().catalog.list_documents(limit=limit, offset=offset)


@app.get("/documents/{document_id:path}", tags=["documentos"], response_model=Document)
def get_document(document_id: str) -> Document:
    document = container().catalog.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"documento {document_id} nao encontrado")
    return document


@app.post("/documents", tags=["documentos"], status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    metadata_json: str | None = Form(default=None),
) -> dict[str, Any]:
    """Upload + ingestão síncrona.

    Síncrona de propósito: o cliente precisa saber se o documento foi
    indexado, não apenas se o arquivo chegou. Para lotes grandes existe o
    `scripts/ingest.py`.
    """
    instance = container()
    settings = instance.settings
    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(file.filename or "documento").name
    target = upload_dir / filename
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    size_mb = target.stat().st_size / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413, detail=f"arquivo acima de {settings.max_upload_mb} MB"
        )

    if metadata_json:
        sidecar = target.with_suffix(target.suffix + ".meta.json")
        sidecar.write_text(metadata_json, encoding="utf-8")

    document = instance.pipeline.ingest_file(target, root=upload_dir, force=True)
    if document.status.value == "failed":
        raise HTTPException(status_code=422, detail=document.error or "falha na ingestao")
    return {"document": document.model_dump(mode="json"), "chunks": document.chunk_count}


@app.delete("/documents/{document_id:path}", tags=["documentos"])
def delete_document(document_id: str) -> dict[str, Any]:
    instance = container()
    if instance.catalog.get_document(document_id) is None:
        raise HTTPException(status_code=404, detail=f"documento {document_id} nao encontrado")
    removed = instance.pipeline.delete_document(document_id)
    return {"document_id": document_id, "chunks_removed": removed}


# ── busca e RAG ─────────────────────────────────────────────────────────


@app.post("/search", tags=["rag"], response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Somente recuperação: nenhum token de LLM é gasto.

    Use este endpoint para responder "o chunk certo foi recuperado?" antes de
    culpar o modelo.
    """
    instance = container()
    response = instance.orchestrator.search(
        query=request.query,
        top_k=request.top_k,
        filters=request.filters,
        rerank=request.rerank,
    )
    instance.catalog.log_query(
        trace_id=response.trace_id,
        endpoint="search",
        query=request.query,
        chunks_used=len(response.results),
        latency_ms=response.metrics.total_latency_ms,
        detail={"filters": response.plan.filters},
    )
    return response


@app.post("/ask", tags=["rag"], response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """RAG completo: recuperação, reranking, contexto, geração e citações."""
    instance = container()
    try:
        response = instance.orchestrator.ask(
            question=request.question, top_k=request.top_k, filters=request.filters
        )
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=f"gerador indisponivel: {exc}") from exc

    instance.catalog.log_query(
        trace_id=response.trace_id,
        endpoint="ask",
        query=request.question,
        answer_found=response.answer_found,
        chunks_used=response.context.chunks_in_context,
        latency_ms=response.metrics.total_latency_ms,
        detail={
            "filters": response.plan.filters if response.plan else {},
            "sources": [source.chunk_id for source in response.sources],
        },
    )
    return response


# ── avaliação ───────────────────────────────────────────────────────────


@app.post("/evaluate", tags=["avaliacao"])
def evaluate(request: EvaluateRequest, background: BackgroundTasks) -> JSONResponse:
    """Roda o golden dataset contra a configuração atualmente em execução."""
    instance = container()
    dataset_path = Path(request.dataset)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"dataset {dataset_path} nao encontrado")

    items = load_golden_dataset(dataset_path)
    report = run_evaluation(
        instance,
        items,
        dataset_name=dataset_path.name,
        skip_generation=request.skip_generation,
    )
    if request.output:
        background.add_task(report.save, Path(request.output))
    return JSONResponse(content=report.to_dict())


def _safe(check: Any) -> bool:
    try:
        return bool(check())
    except Exception:
        return False


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "apps.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "local",
    )


if __name__ == "__main__":
    main()
