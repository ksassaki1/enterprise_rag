"""Pipeline de ingestão.

    descoberta -> hash -> parse -> normalização -> metadados -> chunking
                                                -> dense + sparse -> store

Duas propriedades importam mais que o resto:

* **incremental**: arquivo com o mesmo SHA-256 de antes não é reprocessado.
  Em um acervo corporativo, reindexar tudo a cada execução é o custo dominante
  e a causa mais comum de "o RAG está caro".
* **idempotente**: reingerir um arquivo alterado apaga os chunks antigos
  daquele `document_id` antes de gravar os novos, então nenhuma resposta cita
  uma versão que não existe mais.
"""

from __future__ import annotations

import time
import unicodedata
from pathlib import Path

from enterprise_rag.config.logging import get_logger
from enterprise_rag.config.settings import Settings
from enterprise_rag.embeddings.dense import DenseEmbedder
from enterprise_rag.embeddings.sparse import BM25SparseEncoder
from enterprise_rag.ingestion.chunker import ChunkDraft, Chunker
from enterprise_rag.ingestion.deduplication import drop_exact_duplicates, file_sha256
from enterprise_rag.ingestion.metadata import build_document_id, extract_metadata
from enterprise_rag.ingestion.parsers import ParserRegistry
from enterprise_rag.models.chunk import Chunk
from enterprise_rag.models.document import Document, DocumentStatus, IngestionReport, ParsedBlock
from enterprise_rag.observability.metrics import METRICS
from enterprise_rag.storage.base import VectorRecord, VectorStore
from enterprise_rag.storage.catalog import Catalog

logger = get_logger(__name__)

IGNORED_SUFFIXES = (".meta.json",)
IGNORED_NAMES = {"_metadata.json"}


def normalize_text(text: str) -> str:
    """Normalização mínima, e reversível o suficiente para não perder sentido.

    Unicode NFKC (une variantes de acento e largura), hífen de quebra de linha
    remendado (praga clássica de PDF), espaços colapsados. Sem lowercase e sem
    remover pontuação: isso é trabalho do tokenizador do BM25 e atrapalharia a
    leitura do LLM.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "").replace("-\n", "")
    cleaned = "\n".join(" ".join(line.split()) for line in text.splitlines())
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        catalog: Catalog,
        embedder: DenseEmbedder,
        sparse_encoder: BM25SparseEncoder,
        registry: ParserRegistry,
        chunker: Chunker,
    ) -> None:
        self.settings = settings
        self.store = store
        self.catalog = catalog
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.registry = registry
        self.chunker = chunker

    def discover(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root]
        return [
            path
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and not path.name.startswith(".")
            and path.name not in IGNORED_NAMES
            and not path.name.endswith(IGNORED_SUFFIXES)
            and self.registry.resolve(path) is not None
        ]

    def ingest_path(self, root: Path, force: bool = False) -> IngestionReport:
        started = time.perf_counter()
        report = IngestionReport()
        base = root if root.is_dir() else root.parent
        files = self.discover(root)
        logger.info("documentos encontrados", extra={"total": len(files), "raiz": str(root)})

        for path in files:
            document = self.ingest_file(path, root=base, force=force)
            if document.status == DocumentStatus.INDEXED:
                report.processed.append(document)
                report.total_chunks += document.chunk_count
            elif document.status == DocumentStatus.SKIPPED:
                report.skipped.append(document)
            else:
                report.failed.append(document)

        self.sparse_encoder.save()
        report.duration_ms = (time.perf_counter() - started) * 1000
        METRICS.indexed_documents.set(self.catalog.stats()["documents"])
        METRICS.indexed_chunks.set(self.store.count())
        logger.info("ingestao concluida", extra={"resumo": report.summary})
        return report

    def ingest_file(self, path: Path, root: Path | None = None, force: bool = False) -> Document:
        document_id = build_document_id(path, root)
        sha256 = file_sha256(path)

        if not force and self.catalog.is_unchanged(document_id, sha256):
            existing = self.catalog.get_document(document_id)
            logger.info("inalterado, ignorando", extra={"document_id": document_id})
            base = existing or _stub(document_id, path, sha256)
            return base.model_copy(update={"status": DocumentStatus.SKIPPED})

        parser = self.registry.resolve(path)
        if parser is None:
            document = _failed(document_id, path, sha256, f"nenhum parser para {path.suffix}")
            self.catalog.upsert_document(document)
            return document

        try:
            blocks = parser.parse(path)
        except Exception as exc:
            METRICS.errors_total.inc(stage="parse")
            logger.warning(
                "falha ao interpretar documento",
                extra={"arquivo": path.name, "erro": f"{type(exc).__name__}: {exc}"},
            )
            document = _failed(document_id, path, sha256, f"{type(exc).__name__}: {exc}")
            self.catalog.upsert_document(document)
            return document

        blocks = [
            block.model_copy(update={"text": normalize_text(block.text)})
            for block in blocks
            if block.text.strip()
        ]
        sample = "\n".join(block.text for block in blocks[:5])
        frontmatter = dict(blocks[0].metadata) if blocks else {}
        metadata = extract_metadata(path, root=root, frontmatter=frontmatter, sample_text=sample)

        previous = self.catalog.get_document(document_id)
        document = Document(
            document_id=document_id,
            filename=path.name,
            source=str(path),
            file_type=path.suffix.lower().lstrip("."),
            content_sha256=sha256,
            size_bytes=path.stat().st_size,
            page_count=parser.page_count(path),
            status=DocumentStatus.PENDING,
            version=(previous.version + 1) if previous else 1,
            metadata=metadata,
        )

        chunks = self._build_chunks(document, blocks)
        if not chunks:
            document.status = DocumentStatus.FAILED
            document.error = "nenhum chunk gerado"
            self.catalog.upsert_document(document)
            return document

        # do ponto de vista da consulta a troca é atômica: a versão anterior
        # sai do índice antes de a nova entrar
        self.store.delete_document(document_id)
        self._index(chunks)

        document.status = DocumentStatus.INDEXED
        document.chunk_count = len(chunks)
        self.catalog.upsert_document(document)
        METRICS.documents_indexed_total.inc(file_type=document.file_type)
        METRICS.chunks_indexed_total.inc(len(chunks), file_type=document.file_type)
        logger.info(
            "documento indexado",
            extra={"document_id": document_id, "chunks": len(chunks), "versao": document.version},
        )
        return document

    def _build_chunks(self, document: Document, blocks: list[ParsedBlock]) -> list[Chunk]:
        drafts: list[ChunkDraft] = self.chunker.chunk(blocks)
        chunks = [
            Chunk(
                chunk_id=f"{document.document_id}#{position:04d}",
                document_id=document.document_id,
                content=draft.text,
                source=document.source,
                filename=document.filename,
                file_type=document.file_type,
                page=draft.page,
                section=draft.section,
                position=position,
                metadata={**document.metadata, **draft.metadata, "kind": draft.kind},
            )
            for position, draft in enumerate(drafts)
        ]
        chunks, removed = drop_exact_duplicates(chunks)
        if removed:
            logger.debug(
                "chunks duplicados removidos",
                extra={"document_id": document.document_id, "removidos": removed},
            )
        return chunks

    def _index(self, chunks: list[Chunk]) -> None:
        texts = [chunk.content for chunk in chunks]

        # o BM25 precisa das estatísticas do corpus atualizadas ANTES de pesar
        # os documentos: caso contrário o primeiro lote fica com comprimento
        # médio errado e o score sai enviesado
        self.sparse_encoder.fit(texts)

        dense_vectors = self.embedder.encode_documents(texts)
        records = [
            VectorRecord(
                chunk=chunk,
                dense=dense,
                sparse=self.sparse_encoder.encode_document(chunk.content),
            )
            for chunk, dense in zip(chunks, dense_vectors, strict=True)
        ]
        self.store.ensure_ready(self.embedder.dim)
        self.store.upsert(records)

    def delete_document(self, document_id: str) -> int:
        """Remove documento do índice, do catálogo e das estatísticas do BM25."""
        chunks = self.store.get_document_chunks(document_id)
        if chunks:
            self.sparse_encoder.forget([chunk.content for chunk in chunks])
            self.sparse_encoder.save()
        removed = self.store.delete_document(document_id)
        self.catalog.delete_document(document_id)
        METRICS.indexed_chunks.set(self.store.count())
        METRICS.indexed_documents.set(self.catalog.stats()["documents"])
        return removed


def _stub(document_id: str, path: Path, sha256: str) -> Document:
    return Document(
        document_id=document_id,
        filename=path.name,
        source=str(path),
        file_type=path.suffix.lower().lstrip("."),
        content_sha256=sha256,
        size_bytes=path.stat().st_size if path.exists() else 0,
    )


def _failed(document_id: str, path: Path, sha256: str, error: str) -> Document:
    document = _stub(document_id, path, sha256)
    document.status = DocumentStatus.FAILED
    document.error = error
    return document
