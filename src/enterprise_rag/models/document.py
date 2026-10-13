"""Documento (a unidade que o usuário enxerga) e artefatos da ingestão."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DocumentStatus(StrEnum):
    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ParsedBlock(BaseModel):
    """Bloco lógico devolvido por um parser, antes do chunking.

    O parser não decide tamanho de chunk: ele entrega apenas estrutura
    (texto + página + seção + tipo). Essa separação é o que permite trocar a
    estratégia de chunking sem tocar em nenhum parser.
    """

    text: str
    page: int | None = None
    section: str | None = None
    kind: str = "paragraph"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    document_id: str
    filename: str
    source: str
    file_type: str

    content_sha256: str
    size_bytes: int = 0
    page_count: int | None = None

    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    error: str | None = None

    version: int = 1
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionReport(BaseModel):
    """Resultado de uma execução do pipeline: o que o CLI imprime no final."""

    processed: list[Document] = Field(default_factory=list)
    skipped: list[Document] = Field(default_factory=list)
    failed: list[Document] = Field(default_factory=list)
    total_chunks: int = 0
    duration_ms: float = 0.0

    @property
    def summary(self) -> str:
        return (
            f"{len(self.processed)} indexados, {len(self.skipped)} inalterados, "
            f"{len(self.failed)} com erro, {self.total_chunks} chunks, "
            f"{self.duration_ms / 1000:.1f}s"
        )
