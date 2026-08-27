"""O objeto fundamental da plataforma.

Tudo -- ingestao, busca, reranking, contexto, citacao -- circula como `Chunk`.
Ter um schema canonico e o que permite filtro por metadados, deduplicacao por
hash e citacao com pagina/secao sem gambiarra depois.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


def content_hash(text: str) -> str:
    """Hash normalizado do conteudo (insensivel a caixa e espacamento)."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """Aproximacao barata de tokens.

    Deliberadamente sem tiktoken: o orcamento de contexto so precisa de uma
    estimativa estavel e conservadora, e a heuristica de ~4 chars/token vale
    tanto para pt-BR quanto para ingles. Trocar por um tokenizer real e um
    detalhe de implementacao isolado nesta funcao.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 4) + text.count("\n"))


class SparseVector(BaseModel):
    """Vetor esparso no formato aceito pelo Qdrant (indices + pesos)."""

    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)

    def as_dict(self) -> dict[int, float]:
        return dict(zip(self.indices, self.values, strict=True))

    def dot(self, other: SparseVector) -> float:
        mine = self.as_dict()
        pairs = zip(other.indices, other.values, strict=True)
        return sum(weight * mine.get(index, 0.0) for index, weight in pairs)


class Chunk(BaseModel):
    """Unidade indexavel e citavel."""

    chunk_id: str
    document_id: str

    content: str

    source: str
    filename: str
    file_type: str

    page: int | None = None
    section: str | None = None
    position: int = 0

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    content_hash: str = ""
    token_count: int = 0

    def model_post_init(self, __context: Any) -> None:
        if not self.content_hash:
            self.content_hash = content_hash(self.content)
        if not self.token_count:
            self.token_count = estimate_tokens(self.content)

    @property
    def citation_label(self) -> str:
        label = self.filename
        if self.page is not None:
            label += f", p. {self.page}"
        if self.section:
            label += f" -- {self.section}"
        return label

    def payload(self) -> dict[str, Any]:
        """Representacao plana persistida no vector store."""
        return self.model_dump(mode="json")

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Chunk:
        return cls.model_validate(payload)


class ScoredChunk(BaseModel):
    """Chunk com a trilha de scores de cada estagio do funil."""

    chunk: Chunk
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None

    @property
    def score(self) -> float:
        """Score efetivo: o do estagio mais tardio disponivel."""
        for value in (self.rerank_score, self.fusion_score, self.dense_score, self.sparse_score):
            if value is not None:
                return value
        return 0.0
