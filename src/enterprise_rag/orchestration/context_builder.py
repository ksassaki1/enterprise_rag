"""Montagem do contexto.

Entre o reranker e o LLM existe uma decisão que quase todo tutorial ignora:
*quais* chunks realmente entram no prompt, e sob que orçamento. Três regras,
todas com motivo prático:

* **orçamento de tokens** -- estourar a janela não dá erro bonito: o modelo
  simplesmente esquece o começo. O corte precisa ser nosso, não do runtime.
* **teto por documento** (`max_chunks_per_document`) -- sem ele, um relatório
  de 400 páginas ocupa as seis vagas e a resposta ignora os outros documentos,
  mesmo quando a pergunta era comparativa.
* **deduplicação semântica** -- v1 e v2 do mesmo procedimento têm
  similaridade altíssima e não podem consumir duas vagas.

O bloco final é numerado (`[FONTE n]`) porque é assim que o prompt pede a
citação e é assim que a resposta é auditada depois.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from enterprise_rag.ingestion.deduplication import is_near_duplicate
from enterprise_rag.models.chunk import ScoredChunk, estimate_tokens
from enterprise_rag.models.response import ContextStats


@dataclass
class BuiltContext:
    text: str = ""
    chunks: list[ScoredChunk] = field(default_factory=list)
    stats: ContextStats = field(default_factory=ContextStats)

    @property
    def is_empty(self) -> bool:
        return not self.chunks


class ContextBuilder:
    def __init__(
        self,
        max_context_tokens: int = 7000,
        max_chunks_per_document: int = 3,
        dedup_similarity: float = 0.92,
        max_chunks: int | None = None,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.max_chunks_per_document = max_chunks_per_document
        self.dedup_similarity = dedup_similarity
        self.max_chunks = max_chunks

    def build(self, candidates: list[ScoredChunk]) -> BuiltContext:
        stats = ContextStats()
        selected: list[ScoredChunk] = []
        accepted_texts: list[str] = []
        per_document: dict[str, int] = {}
        seen_hashes: set[str] = set()
        used_tokens = 0

        for candidate in candidates:
            chunk = candidate.chunk

            if chunk.content_hash in seen_hashes:
                stats.dropped_duplicates += 1
                continue

            document_count = per_document.get(chunk.document_id, 0)
            if document_count >= self.max_chunks_per_document:
                stats.dropped_by_document_cap += 1
                continue

            block = render_source(len(selected) + 1, candidate)
            block_tokens = estimate_tokens(block)
            if used_tokens + block_tokens > self.max_context_tokens:
                stats.dropped_by_budget += 1
                continue

            if is_near_duplicate(chunk.content, accepted_texts, self.dedup_similarity):
                stats.dropped_duplicates += 1
                continue

            selected.append(candidate)
            accepted_texts.append(chunk.content)
            seen_hashes.add(chunk.content_hash)
            per_document[chunk.document_id] = document_count + 1
            used_tokens += block_tokens

            if self.max_chunks and len(selected) >= self.max_chunks:
                break

        text = "\n\n".join(
            render_source(index, candidate) for index, candidate in enumerate(selected, start=1)
        )
        stats.chunks_in_context = len(selected)
        stats.context_tokens = used_tokens
        stats.documents_in_context = len(per_document)
        return BuiltContext(text=text, chunks=selected, stats=stats)


def render_source(index: int, candidate: ScoredChunk) -> str:
    """Serializa um chunk como fonte numerada.

    Os metadados vão no cabeçalho, e não misturados ao texto, para que o modelo
    consiga distinguir o que é conteúdo do documento e o que é procedência --
    e para que a citação `[FONTE 3]` seja resolvível de volta ao `chunk_id`.
    """
    chunk = candidate.chunk
    header = [f"[FONTE {index}]", f"documento: {chunk.filename}"]
    if chunk.page is not None:
        header.append(f"pagina: {chunk.page}")
    if chunk.section:
        header.append(f"secao: {chunk.section}")
    for key in ("project", "country", "document_type", "year"):
        value = chunk.metadata.get(key)
        if value:
            header.append(f"{key}: {value}")
    return "\n".join(header) + "\n" + chunk.content
