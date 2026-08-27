"""Estratégias de chunking.

Chunking é a decisão de maior impacto em recall e a mais fácil de fazer mal.
`texto[i:i+1000]` corta no meio da frase, separa o número do seu rótulo e
espalha uma tabela em três pedaços inúteis.

Duas estratégias implementadas, propositalmente comparáveis:

* **fixed** -- janela de tamanho fixo com sobreposição. Baseline honesto.
* **structural** -- respeita a hierarquia que o parser já extraiu
  (título -> seção -> parágrafo -> tabela); nunca atravessa fronteira de seção
  e nunca parte uma tabela.

Nenhuma das duas é "a certa": `scripts/benchmark.py` mede as duas no mesmo
golden dataset e imprime Recall@K lado a lado. A afirmação defensável não é
"estrutural é melhor", é "eu medi".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from enterprise_rag.config.settings import ChunkStrategy, Settings
from enterprise_rag.models.chunk import estimate_tokens
from enterprise_rag.models.document import ParsedBlock

# 1 token ~ 0.75 palavra em pt-BR/en. Trabalhar em palavras mantém o corte
# alinhado com fronteiras reais de texto.
WORDS_PER_TOKEN = 0.75


@dataclass
class ChunkDraft:
    """Chunk antes de ganhar identidade (id, hash, metadados do documento)."""

    text: str
    page: int | None = None
    section: str | None = None
    kind: str = "paragraph"
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


@runtime_checkable
class Chunker(Protocol):
    name: str

    def chunk(self, blocks: list[ParsedBlock]) -> list[ChunkDraft]: ...


def _split_words(text: str, size_tokens: int, overlap_tokens: int) -> list[str]:
    """Janela deslizante em palavras, com sobreposição."""
    words = text.split()
    if not words:
        return []
    size = max(1, int(size_tokens * WORDS_PER_TOKEN))
    overlap = max(0, min(int(overlap_tokens * WORDS_PER_TOKEN), size - 1))
    step = size - overlap
    windows: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            break
        windows.append(" ".join(window))
        if start + size >= len(words):
            break
    return windows


class FixedChunker:
    """Janela fixa sobre o texto de cada bloco.

    Mantém página e seção do bloco de origem: a citação continua apontando
    para um lugar plausível mesmo quando o corte ignora a estrutura.
    """

    name = "fixed"

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 60, min_size: int = 40) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_size = min_size

    def chunk(self, blocks: list[ParsedBlock]) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        for block in blocks:
            if block.kind == "heading":
                continue
            for window in _split_words(block.text, self.chunk_size, self.chunk_overlap):
                drafts.append(
                    ChunkDraft(
                        text=window,
                        page=block.page,
                        section=block.section,
                        kind=block.kind,
                        metadata=dict(block.metadata),
                    )
                )
        return _merge_tiny(drafts, self.min_size)


class StructuralChunker:
    """Agrupa blocos consecutivos da mesma seção até o orçamento de tokens.

    Regras que valem mais do que parecem em documentos técnicos:

    * fronteira de seção sempre fecha o chunk (assunto novo, chunk novo);
    * tabela é atômica: vira um chunk próprio e nunca é dividida no meio;
    * bloco maior que o orçamento cai para janela fixa, com sobreposição,
      para não perder conteúdo.
    """

    name = "structural"

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 60, min_size: int = 40) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_size = min_size

    def chunk(self, blocks: list[ParsedBlock]) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        buffer: list[ParsedBlock] = []
        buffer_tokens = 0
        current_section: str | None = None

        def flush() -> None:
            nonlocal buffer, buffer_tokens
            if not buffer:
                return
            head = buffer[0]
            text = "\n\n".join(block.text for block in buffer).strip()
            if text:
                drafts.append(
                    ChunkDraft(
                        text=text,
                        page=head.page,
                        section=head.section,
                        kind=head.kind,
                        metadata=dict(head.metadata),
                    )
                )
            buffer = []
            buffer_tokens = 0

        for block in blocks:
            if block.kind == "heading":
                # o título contextualiza o próximo chunk; não vira chunk sozinho
                flush()
                current_section = block.section or block.text
                continue

            section = block.section or current_section
            if section != current_section and buffer:
                flush()
            current_section = section

            block_tokens = estimate_tokens(block.text)
            enriched = block.model_copy(update={"section": section})

            if block.kind == "table":
                flush()
                if block_tokens <= self.chunk_size * 2:
                    drafts.append(
                        ChunkDraft(
                            text=enriched.text,
                            page=enriched.page,
                            section=section,
                            kind="table",
                            metadata=dict(enriched.metadata),
                        )
                    )
                else:
                    drafts.extend(self._split_oversized(enriched, section))
                continue

            if block_tokens > self.chunk_size:
                flush()
                drafts.extend(self._split_oversized(enriched, section))
                continue

            if buffer_tokens + block_tokens > self.chunk_size:
                flush()

            buffer.append(enriched)
            buffer_tokens += block_tokens

        flush()
        return _merge_tiny(drafts, self.min_size)

    def _split_oversized(self, block: ParsedBlock, section: str | None) -> list[ChunkDraft]:
        return [
            ChunkDraft(
                text=window,
                page=block.page,
                section=block.section or section,
                kind=block.kind,
                metadata=dict(block.metadata),
            )
            for window in _split_words(block.text, self.chunk_size, self.chunk_overlap)
        ]


def _merge_tiny(drafts: list[ChunkDraft], min_size: int) -> list[ChunkDraft]:
    """Junta fragmentos curtos ao chunk anterior da mesma seção.

    Chunks de 10 tokens poluem o ranking: casam com qualquer coisa e não
    carregam informação suficiente para responder nada.
    """
    merged: list[ChunkDraft] = []
    for draft in drafts:
        if (
            merged
            and draft.tokens < min_size
            and merged[-1].section == draft.section
            and merged[-1].kind != "table"
            and draft.kind != "table"
        ):
            merged[-1].text = f"{merged[-1].text}\n\n{draft.text}".strip()
            continue
        merged.append(draft)
    return [draft for draft in merged if draft.text.strip()]


def build_chunker(settings: Settings, strategy: ChunkStrategy | None = None) -> Chunker:
    chosen = strategy or settings.chunk_strategy
    if chosen == "fixed":
        return FixedChunker(settings.chunk_size, settings.chunk_overlap, settings.chunk_min_size)
    return StructuralChunker(settings.chunk_size, settings.chunk_overlap, settings.chunk_min_size)
