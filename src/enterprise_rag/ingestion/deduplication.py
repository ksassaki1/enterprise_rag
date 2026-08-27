"""Deduplicação em três níveis.

1. **arquivo** -- SHA-256 do conteúdo. É o que torna a ingestão incremental:
   mesmo hash, mesmo id, nada é reprocessado. Vale mais do que parece: em
   ambiente corporativo, reindexar tudo a cada execução é o custo dominante.
2. **chunk exato** -- hash normalizado do texto. Cabeçalho e rodapé repetidos
   em 200 páginas viram 200 chunks idênticos se ninguém cuidar disso.
3. **chunk quase-igual** -- similaridade de Jaccard sobre shingles, aplicada
   na montagem do contexto. Versões v1/v2 do mesmo procedimento não podem
   ocupar duas vagas das seis disponíveis.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from enterprise_rag.embeddings.sparse import tokenize
from enterprise_rag.models.chunk import Chunk

SHINGLE_SIZE = 4


def file_sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    tokens = tokenize(text, remove_stopwords=False)
    if len(tokens) < size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def jaccard(left: str, right: str) -> float:
    a, b = shingles(left), shingles(right)
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def drop_exact_duplicates(chunks: list[Chunk]) -> tuple[list[Chunk], int]:
    """Remove chunks com conteúdo idêntico dentro do mesmo documento."""
    seen: set[str] = set()
    kept: list[Chunk] = []
    for chunk in chunks:
        if chunk.content_hash in seen:
            continue
        seen.add(chunk.content_hash)
        kept.append(chunk)
    return kept, len(chunks) - len(kept)


def is_near_duplicate(candidate: str, accepted: list[str], threshold: float = 0.92) -> bool:
    return any(jaccard(candidate, existing) >= threshold for existing in accepted)
