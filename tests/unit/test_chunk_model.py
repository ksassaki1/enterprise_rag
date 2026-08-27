from __future__ import annotations

from enterprise_rag.models.chunk import Chunk, SparseVector, content_hash, estimate_tokens


def make_chunk(content: str = "Conteudo de teste do chunk.") -> Chunk:
    return Chunk(
        chunk_id="doc#0001",
        document_id="doc",
        content=content,
        source="/tmp/doc.pdf",
        filename="doc.pdf",
        file_type="pdf",
        page=37,
        section="Caracteristicas da barragem",
    )


def test_hash_ignora_caixa_e_espacamento():
    assert content_hash("A   Barragem  ") == content_hash("a barragem")


def test_chunk_preenche_hash_e_tokens():
    chunk = make_chunk()
    assert len(chunk.content_hash) == 64
    assert chunk.token_count == estimate_tokens(chunk.content)


def test_citation_label_inclui_pagina_e_secao():
    assert make_chunk().citation_label == "doc.pdf, p. 37 -- Caracteristicas da barragem"


def test_payload_roundtrip():
    chunk = make_chunk()
    assert Chunk.from_payload(chunk.payload()) == chunk


def test_sparse_dot_product():
    left = SparseVector(indices=[1, 2, 3], values=[1.0, 2.0, 3.0])
    right = SparseVector(indices=[2, 3, 9], values=[1.0, 1.0, 5.0])
    assert left.dot(right) == 5.0
