from __future__ import annotations

from enterprise_rag.models.chunk import Chunk, ScoredChunk
from enterprise_rag.orchestration.context_builder import ContextBuilder


def scored(chunk_id: str, document_id: str, content: str, score: float = 1.0) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        source=f"{document_id}.md",
        filename=f"{document_id}.md",
        file_type="md",
        page=1,
    )
    return ScoredChunk(chunk=chunk, rerank_score=score)


def test_teto_por_documento_abre_espaco_para_outros():
    candidates = [scored(f"a#{i}", "doc_a", f"conteudo unico numero {i}") for i in range(5)]
    candidates.append(scored("b#1", "doc_b", "conteudo do segundo documento"))
    context = ContextBuilder(max_chunks_per_document=2).build(candidates)

    documentos = {item.chunk.document_id for item in context.chunks}
    assert documentos == {"doc_a", "doc_b"}
    assert context.stats.dropped_by_document_cap == 3


def test_orcamento_de_tokens_corta_o_excedente():
    candidates = [scored(f"a#{i}", f"doc_{i}", "palavra " * 400) for i in range(10)]
    context = ContextBuilder(max_context_tokens=300, max_chunks_per_document=1).build(candidates)
    assert context.stats.context_tokens <= 300
    assert context.stats.dropped_by_budget > 0


def test_duplicata_exata_e_descartada():
    candidates = [
        scored("a#1", "doc_a", "A altura maxima da barragem e de 76 metros."),
        scored("b#1", "doc_b", "A ALTURA MAXIMA DA BARRAGEM E DE 76 METROS."),
    ]
    context = ContextBuilder().build(candidates)
    assert len(context.chunks) == 1
    assert context.stats.dropped_duplicates == 1


def test_quase_duplicata_e_descartada():
    base = "A altura maxima da barragem e de 76 metros medidos entre a fundacao e a crista."
    candidates = [
        scored("a#1", "doc_a", base),
        scored("b#1", "doc_b", base + " Revisao 2."),
    ]
    context = ContextBuilder(dedup_similarity=0.7).build(candidates)
    assert len(context.chunks) == 1


def test_contexto_e_numerado_e_traz_procedencia():
    context = ContextBuilder().build([scored("a#1", "doc_a", "Texto do chunk.")])
    assert "[FONTE 1]" in context.text
    assert "documento: doc_a.md" in context.text
    assert "pagina: 1" in context.text


def test_contexto_vazio():
    context = ContextBuilder().build([])
    assert context.is_empty
    assert context.text == ""
