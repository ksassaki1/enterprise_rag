from __future__ import annotations

from enterprise_rag.ingestion.chunker import FixedChunker, StructuralChunker
from enterprise_rag.models.document import ParsedBlock


def blocks() -> list[ParsedBlock]:
    return [
        ParsedBlock(text="Caracteristicas da barragem", kind="heading", section="Barragem"),
        ParsedBlock(text="A altura maxima e de 76 metros.", section="Barragem", page=3),
        ParsedBlock(text="O comprimento da crista e de 1.240 metros.", section="Barragem", page=3),
        ParsedBlock(text="Geracao", kind="heading", section="Geracao"),
        ParsedBlock(text="A potencia instalada e de 850 MW.", section="Geracao", page=4),
        ParsedBlock(
            text="| ug | mw |\n| --- | --- |\n| 1 | 212.5 |",
            kind="table",
            section="Geracao",
            page=4,
        ),
    ]


def test_structural_nao_atravessa_fronteira_de_secao():
    drafts = StructuralChunker(chunk_size=500).chunk(blocks())
    for draft in drafts:
        assert "76 metros" not in draft.text or "850 MW" not in draft.text


def test_structural_mantem_tabela_atomica():
    drafts = StructuralChunker(chunk_size=500).chunk(blocks())
    tables = [draft for draft in drafts if draft.kind == "table"]
    assert len(tables) == 1
    assert "212.5" in tables[0].text
    assert "potencia instalada" not in tables[0].text


def test_structural_propaga_secao_e_pagina():
    drafts = StructuralChunker(chunk_size=500).chunk(blocks())
    barragem = [draft for draft in drafts if draft.section == "Barragem"]
    assert barragem
    assert barragem[0].page == 3


def test_fixed_ignora_headings_e_divide_por_janela():
    long_block = [ParsedBlock(text=" ".join(f"palavra{i}" for i in range(600)))]
    drafts = FixedChunker(chunk_size=100, chunk_overlap=20, min_size=1).chunk(long_block)
    assert len(drafts) > 1


def test_fixed_aplica_sobreposicao():
    text = " ".join(f"t{i}" for i in range(200))
    drafts = FixedChunker(chunk_size=100, chunk_overlap=40, min_size=1).chunk(
        [ParsedBlock(text=text)]
    )
    first_words = set(drafts[0].text.split())
    second_words = set(drafts[1].text.split())
    assert first_words & second_words


def test_bloco_maior_que_orcamento_e_dividido_no_structural():
    huge = ParsedBlock(text=" ".join(f"p{i}" for i in range(2000)), section="X")
    drafts = StructuralChunker(chunk_size=200, chunk_overlap=20).chunk([huge])
    assert len(drafts) > 1
    assert all(draft.section == "X" for draft in drafts)


def test_fragmentos_curtos_sao_agrupados():
    tiny = [ParsedBlock(text="Um.", section="S"), ParsedBlock(text="Dois.", section="S")]
    drafts = StructuralChunker(chunk_size=500, min_size=40).chunk(tiny)
    assert len(drafts) == 1
