from __future__ import annotations

from enterprise_rag.ingestion.metadata import build_document_id, extract_metadata
from enterprise_rag.ingestion.parsers import default_registry
from enterprise_rag.ingestion.parsers.txt import TextParser, split_frontmatter
from enterprise_rag.ingestion.pipeline import normalize_text

MARKDOWN = """---
project: Usina Serra Azul
document_type: technical_report
---

# Caracteristicas da barragem

A altura maxima e de 76 metros.

## Geracao

A potencia instalada e de 850 MW.
"""


def test_frontmatter_e_separado_do_corpo():
    body, metadata = split_frontmatter(MARKDOWN)
    assert metadata["project"] == "Usina Serra Azul"
    assert "---" not in body


def test_markdown_vira_blocos_com_secao(tmp_path):
    path = tmp_path / "relatorio.md"
    path.write_text(MARKDOWN, encoding="utf-8")
    blocks = TextParser().parse(path)

    paragrafos = [block for block in blocks if block.kind == "paragraph"]
    assert any(block.section == "Caracteristicas da barragem" for block in paragrafos)
    assert any(block.section == "Geracao" for block in paragrafos)


def test_registry_resolve_por_extensao(tmp_path):
    registry = default_registry()
    assert registry.resolve(tmp_path / "a.md") is not None
    assert registry.resolve(tmp_path / "a.csv") is not None
    assert registry.resolve(tmp_path / "a.zip") is None
    assert ".pdf" in registry.supported_extensions


def test_csv_gera_bloco_de_tabela_com_cabecalho(tmp_path):
    path = tmp_path / "equipamentos.csv"
    path.write_text("tag,potencia\nUG-01,212.5\nUG-02,212.5\n", encoding="utf-8")
    blocks = default_registry().resolve(path).parse(path)

    assert blocks[0].kind == "table"
    assert "tag" in blocks[0].text
    assert "UG-02" in blocks[0].text


def test_normalizacao_remove_hifen_de_quebra_e_espacos():
    assert normalize_text("infra-\nestrutura") == "infraestrutura"
    assert normalize_text("a    b\n\n\n\nc") == "a b\n\nc"


def test_document_id_vem_do_caminho_relativo(tmp_path):
    root = tmp_path / "docs"
    path = root / "brazil" / "Usina Serra Azul" / "Relatorio 2024.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")
    assert build_document_id(path, root) == "brazil/usina-serra-azul/relatorio-2024"


def test_metadata_prioriza_sidecar_sobre_heuristica(tmp_path):
    path = tmp_path / "brazil" / "relatorio.md"
    path.parent.mkdir(parents=True)
    path.write_text("conteudo", encoding="utf-8")
    (path.parent / "_metadata.json").write_text('{"country": "Chile"}', encoding="utf-8")

    metadata = extract_metadata(path, root=tmp_path)
    assert metadata["country"] == "Chile"
    assert metadata["file_type"] == "md"
