"""Ingestão ponta a ponta com os backends determinísticos."""

from __future__ import annotations

import shutil
from pathlib import Path

from enterprise_rag.container import Container
from enterprise_rag.models.document import DocumentStatus


def test_ingestao_indexa_todos_os_documentos_de_exemplo(
    container: Container, sample_documents: Path
):
    report = container.pipeline.ingest_path(sample_documents)

    assert not report.failed
    assert len(report.processed) >= 5
    assert report.total_chunks > 10
    assert container.store.count() == report.total_chunks


def test_reingestao_sem_alteracao_nao_reprocessa(container: Container, sample_documents: Path):
    container.pipeline.ingest_path(sample_documents)
    segunda = container.pipeline.ingest_path(sample_documents)

    assert not segunda.processed
    assert len(segunda.skipped) >= 5
    assert segunda.total_chunks == 0


def test_arquivo_alterado_e_reindexado_com_nova_versao(container: Container, tmp_path: Path):
    workspace = tmp_path / "docs"
    workspace.mkdir()
    document = workspace / "nota.md"
    document.write_text("# Nota\n\nA altura e de 76 metros.\n", encoding="utf-8")

    primeira = container.pipeline.ingest_path(workspace)
    assert len(primeira.processed) == 1
    assert primeira.processed[0].version == 1

    document.write_text("# Nota\n\nA altura e de 80 metros.\n", encoding="utf-8")
    segunda = container.pipeline.ingest_path(workspace)

    assert len(segunda.processed) == 1
    assert segunda.processed[0].version == 2
    # a versao antiga saiu do indice: nada responde mais "76"
    conteudos = " ".join(chunk.content for chunk in container.store.iter_chunks())
    assert "80 metros" in conteudos
    assert "76 metros" not in conteudos


def test_metadados_de_diretorio_e_frontmatter_chegam_ao_chunk(
    indexed_container: Container,
):
    chunks = [
        chunk
        for chunk in indexed_container.store.iter_chunks()
        if "serra-azul" in chunk.document_id
    ]
    assert chunks
    assert chunks[0].metadata["country"] == "Brazil"
    assert chunks[0].metadata["project"] == "Usina Serra Azul"


def test_remocao_apaga_chunks_catalogo_e_estatisticas(indexed_container: Container):
    documento = indexed_container.catalog.list_documents()[0]
    antes = indexed_container.store.count()

    removidos = indexed_container.pipeline.delete_document(documento.document_id)

    assert removidos > 0
    assert indexed_container.store.count() == antes - removidos
    assert indexed_container.catalog.get_document(documento.document_id) is None


def test_arquivo_corrompido_falha_sem_derrubar_o_lote(container: Container, tmp_path: Path):
    workspace = tmp_path / "docs"
    workspace.mkdir()
    (workspace / "ok.md").write_text("# Ok\n\nConteudo valido para indexar.\n", encoding="utf-8")
    (workspace / "quebrado.pdf").write_bytes(b"isto nao e um PDF")

    report = container.pipeline.ingest_path(workspace)

    assert len(report.processed) == 1
    assert len(report.failed) == 1
    assert report.failed[0].status == DocumentStatus.FAILED
    assert report.failed[0].error


def test_upload_com_sidecar_de_metadata(container: Container, tmp_path: Path):
    workspace = tmp_path / "docs"
    workspace.mkdir()
    documento = workspace / "spec.md"
    documento.write_text("# Spec\n\nTexto qualquer para indexar.\n", encoding="utf-8")
    documento.with_suffix(".md.meta.json").write_text(
        '{"country": "Peru", "project": "Projeto Z"}', encoding="utf-8"
    )

    container.pipeline.ingest_path(workspace)
    chunks = list(container.store.iter_chunks())

    assert chunks[0].metadata["country"] == "Peru"
    assert chunks[0].metadata["project"] == "Projeto Z"


def test_indice_fica_no_diretorio_de_dados_do_teste(indexed_container: Container):
    settings = indexed_container.settings
    assert settings.memory_store_path.exists()
    assert settings.memory_store_path.parent == settings.data_dir
    assert settings.sparse_stats_path.exists()


def test_copia_do_mesmo_arquivo_em_outro_caminho_gera_documento_distinto(
    container: Container, tmp_path: Path
):
    workspace = tmp_path / "docs"
    (workspace / "a").mkdir(parents=True)
    (workspace / "b").mkdir(parents=True)
    origem = workspace / "a" / "nota.md"
    origem.write_text("# Nota\n\nTexto suficiente para gerar um chunk.\n", encoding="utf-8")
    shutil.copy(origem, workspace / "b" / "nota.md")

    report = container.pipeline.ingest_path(workspace)
    assert len(report.processed) == 2
