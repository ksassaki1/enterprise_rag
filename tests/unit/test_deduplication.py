from __future__ import annotations

from enterprise_rag.ingestion.deduplication import (
    drop_exact_duplicates,
    file_sha256,
    is_near_duplicate,
    jaccard,
)
from enterprise_rag.models.chunk import Chunk


def make(chunk_id: str, content: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        content=content,
        source="doc.md",
        filename="doc.md",
        file_type="md",
    )


def test_hash_de_arquivo_muda_com_o_conteudo(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("conteudo original", encoding="utf-8")
    original = file_sha256(path)
    assert file_sha256(path) == original

    path.write_text("conteudo alterado", encoding="utf-8")
    assert file_sha256(path) != original


def test_duplicatas_exatas_sao_removidas():
    chunks = [make("1", "Cabecalho institucional"), make("2", "cabecalho INSTITUCIONAL")]
    kept, removed = drop_exact_duplicates(chunks)
    assert len(kept) == 1
    assert removed == 1


def test_jaccard_reconhece_texto_quase_igual():
    base = "A altura maxima da barragem e de 76 metros medidos da fundacao ate a crista"
    assert jaccard(base, base) == 1.0
    assert jaccard(base, base + " revisao 2") > 0.7
    assert jaccard(base, "A politica de senhas exige 14 caracteres") < 0.1


def test_near_duplicate_usa_limiar():
    base = "A altura maxima da barragem e de 76 metros medidos da fundacao ate a crista"
    assert is_near_duplicate(base + " revisao 2", [base], threshold=0.7)
    assert not is_near_duplicate("texto totalmente diferente aqui", [base], threshold=0.7)
