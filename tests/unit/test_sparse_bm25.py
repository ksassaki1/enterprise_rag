from __future__ import annotations

from enterprise_rag.embeddings.sparse import BM25SparseEncoder, tokenize


def test_tokenize_preserva_codigos_tecnicos():
    tokens = tokenize("A unidade UHE-SRA-01 segue a NBR 6118.")
    assert "uhe-sra-01" in tokens
    assert "nbr" in tokens
    assert "6118" in tokens


def test_tokenize_remove_stopwords_com_acento():
    assert "nao" not in tokenize("nao e para isso")


def test_idf_penaliza_termo_frequente():
    encoder = BM25SparseEncoder()
    encoder.fit(["barragem concreto"] * 10 + ["turbina francis"])
    assert encoder.idf("turbina") > encoder.idf("barragem")


def test_documento_com_termo_raro_pontua_mais():
    encoder = BM25SparseEncoder()
    corpus = [
        "A altura maxima da barragem e de 76 metros.",
        "A potencia instalada da unidade UHE-SRA-01 e de 850 MW.",
        "O reservatorio possui volume total de 1.980 hm3.",
    ]
    encoder.fit(corpus)
    query = encoder.encode_query("potencia instalada UHE-SRA-01")
    scores = [query.dot(encoder.encode_document(text)) for text in corpus]
    assert scores.index(max(scores)) == 1


def test_forget_reverte_estatisticas():
    encoder = BM25SparseEncoder()
    encoder.fit(["barragem de concreto"])
    encoder.forget(["barragem de concreto"])
    assert encoder.document_count == 0
    assert "barragem" not in encoder.document_frequency


def test_persistencia_preserva_estatisticas(tmp_path):
    path = tmp_path / "stats.json"
    encoder = BM25SparseEncoder(stats_path=path)
    encoder.fit(["turbina francis de eixo vertical"])
    encoder.save()

    reloaded = BM25SparseEncoder(stats_path=path)
    assert reloaded.document_count == 1
    assert reloaded.idf("turbina") == encoder.idf("turbina")
