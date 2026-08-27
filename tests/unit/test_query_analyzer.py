from __future__ import annotations

from enterprise_rag.orchestration.query_analyzer import RuleBasedQueryAnalyzer

analyzer = RuleBasedQueryAnalyzer()


def test_infere_pais_da_pergunta():
    plan = analyzer.analyze("Quais barragens brasileiras possuem altura superior a 50 metros?")
    assert plan.filters["country"] == "Brazil"
    assert plan.notes


def test_infere_tipo_de_documento_e_ano():
    plan = analyzer.analyze("O que diz o relatorio de 2024 sobre percolacao?")
    assert plan.filters["document_type"] == "technical_report"
    assert plan.filters["year"] == 2024


def test_detecta_codigos_tecnicos():
    plan = analyzer.analyze("Qual a potencia da UHE-SRA-01 segundo a NBR 6118?")
    assert "UHE-SRA-01" in plan.detected_entities


def test_query_esparsa_preserva_codigo_e_remove_stopwords():
    plan = analyzer.analyze("Qual e a potencia instalada da UHE-SRA-01?")
    assert "uhe-sra-01" in plan.sparse_query
    assert " e " not in f" {plan.sparse_query} "


def test_query_densa_preserva_a_pergunta_original():
    question = "Qual e a altura maxima da barragem?"
    assert analyzer.analyze(question).dense_query == question


def test_intencao_de_comparacao_e_resumo():
    assert analyzer.analyze("Compare as duas usinas").intent == "comparison"
    assert analyzer.analyze("Resuma o relatorio").intent == "summary"
    assert analyzer.analyze("Qual a altura da barragem?").intent == "qa"


def test_pergunta_sem_pista_nao_gera_filtro():
    assert analyzer.analyze("Qual a altura da barragem?").filters == {}
