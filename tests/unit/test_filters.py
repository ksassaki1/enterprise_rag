from __future__ import annotations

from enterprise_rag.retrieval.filters import FilterSet

PAYLOAD = {
    "chunk_id": "d#0001",
    "document_id": "brazil/usina/relatorio",
    "filename": "relatorio.pdf",
    "file_type": "pdf",
    "page": 37,
    "metadata": {"country": "Brazil", "year": 2024, "document_type": "technical_report"},
}


def test_campo_de_metadata_e_resolvido_automaticamente():
    assert FilterSet.from_dict({"country": "Brazil"}).matches(PAYLOAD)


def test_comparacao_e_insensivel_a_caixa():
    assert FilterSet.from_dict({"country": "brazil"}).matches(PAYLOAD)


def test_campo_de_topo_nao_vira_metadata():
    assert FilterSet.from_dict({"file_type": "pdf"}).matches(PAYLOAD)
    assert not FilterSet.from_dict({"file_type": "docx"}).matches(PAYLOAD)


def test_operadores_numericos():
    assert FilterSet.from_dict({"page": {"gte": 30}}).matches(PAYLOAD)
    assert not FilterSet.from_dict({"page": {"lt": 10}}).matches(PAYLOAD)
    assert FilterSet.from_dict({"year": {"gte": 2020, "lte": 2025}}).matches(PAYLOAD)


def test_operador_in_e_lista_implicita():
    assert FilterSet.from_dict({"file_type": ["pdf", "docx"]}).matches(PAYLOAD)
    assert FilterSet.from_dict({"country": {"in": ["Brazil", "Chile"]}}).matches(PAYLOAD)
    assert not FilterSet.from_dict({"country": {"nin": ["Brazil"]}}).matches(PAYLOAD)


def test_contains_e_exists():
    assert FilterSet.from_dict({"filename": {"contains": "relat"}}).matches(PAYLOAD)
    assert FilterSet.from_dict({"country": {"exists": True}}).matches(PAYLOAD)
    assert FilterSet.from_dict({"projeto_x": {"exists": False}}).matches(PAYLOAD)


def test_condicoes_sao_combinadas_com_and():
    assert not FilterSet.from_dict({"country": "Brazil", "file_type": "docx"}).matches(PAYLOAD)


def test_filtro_vazio_e_falsy():
    assert not FilterSet.from_dict({})
    assert FilterSet.from_dict({}).matches(PAYLOAD)
