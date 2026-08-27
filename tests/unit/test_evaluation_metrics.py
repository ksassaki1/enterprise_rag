from __future__ import annotations

from enterprise_rag.evaluation.generation import (
    GenerationCase,
    aggregate_generation,
    contains_answer,
    token_f1,
)
from enterprise_rag.evaluation.regression import compare
from enterprise_rag.evaluation.retrieval import (
    aggregate_retrieval,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_e_precision():
    retrieved = ["doc_b", "doc_a", "doc_c"]
    assert recall_at_k(retrieved, ["doc_a"], 1) == 0.0
    assert recall_at_k(retrieved, ["doc_a"], 2) == 1.0
    assert precision_at_k(retrieved, ["doc_a", "doc_b"], 2) == 1.0


def test_casamento_tolerante_por_sufixo():
    assert recall_at_k(["brazil/usina/relatorio-2024"], ["relatorio-2024"], 1) == 1.0


def test_mrr_e_ndcg():
    assert reciprocal_rank(["x", "alvo"], ["alvo"]) == 0.5
    assert ndcg_at_k(["alvo", "x"], ["alvo"], 10) == 1.0
    assert ndcg_at_k(["x", "alvo"], ["alvo"], 10) < 1.0


def test_agregacao_registra_perguntas_sem_acerto():
    metrics = aggregate_retrieval(
        [
            ("Q1", ["alvo"], ["alvo"]),
            ("Q2", ["outro"], ["alvo"]),
        ]
    )
    assert metrics.questions == 2
    assert metrics.recall_at_10 == 0.5
    assert metrics.misses == ["Q2"]


def test_containment_e_f1():
    assert contains_answer("A altura maxima e de 76 metros.", "76 metros")
    assert not contains_answer("A potencia e de 850 MW.", "76 metros")
    assert token_f1("76 metros", "76 metros") == 1.0


def test_pergunta_nao_respondivel_conta_como_acerto_quando_recusada():
    case = GenerationCase(
        question_id="Q016",
        answer="Nao encontrei essa informacao.",
        expected_answer="",
        answer_found=True,  # o runner ja inverteu para perguntas unanswerable
        cited_documents=[],
        expected_documents=[],
    )
    metrics = aggregate_generation([case])
    assert metrics.not_found == 0


def test_portao_de_regressao_detecta_queda_de_recall():
    baseline = {"retrieval": {"recall@10": 0.91, "recall@5": 0.84, "mrr": 0.82}}
    current = {"retrieval": {"recall@10": 0.83, "recall@5": 0.84, "mrr": 0.82}}
    outcome = compare(current, baseline)
    assert outcome.failed
    assert any(result.metric == "recall@10" and result.regressed for result in outcome.results)


def test_portao_tolera_variacao_pequena():
    baseline = {"retrieval": {"recall@10": 0.91}}
    current = {"retrieval": {"recall@10": 0.90}}
    assert not compare(current, baseline).failed


def test_latencia_usa_tolerancia_relativa():
    baseline = {"latency": {"p95_ms": 1000.0}}
    assert not compare({"latency": {"p95_ms": 1200.0}}, baseline).failed
    assert compare({"latency": {"p95_ms": 2000.0}}, baseline).failed
