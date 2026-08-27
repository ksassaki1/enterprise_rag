"""Avaliação executável em CI.

Roda com os backends determinísticos, então os números são reprodutíveis
entre máquinas. Os limiares são o piso do sistema, não a meta: eles existem
para pegar regressão grosseira (um parser quebrado, o BM25 desligado sem
querer), enquanto `scripts/evaluate.py --baseline` cuida das variações finas.
"""

from __future__ import annotations

from pathlib import Path

from enterprise_rag.container import Container
from enterprise_rag.evaluation.runner import load_golden_dataset, run_evaluation


def test_dataset_e_valido(golden_dataset: Path):
    items = load_golden_dataset(golden_dataset)

    assert len(items) == 16
    assert len({item.id for item in items}) == len(items)
    for item in items:
        assert item.question
        assert item.expected_documents or item.unanswerable


def test_recuperacao_atinge_o_piso(indexed_container: Container, golden_dataset: Path):
    items = load_golden_dataset(golden_dataset)
    report = run_evaluation(
        indexed_container, items, dataset_name=golden_dataset.name, skip_generation=True
    )

    assert report.retrieval["recall@10"] >= 0.6, report.retrieval["misses"]
    assert report.retrieval["recall@5"] >= 0.5
    assert report.retrieval["mrr"] >= 0.4


def test_relatorio_descreve_a_configuracao_avaliada(
    indexed_container: Container, golden_dataset: Path
):
    items = load_golden_dataset(golden_dataset)[:3]
    report = run_evaluation(indexed_container, items, skip_generation=False)

    assert report.configuration["chunk_strategy"] == "structural"
    assert report.configuration["llm"] == "echo"
    assert report.dataset_size == 3
    assert report.latency["p50_ms"] > 0
    assert set(report.generation) >= {"accuracy", "not_found_rate", "valid_citation_rate"}
