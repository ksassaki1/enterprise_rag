"""Benchmark comparativo de configurações.

O ponto do projeto não é afirmar "chunking estrutural é melhor" nem
"reranking vale a pena". É medir as duas coisas no mesmo dataset e mostrar a
tabela:

    Estrategia          Recall@5   Recall@10   MRR     P50
    fixed-500             0.72       0.78      0.66   0.31s
    fixed-1000            0.75       0.81      0.69   0.29s
    structural            0.83       0.87      0.78   0.33s

Cada configuração é indexada do zero em um diretório temporário próprio, com
o backend em memória: sem isso, um índice contaminaria o outro e o número não
significaria nada.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Any

from enterprise_rag.config.settings import Settings, get_settings
from enterprise_rag.container import build_container
from enterprise_rag.evaluation.runner import load_golden_dataset, run_evaluation

# nome -> overrides de Settings
CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "fixed-500": {"chunk_strategy": "fixed", "chunk_size": 500, "chunk_overlap": 50},
    "fixed-1000": {"chunk_strategy": "fixed", "chunk_size": 1000, "chunk_overlap": 100},
    "structural": {"chunk_strategy": "structural", "chunk_size": 500, "chunk_overlap": 60},
    "structural-no-rerank": {
        "chunk_strategy": "structural",
        "chunk_size": 500,
        "chunk_overlap": 60,
        "reranker_backend": "none",
    },
    "structural-dense-only": {
        "chunk_strategy": "structural",
        "chunk_size": 500,
        "chunk_overlap": 60,
        "use_hybrid": False,
    },
}


def build_variant(base: Settings, overrides: dict[str, Any], workdir: Path) -> Settings:
    data = base.model_dump()
    data.update(overrides)
    # isolamento total: indice proprio, catalogo proprio, tudo em memoria
    data["data_dir"] = workdir
    data["catalog_path"] = workdir / "catalog.sqlite3"
    data["vector_backend"] = "memory"
    return Settings(**data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara configuracoes de RAG.")
    parser.add_argument("--documents", type=Path, default=Path("datasets/sample_documents"))
    parser.add_argument("--dataset", type=Path, default=Path("datasets/golden_dataset.jsonl"))
    parser.add_argument(
        "--only", nargs="*", default=None, help="subconjunto de configuracoes a rodar"
    )
    parser.add_argument(
        "--with-generation",
        action="store_true",
        help="tambem mede geracao (bem mais lento; o padrao mede so recuperacao)",
    )
    args = parser.parse_args()

    base = get_settings()
    items = load_golden_dataset(args.dataset)
    names = args.only or list(CONFIGURATIONS)
    rows: list[tuple[str, dict[str, Any]]] = []

    workroot = Path(tempfile.mkdtemp(prefix="erag-benchmark-"))
    try:
        for name in names:
            overrides = CONFIGURATIONS.get(name)
            if overrides is None:
                print(f"configuracao desconhecida: {name}")
                continue

            workdir = workroot / name
            workdir.mkdir(parents=True, exist_ok=True)
            settings = build_variant(base, overrides, workdir)
            container = build_container(settings, configure_logging=False)
            try:
                container.pipeline.ingest_path(args.documents, force=True)
                report = run_evaluation(
                    container,
                    items,
                    dataset_name=args.dataset.name,
                    skip_generation=not args.with_generation,
                )
            finally:
                container.close()
            rows.append((name, report.to_dict()))
            print(f"[ok] {name}")
    finally:
        shutil.rmtree(workroot, ignore_errors=True)

    print()
    print("=" * 78)
    print("CHUNKING / RETRIEVAL BENCHMARK")
    print("=" * 78)
    header = f"{'configuracao':<24}{'Recall@5':>10}{'Recall@10':>11}{'MRR':>8}{'nDCG@10':>10}{'P50':>9}"
    print(header)
    print("-" * 78)
    for name, report in rows:
        retrieval = report["retrieval"]
        latency = report["latency"]
        print(
            f"{name:<24}{retrieval['recall@5']:>10.2f}{retrieval['recall@10']:>11.2f}"
            f"{retrieval['mrr']:>8.2f}{retrieval['ndcg@10']:>10.2f}"
            f"{latency['p50_ms'] / 1000:>8.2f}s"
        )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
