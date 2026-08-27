"""CLI de avaliação e portão de regressão.

    python scripts/evaluate.py
    python scripts/evaluate.py --baseline reports/baseline.json
    python scripts/evaluate.py --update-baseline

Sai com código 1 quando alguma métrica regride além da tolerância -- é isso
que o GitHub Actions usa para reprovar o PR.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from enterprise_rag.config.settings import get_settings
from enterprise_rag.container import build_container
from enterprise_rag.evaluation.regression import compare, load_report
from enterprise_rag.evaluation.runner import load_golden_dataset, run_evaluation

DEFAULT_DATASET = Path("datasets/golden_dataset.jsonl")
DEFAULT_OUTPUT = Path("reports/evaluation.json")


def render(report: dict[str, Any]) -> str:
    retrieval = report["retrieval"]
    generation = report["generation"]
    latency = report["latency"]
    configuration = report["configuration"]

    lines = [
        "=" * 62,
        "RAG EVALUATION",
        "=" * 62,
        f"Dataset            {report['dataset']}  ({report['dataset_size']} perguntas)",
        f"Embeddings         {configuration['embedding_model']}",
        f"Chunking           {configuration['chunk_strategy']} "
        f"({configuration['chunk_size']}/{configuration['chunk_overlap']})",
        f"Reranker           {configuration['reranker']}",
        f"LLM                {configuration['llm']}",
        "",
        "Retrieval",
        f"  Recall@1         {retrieval['recall@1']:.2f}",
        f"  Recall@5         {retrieval['recall@5']:.2f}",
        f"  Recall@10        {retrieval['recall@10']:.2f}",
        f"  Precision@5      {retrieval['precision@5']:.2f}",
        f"  MRR              {retrieval['mrr']:.2f}",
        f"  nDCG@10          {retrieval['ndcg@10']:.2f}",
        "",
        "Generation",
        f"  Corretas         {generation['correct']}",
        f"  Incorretas       {generation['incorrect']}",
        f"  Nao encontrado   {generation['not_found']}",
        f"  Accuracy         {generation['accuracy']:.2f}",
        f"  Token F1         {generation['avg_token_f1']:.2f}",
        f"  Citacao valida   {generation['valid_citation_rate']:.2f}",
        "",
        "Performance",
        f"  P50              {latency['p50_ms'] / 1000:.2f}s",
        f"  P95              {latency['p95_ms'] / 1000:.2f}s",
        f"  Retrieval P50    {latency['retrieval_p50_ms'] / 1000:.2f}s",
        f"  Generation P50   {latency['generation_p50_ms'] / 1000:.2f}s",
        "=" * 62,
    ]
    if retrieval["misses"]:
        lines.append("Perguntas sem documento esperado no top-10:")
        lines.extend(f"  - {item}" for item in retrieval["misses"])
        lines.append("=" * 62)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia a plataforma RAG.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--markdown", type=Path, default=None, help="salva resumo em Markdown")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"dataset nao encontrado: {args.dataset}")
        return 2

    container = build_container(get_settings())
    try:
        items = load_golden_dataset(args.dataset)
        report = run_evaluation(
            container,
            items,
            dataset_name=args.dataset.name,
            skip_generation=args.skip_generation,
        )
    finally:
        container.close()

    payload = report.to_dict()
    report.save(args.output)
    print(render(payload))
    print(f"\nrelatorio salvo em {args.output}")

    if args.update_baseline:
        baseline_path = args.baseline or Path("reports/baseline.json")
        report.save(baseline_path)
        print(f"baseline atualizado em {baseline_path}")
        return 0

    if args.baseline:
        if not args.baseline.exists():
            print(f"\nbaseline {args.baseline} nao existe; gere com --update-baseline")
            return 0
        outcome = compare(payload, load_report(args.baseline))
        print()
        print("=" * 62)
        print("REGRESSION GATE")
        print("=" * 62)
        print(outcome.render())
        if args.markdown:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(outcome.to_markdown(), encoding="utf-8")
        return 1 if outcome.failed else 0

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            "```\n" + render(payload) + "\n```", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
