"""CLI de ingestão.

    python scripts/ingest.py ./datasets/sample_documents
    python scripts/ingest.py ./docs --force --strategy fixed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from enterprise_rag.config.settings import get_settings
from enterprise_rag.container import build_container
from enterprise_rag.ingestion.chunker import build_chunker


def main() -> int:
    parser = argparse.ArgumentParser(description="Indexa documentos na plataforma.")
    parser.add_argument("path", type=Path, help="arquivo ou diretorio a indexar")
    parser.add_argument(
        "--force", action="store_true", help="reindexa mesmo se o hash nao mudou"
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed", "structural"],
        default=None,
        help="sobrescreve a estrategia de chunking desta execucao",
    )
    parser.add_argument(
        "--clear", action="store_true", help="apaga o indice antes de comecar"
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"caminho nao encontrado: {args.path}", file=sys.stderr)
        return 2

    settings = get_settings()
    container = build_container(settings)

    if args.strategy:
        container.pipeline.chunker = build_chunker(settings, args.strategy)
    if args.clear:
        container.store.clear()

    try:
        report = container.pipeline.ingest_path(args.path, force=args.force)
    finally:
        container.close()

    print()
    print("=" * 62)
    print("INGESTAO")
    print("=" * 62)
    print(f"Indexados      {len(report.processed)}")
    print(f"Inalterados    {len(report.skipped)}")
    print(f"Com erro       {len(report.failed)}")
    print(f"Chunks         {report.total_chunks}")
    print(f"Duracao        {report.duration_ms / 1000:.1f}s")
    if report.failed:
        print()
        print("Falhas:")
        for document in report.failed:
            print(f"  - {document.filename}: {document.error}")
    print("=" * 62)
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
