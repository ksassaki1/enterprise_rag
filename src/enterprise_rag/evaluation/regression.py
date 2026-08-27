"""Portão de regressão.

O que transforma "eu medi uma vez" em "eu não deixo piorar". Cada PR roda o
golden dataset e compara com o baseline versionado em `reports/baseline.json`.

A tolerância existe porque nem toda variação é regressão: reranker em GPU vs
CPU, mudança de versão do modelo e latência de máquina de CI oscilam. O que
não pode oscilar é qualidade caindo sem ninguém ver.

Métrica com `higher_is_better=False` (latência, taxa de não encontrado) inverte
a comparação.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# métrica -> (caminho no relatorio, tolerancia absoluta, maior_e_melhor)
GATES: dict[str, tuple[tuple[str, str], float, bool]] = {
    "recall@5": (("retrieval", "recall@5"), 0.03, True),
    "recall@10": (("retrieval", "recall@10"), 0.03, True),
    "mrr": (("retrieval", "mrr"), 0.03, True),
    "accuracy": (("generation", "accuracy"), 0.05, True),
    "not_found_rate": (("generation", "not_found_rate"), 0.05, False),
    "p95_latency_ms": (("latency", "p95_ms"), 0.35, True),  # tolerancia relativa
}

RELATIVE_METRICS = {"p95_latency_ms"}


@dataclass
class GateResult:
    metric: str
    baseline: float
    current: float
    tolerance: float
    higher_is_better: bool
    regressed: bool

    @property
    def delta(self) -> float:
        return self.current - self.baseline

    def render(self) -> str:
        arrow = "OK " if not self.regressed else "FALHA"
        sign = "+" if self.delta >= 0 else ""
        return (
            f"{arrow} {self.metric:<18} baseline={self.baseline:<10.4f} "
            f"atual={self.current:<10.4f} delta={sign}{self.delta:.4f}"
        )


@dataclass
class RegressionOutcome:
    results: list[GateResult]

    @property
    def failed(self) -> bool:
        return any(result.regressed for result in self.results)

    def render(self) -> str:
        lines = [result.render() for result in self.results]
        lines.append("")
        lines.append(
            "REGRESSAO DETECTADA" if self.failed else "Nenhuma regressao acima da tolerancia."
        )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Comentario de PR: a tabela que o revisor le antes do diff."""
        header = "| metrica | baseline | atual | delta | status |\n|---|---|---|---|---|"
        rows = [
            f"| {result.metric} | {result.baseline:.4f} | {result.current:.4f} | "
            f"{result.delta:+.4f} | {'REGRESSAO' if result.regressed else 'ok'} |"
            for result in self.results
        ]
        return "\n".join([header, *rows])


def _dig(report: dict[str, Any], path: tuple[str, str]) -> float | None:
    section = report.get(path[0])
    if not isinstance(section, dict):
        return None
    value = section.get(path[1])
    return float(value) if isinstance(value, (int, float)) else None


def compare(current: dict[str, Any], baseline: dict[str, Any]) -> RegressionOutcome:
    results: list[GateResult] = []
    for metric, (path, tolerance, higher_is_better) in GATES.items():
        current_value = _dig(current, path)
        baseline_value = _dig(baseline, path)
        if current_value is None or baseline_value is None:
            continue

        if metric in RELATIVE_METRICS:
            limit = baseline_value * (1 + tolerance)
            regressed = current_value > limit and baseline_value > 0
        elif higher_is_better:
            regressed = current_value < baseline_value - tolerance
        else:
            regressed = current_value > baseline_value + tolerance

        results.append(
            GateResult(
                metric=metric,
                baseline=baseline_value,
                current=current_value,
                tolerance=tolerance,
                higher_is_better=higher_is_better,
                regressed=regressed,
            )
        )
    return RegressionOutcome(results=results)


def load_report(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))
