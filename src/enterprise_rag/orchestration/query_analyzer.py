"""Processamento da consulta.

Antes de buscar qualquer coisa, o sistema decide três coisas:

1. **filtros** -- "quais barragens brasileiras..." vira `country=Brazil`, e o
   espaço de busca encolhe *antes* do vetor. Sem isso, metade do top-30 é
   gasto com documentos de outro país;
2. **reescrita por canal** -- o recuperador denso recebe a pergunta em
   linguagem natural (é assim que o modelo foi treinado); o esparso recebe
   apenas os termos discriminativos, com códigos preservados;
3. **intenção** -- pergunta pontual, comparação ou resumo pedem quantidades
   diferentes de contexto.

Tudo por regra, de propósito: é determinístico, custa microssegundos, é
testável e não gasta uma chamada de LLM antes da chamada de LLM. Quando as
regras não bastarem, `LLMQueryAnalyzer` faz o mesmo trabalho com o modelo --
mas aí o custo aparece no trace, explicitamente.
"""

from __future__ import annotations

import json
import re
from typing import Any

from enterprise_rag.config.logging import get_logger
from enterprise_rag.embeddings.sparse import strip_accents, tokenize
from enterprise_rag.ingestion.metadata import COUNTRY_HINTS, DOCUMENT_TYPE_HINTS
from enterprise_rag.models.response import QueryPlan

logger = get_logger(__name__)

# Codigos tecnicos: UHE-SRA-01, UHE-SRA-01-UG-04, NBR 6118, PR-045/2024, ABC123.
# Duas formas aceitas: segmentos alfanumericos ligados por separador, ou
# sigla seguida de numero. Sem grupos de captura -- findall devolve o codigo
# inteiro, e perder o prefixo ("SRA-01" no lugar de "UHE-SRA-01") custaria
# justamente o termo mais discriminativo da consulta.
CODE_RE = re.compile(
    r"\b[A-Z]{2,}(?:[-_/][A-Z0-9]{1,8})+\b"
    r"|\b[A-Z]{2,}[- ]?\d{2,6}\b"
)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

FILE_TYPE_HINTS = {
    "pdf": "pdf",
    "planilha": "xlsx",
    "planilhas": "xlsx",
    "excel": "xlsx",
    "word": "docx",
    "docx": "docx",
}

COMPARISON_MARKERS = ("compare", "comparar", "diferenca", "versus", " vs ", "melhor que")
SUMMARY_MARKERS = ("resuma", "resumo", "sintetize", "visao geral", "overview")


class RuleBasedQueryAnalyzer:
    name = "rules"

    def analyze(self, query: str) -> QueryPlan:
        normalized = strip_accents(query.lower())
        filters: dict[str, Any] = {}
        notes: list[str] = []

        country = self._detect(normalized, COUNTRY_HINTS)
        if country:
            filters["country"] = country
            notes.append(f"filtro country={country} inferido da pergunta")

        document_type = self._detect(normalized, DOCUMENT_TYPE_HINTS)
        if document_type:
            filters["document_type"] = document_type
            notes.append(f"filtro document_type={document_type} inferido da pergunta")

        file_type = self._detect(normalized, FILE_TYPE_HINTS)
        if file_type:
            filters["file_type"] = file_type

        year_match = YEAR_RE.search(query)
        if year_match:
            filters["year"] = int(year_match.group(0))
            notes.append(f"filtro year={year_match.group(0)} inferido da pergunta")

        codes = CODE_RE.findall(query)

        return QueryPlan(
            original_query=query.strip(),
            dense_query=query.strip(),
            sparse_query=self._sparse_query(query, codes),
            filters=filters,
            intent=self._intent(normalized),
            detected_entities=codes,
            notes=notes,
        )

    def _detect(self, normalized: str, table: dict[str, str]) -> str | None:
        for token, value in table.items():
            if re.search(rf"\b{re.escape(token)}\b", normalized):
                return value
        return None

    def _sparse_query(self, query: str, codes: list[str]) -> str:
        """Só os termos discriminativos, com os códigos garantidos no fim.

        Códigos são reinseridos porque o BM25 os pontua com IDF altíssimo:
        perdê-los na limpeza é abrir mão da única vantagem do canal lexical.
        """
        terms = tokenize(query)
        for code in codes:
            token = strip_accents(code.lower())
            if token not in terms:
                terms.append(token)
        return " ".join(terms) or query.strip()

    def _intent(self, normalized: str) -> str:
        if any(marker in normalized for marker in COMPARISON_MARKERS):
            return "comparison"
        if any(marker in normalized for marker in SUMMARY_MARKERS):
            return "summary"
        return "qa"


class LLMQueryAnalyzer:
    """Extrai filtros com o LLM local, caindo para as regras se algo falhar.

    Útil quando o vocabulário de metadados cresce além do que uma tabela de
    sinônimos aguenta. O custo é uma chamada extra por pergunta -- visível no
    trace como o span `query_processing`.
    """

    name = "llm"

    def __init__(self, llm: Any, fallback: RuleBasedQueryAnalyzer | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or RuleBasedQueryAnalyzer()

    PROMPT = (
        "Extraia filtros de metadados da pergunta do usuario. "
        "Responda SOMENTE com JSON no formato "
        '{"filters": {"country": "...", "document_type": "...", "year": 2024}} '
        "usando apenas as chaves aplicaveis. Se nao houver filtro claro, responda "
        '{"filters": {}}.\n\nPergunta: '
    )

    def analyze(self, query: str) -> QueryPlan:
        plan = self.fallback.analyze(query)
        try:
            raw = self.llm.complete(self.PROMPT + query, max_tokens=200)
            payload = json.loads(_extract_json(raw))
            extracted = payload.get("filters", {})
            if isinstance(extracted, dict) and extracted:
                plan.filters = {**plan.filters, **extracted}
                plan.notes.append("filtros refinados pelo LLM")
        except Exception as exc:
            logger.debug("analise por LLM falhou; usando regras", extra={"erro": str(exc)})
        return plan


def _extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start >= 0 and end > start else "{}"
