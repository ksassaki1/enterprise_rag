"""Filtro por metadados, independente de backend.

A mesma estrutura é avaliada em memória (para o backend `memory` e para os
testes) e traduzida para o `Filter` nativo do Qdrant. É isso que permite
responder "quais barragens **brasileiras**..." reduzindo o espaço de busca
*antes* do vetor, em vez de gastar top-k com documentos de outro país.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Operator = Literal["eq", "ne", "in", "nin", "gt", "gte", "lt", "lte", "contains", "exists"]

# Campos que vivem na raiz do payload; qualquer outro é procurado em metadata.*
TOP_LEVEL_FIELDS = frozenset(
    {
        "chunk_id",
        "document_id",
        "source",
        "filename",
        "file_type",
        "page",
        "section",
        "content_hash",
        "position",
    }
)


class Condition(BaseModel):
    field: str
    op: Operator = "eq"
    value: Any = None

    @property
    def payload_key(self) -> str:
        """Caminho real no payload gravado no vector store."""
        if self.field in TOP_LEVEL_FIELDS or self.field.startswith("metadata."):
            return self.field
        return f"metadata.{self.field}"

    def matches(self, payload: dict[str, Any]) -> bool:
        actual = _resolve(payload, self.payload_key)
        if self.op == "exists":
            return (actual is not None) == bool(self.value)
        if actual is None:
            return False
        return _compare(actual, self.op, self.value)


class FilterSet(BaseModel):
    """Conjunto de condições combinadas com AND."""

    conditions: list[Condition] = Field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.conditions)

    def matches(self, payload: dict[str, Any]) -> bool:
        return all(condition.matches(payload) for condition in self.conditions)

    def as_dict(self) -> dict[str, Any]:
        return {
            condition.field: (
                condition.value if condition.op == "eq" else {condition.op: condition.value}
            )
            for condition in self.conditions
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> FilterSet:
        """Aceita a forma amigável usada na API e pelo query analyzer.

        Exemplos:
            {"country": "Brazil"}
            {"file_type": {"in": ["pdf", "docx"]}}
            {"page": {"gte": 10}}
        """
        conditions: list[Condition] = []
        for field, value in (raw or {}).items():
            if isinstance(value, dict):
                for op, operand in value.items():
                    conditions.append(Condition(field=field, op=op, value=operand))  # type: ignore[arg-type]
            elif isinstance(value, list):
                conditions.append(Condition(field=field, op="in", value=value))
            else:
                conditions.append(Condition(field=field, op="eq", value=value))
        return cls(conditions=conditions)

    def to_qdrant(self) -> Any | None:
        """Traduz para `qdrant_client.models.Filter` (import preguiçoso)."""
        if not self.conditions:
            return None
        from qdrant_client import models as qmodels

        must: list[Any] = []
        must_not: list[Any] = []
        for condition in self.conditions:
            key = condition.payload_key
            op, value = condition.op, condition.value
            if op == "eq":
                must.append(qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value)))
            elif op == "ne":
                must_not.append(
                    qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))
                )
            elif op == "in":
                must.append(qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=list(value))))
            elif op == "nin":
                must_not.append(
                    qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=list(value)))
                )
            elif op in {"gt", "gte", "lt", "lte"}:
                bounds = {"gt": None, "gte": None, "lt": None, "lte": None}
                bounds[op] = value
                must.append(qmodels.FieldCondition(key=key, range=qmodels.Range(**bounds)))
            elif op == "contains":
                must.append(qmodels.FieldCondition(key=key, match=qmodels.MatchText(text=str(value))))
            elif op == "exists":
                empty = qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key=key))
                (must_not if value else must).append(empty)
        return qmodels.Filter(must=must or None, must_not=must_not or None)


def _resolve(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _compare(actual: Any, op: Operator, expected: Any) -> bool:
    if op == "eq":
        return _norm(actual) == _norm(expected)
    if op == "ne":
        return _norm(actual) != _norm(expected)
    if op == "in":
        return _norm(actual) in {_norm(item) for item in expected}
    if op == "nin":
        return _norm(actual) not in {_norm(item) for item in expected}
    if op == "contains":
        return str(expected).lower() in str(actual).lower()
    try:
        left, right = float(actual), float(expected)
    except (TypeError, ValueError):
        return False
    return {
        "gt": left > right,
        "gte": left >= right,
        "lt": left < right,
        "lte": left <= right,
    }.get(op, False)


def _norm(value: Any) -> Any:
    return value.lower() if isinstance(value, str) else value
