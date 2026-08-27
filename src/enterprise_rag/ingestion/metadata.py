"""Extração de metadados.

Metadado não é enfeite: é o que transforma "buscar em tudo" em "buscar onde
faz sentido". A pergunta "quais barragens brasileiras..." só fica barata se
`country=Brazil` puder ser aplicado antes do vetor.

Três fontes, em ordem crescente de prioridade:

1. **convenção de caminho e nome** -- `datasets/brazil/usina-x/relatorio.pdf`;
2. **frontmatter** do próprio documento (Markdown);
3. **sidecar explícito** -- `<arquivo>.meta.json`, ou `_metadata.json` valendo
   para o diretório inteiro.

O explícito sempre vence a heurística: inferência erra, e metadado errado é
pior do que metadado ausente, porque filtra silenciosamente o documento certo.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DIRECTORY_METADATA_FILE = "_metadata.json"

COUNTRY_HINTS: dict[str, str] = {
    # inclui os gentilicios: a pergunta diz "barragens brasileiras",
    # nunca "barragens do Brazil"
    "brazil": "Brazil",
    "brasil": "Brazil",
    "brasileira": "Brazil",
    "brasileiras": "Brazil",
    "brasileiro": "Brazil",
    "brasileiros": "Brazil",
    "brazilian": "Brazil",
    "br": "Brazil",
    "portugal": "Portugal",
    "portuguesa": "Portugal",
    "chile": "Chile",
    "chilena": "Chile",
    "chilenas": "Chile",
    "chileno": "Chile",
    "chilenos": "Chile",
    "peru": "Peru",
    "colombia": "Colombia",
    "argentina": "Argentina",
    "mexico": "Mexico",
    "usa": "United States",
    "eua": "United States",
}

DOCUMENT_TYPE_HINTS: dict[str, str] = {
    "relatorio": "technical_report",
    "report": "technical_report",
    "manual": "manual",
    "procedimento": "procedure",
    "procedure": "procedure",
    "politica": "policy",
    "policy": "policy",
    "norma": "standard",
    "standard": "standard",
    "ficha": "datasheet",
    "datasheet": "datasheet",
    "especificacao": "specification",
    "spec": "specification",
    "ata": "meeting_minutes",
    "contrato": "contract",
}

YEAR_RE = re.compile(r"(19|20)\d{2}")


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = "".join(char for char in normalized if not unicodedata.combining(char))
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_only).strip("-").lower()
    return slug or "documento"


def build_document_id(path: Path, root: Path | None = None) -> str:
    """Id estável e legível, derivado do caminho relativo à raiz de ingestão.

    Estável importa: é ele que liga a versão nova do arquivo à antiga e
    permite substituir os chunks em vez de duplicá-los.
    """
    try:
        relative = path.resolve().relative_to(root.resolve()) if root else Path(path.name)
    except ValueError:
        relative = Path(path.name)
    parts = list(relative.with_suffix("").parts)
    return "/".join(slugify(part) for part in parts)


def extract_metadata(
    path: Path,
    root: Path | None = None,
    frontmatter: dict[str, Any] | None = None,
    sample_text: str = "",
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "file_type": path.suffix.lower().lstrip("."),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "modified_at": _modified_at(path),
        "path": str(path),
    }

    tokens = _path_tokens(path, root)
    metadata.update(_infer_from_tokens(tokens))

    year = _infer_year(tokens, sample_text)
    if year:
        metadata["year"] = year

    language = _infer_language(sample_text)
    if language:
        metadata["language"] = language

    # ordem crescente de prioridade: heurística < frontmatter < sidecar
    metadata.update(_clean(frontmatter or {}))
    metadata.update(_clean(_load_sidecars(path)))
    return metadata


def _path_tokens(path: Path, root: Path | None) -> list[str]:
    try:
        relative = path.resolve().relative_to(root.resolve()) if root else Path(path.name)
    except ValueError:
        relative = Path(path.name)
    raw = " ".join(relative.with_suffix("").parts)
    normalized = unicodedata.normalize("NFKD", raw.lower())
    ascii_only = "".join(char for char in normalized if not unicodedata.combining(char))
    return [token for token in re.split(r"[^a-z0-9]+", ascii_only) if token]


def _infer_from_tokens(tokens: list[str]) -> dict[str, Any]:
    inferred: dict[str, Any] = {}
    for token in tokens:
        if "country" not in inferred and token in COUNTRY_HINTS:
            inferred["country"] = COUNTRY_HINTS[token]
        if "document_type" not in inferred and token in DOCUMENT_TYPE_HINTS:
            inferred["document_type"] = DOCUMENT_TYPE_HINTS[token]
    return inferred


def _infer_year(tokens: list[str], sample_text: str) -> int | None:
    for token in tokens:
        if YEAR_RE.fullmatch(token):
            return int(token)
    match = YEAR_RE.search(sample_text[:2000])
    return int(match.group(0)) if match else None


def _infer_language(sample_text: str) -> str | None:
    """Detecção de idioma por stopwords. Suficiente para pt/en e sem dependência."""
    if not sample_text.strip():
        return None
    lowered = sample_text.lower()
    pt_markers = sum(lowered.count(f" {word} ") for word in ("de", "que", "para", "com", "nao"))
    en_markers = sum(lowered.count(f" {word} ") for word in ("the", "of", "and", "for", "with"))
    if pt_markers == en_markers == 0:
        return None
    return "pt" if pt_markers >= en_markers else "en"


def _load_sidecars(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    directory_file = path.parent / DIRECTORY_METADATA_FILE
    if directory_file.exists():
        metadata.update(_read_json(directory_file))
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if sidecar.exists():
        metadata.update(_read_json(sidecar))
    return metadata


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _clean(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in raw.items() if value not in (None, "", [], {})}


def _modified_at(path: Path) -> str:
    if not path.exists():
        return datetime.now(UTC).isoformat()
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
