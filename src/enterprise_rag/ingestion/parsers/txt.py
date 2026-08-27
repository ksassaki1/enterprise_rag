"""Texto puro e Markdown.

Markdown é tratado como formato estruturado de primeira classe: cabeçalhos
`#` viram `section`, o que alimenta o chunking estrutural e as citações.
"""

from __future__ import annotations

import re
from pathlib import Path

from enterprise_rag.ingestion.parsers.base import DocumentParser
from enterprise_rag.models.document import ParsedBlock

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
FRONTMATTER_RE = re.compile(r"^---\s*$")


class TextParser(DocumentParser):
    name = "text"
    extensions = (".txt", ".md", ".markdown", ".log")

    def parse(self, path: Path) -> list[ParsedBlock]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        body, frontmatter = split_frontmatter(raw)

        blocks: list[ParsedBlock] = []
        section: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            text = "\n".join(buffer).strip()
            buffer.clear()
            if text:
                blocks.append(
                    ParsedBlock(text=text, section=section, kind="paragraph", metadata=frontmatter)
                )

        for line in body.splitlines():
            heading = HEADING_RE.match(line)
            if heading:
                flush()
                section = heading.group(2).strip()
                blocks.append(
                    ParsedBlock(
                        text=section,
                        section=section,
                        kind="heading",
                        metadata={**frontmatter, "level": len(heading.group(1))},
                    )
                )
                continue
            if not line.strip():
                flush()
                continue
            buffer.append(line)
        flush()
        return blocks


def split_frontmatter(raw: str) -> tuple[str, dict[str, str]]:
    """Lê frontmatter YAML simples (`chave: valor`) no topo do arquivo.

    Deliberadamente sem PyYAML: só precisamos de pares chave/valor para
    alimentar os metadados de filtro, e um parser de 15 linhas evita mais uma
    dependência no núcleo.
    """
    lines = raw.splitlines()
    if not lines or not FRONTMATTER_RE.match(lines[0]):
        return raw, {}
    for index in range(1, len(lines)):
        if FRONTMATTER_RE.match(lines[index]):
            metadata: dict[str, str] = {}
            for line in lines[1:index]:
                if ":" in line:
                    key, _, value = line.partition(":")
                    metadata[key.strip()] = value.strip().strip("\"'")
            return "\n".join(lines[index + 1 :]), metadata
    return raw, {}
