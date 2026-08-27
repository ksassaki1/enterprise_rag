"""Interface de parsing.

O pipeline não sabe o que é um PDF. Ele conhece apenas `supports()` e
`parse()`, e recebe de volta uma lista de `ParsedBlock` (texto + página +
seção + tipo). Adicionar PowerPoint, HTML ou e-mail depois é escrever uma
classe e registrá-la -- nenhuma linha do pipeline muda.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from enterprise_rag.models.document import ParsedBlock


class ParserError(RuntimeError):
    """Falha ao interpretar um arquivo (formato invalido, corrompido, sem texto)."""


class DocumentParser(ABC):
    name: str = "base"
    extensions: tuple[str, ...] = ()

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    @abstractmethod
    def parse(self, path: Path) -> list[ParsedBlock]:
        """Devolve os blocos lógicos do documento, na ordem de leitura."""

    def page_count(self, path: Path) -> int | None:
        return None


class ParserRegistry:
    """Resolve o parser de um arquivo. Ordem de registro define a precedência."""

    def __init__(self, parsers: list[DocumentParser] | None = None) -> None:
        self._parsers: list[DocumentParser] = list(parsers or [])

    def register(self, parser: DocumentParser) -> None:
        self._parsers.append(parser)

    def resolve(self, path: Path) -> DocumentParser | None:
        for parser in self._parsers:
            if parser.supports(path):
                return parser
        return None

    @property
    def supported_extensions(self) -> set[str]:
        return {extension for parser in self._parsers for extension in parser.extensions}

    def __iter__(self):
        return iter(self._parsers)
