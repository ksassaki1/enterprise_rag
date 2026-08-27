"""Parsers registrados por padrão.

A ordem importa: o primeiro que responder `supports()` vence.
"""

from __future__ import annotations

from enterprise_rag.ingestion.parsers.base import DocumentParser, ParserError, ParserRegistry
from enterprise_rag.ingestion.parsers.docx import DocxParser
from enterprise_rag.ingestion.parsers.pdf import PDFParser
from enterprise_rag.ingestion.parsers.txt import TextParser
from enterprise_rag.ingestion.parsers.xlsx import TabularParser


def default_registry(enable_ocr: bool = True) -> ParserRegistry:
    return ParserRegistry(
        [
            PDFParser(enable_ocr=enable_ocr),
            DocxParser(),
            TabularParser(),
            TextParser(),
        ]
    )


__all__ = [
    "DocumentParser",
    "ParserError",
    "ParserRegistry",
    "PDFParser",
    "DocxParser",
    "TabularParser",
    "TextParser",
    "default_registry",
]
