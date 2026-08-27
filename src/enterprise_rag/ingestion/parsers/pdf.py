"""PDF via PyMuPDF, com OCR opcional.

Dois detalhes que separam um parser utilizável de um `extract_text()` ingênuo:

1. **página é metadado obrigatório.** Sem ela não existe citação verificável,
   e citação verificável é o requisito número um de RAG corporativo.
2. **página sem texto não é página vazia.** Documentos digitalizados chegam
   como imagem; quando o Tesseract está disponível, caímos para OCR e
   marcamos o bloco com `kind="ocr"`, para que a origem do texto seja
   auditável depois.

Cabeçalhos são inferidos por tamanho de fonte: blocos com fonte
significativamente maior que a mediana da página viram `section`.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from enterprise_rag.config.logging import get_logger
from enterprise_rag.ingestion.parsers.base import DocumentParser, ParserError
from enterprise_rag.models.document import ParsedBlock

logger = get_logger(__name__)

MIN_CHARS_FOR_TEXT_PAGE = 40


class PDFParser(DocumentParser):
    name = "pdf"
    extensions = (".pdf",)

    def __init__(self, enable_ocr: bool = True, ocr_language: str = "por+eng") -> None:
        self.enable_ocr = enable_ocr
        self.ocr_language = ocr_language

    def parse(self, path: Path) -> list[ParsedBlock]:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ParserError(
                "PyMuPDF nao instalado; rode: pip install '.[parsers]'"
            ) from exc

        blocks: list[ParsedBlock] = []
        with fitz.open(path) as document:
            for page_number, page in enumerate(document, start=1):
                page_blocks = self._parse_page(page, page_number)
                if not page_blocks or _total_chars(page_blocks) < MIN_CHARS_FOR_TEXT_PAGE:
                    ocr_text = self._ocr_page(page)
                    if ocr_text:
                        page_blocks = [
                            ParsedBlock(text=ocr_text, page=page_number, kind="ocr")
                        ]
                blocks.extend(page_blocks)
        if not blocks:
            raise ParserError(f"nenhum texto extraido de {path.name}")
        return blocks

    def _parse_page(self, page: object, page_number: int) -> list[ParsedBlock]:
        data = page.get_text("dict")  # type: ignore[attr-defined]
        spans: list[tuple[str, float]] = []
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(span.get("text", "") for span in line.get("spans", []))
                if not text.strip():
                    continue
                size = max((span.get("size", 0.0) for span in line.get("spans", [])), default=0.0)
                spans.append((text.strip(), size))

        if not spans:
            return []

        sizes = [size for _, size in spans]
        median = statistics.median(sizes)
        heading_threshold = median * 1.15

        blocks: list[ParsedBlock] = []
        section: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            text = " ".join(buffer).strip()
            buffer.clear()
            if text:
                blocks.append(
                    ParsedBlock(text=text, page=page_number, section=section, kind="paragraph")
                )

        for text, size in spans:
            is_heading = size >= heading_threshold and len(text) < 120
            if is_heading:
                flush()
                section = text
                blocks.append(
                    ParsedBlock(text=text, page=page_number, section=section, kind="heading")
                )
            else:
                buffer.append(text)
        flush()
        return blocks

    def _ocr_page(self, page: object) -> str:
        if not self.enable_ocr:
            return ""
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            return ""
        try:
            pixmap = page.get_pixmap(dpi=200)  # type: ignore[attr-defined]
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            return str(pytesseract.image_to_string(image, lang=self.ocr_language)).strip()
        except Exception as exc:
            logger.warning("OCR falhou", extra={"erro": f"{type(exc).__name__}: {exc}"})
            return ""

    def page_count(self, path: Path) -> int | None:
        try:
            import fitz
        except ImportError:
            return None
        with fitz.open(path) as document:
            return int(document.page_count)


def _total_chars(blocks: list[ParsedBlock]) -> int:
    return sum(len(block.text) for block in blocks)
