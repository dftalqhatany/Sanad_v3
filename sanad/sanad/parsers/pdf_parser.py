"""PDF parsing with PyMuPDF (the PDF library the existing hr_assistant project already uses).

text-layer PDF  -> sections/tables with page numbers
no text layer   -> OCR_REQUIRED (Sanad's MVP performs no OCR); text is never fabricated
"""

from __future__ import annotations

from collections import Counter

from sanad.models.documents import (
    DocumentPage,
    DocumentStatus,
    ExtractionMethod,
    FileType,
    OcrInfo,
    PageStatus,
    ParsedDocument,
    SectionType,
)
from sanad.parsers.base import BaseDocumentParser, DocumentBuilder
from sanad.parsers.detection import DocumentSource
from sanad.parsers.errors import DocumentErrorCode, DocumentParsingError
from sanad.parsers.normalization import count_text_chars

_BULLETS = ("•", "●", "▪", "◦", "‣", "-", "–", "*")
_BOLD_FLAG = 16


def _load_pymupdf():
    try:
        import pymupdf  # PyMuPDF >= 1.24.3
    except ImportError:
        try:
            import fitz as pymupdf  # older PyMuPDF (hr_assistant pins 1.23.8)
        except ImportError as exc:
            raise DocumentParsingError(DocumentStatus.EXTRACTION_FAILED, DocumentErrorCode.DEPENDENCY_MISSING,
                                       "PDF support requires the PyMuPDF package.", cause=exc) from None
    silence = getattr(pymupdf, "no_recommend_layout", None)  # newer PyMuPDF prints an advert on stdout
    if callable(silence):
        silence()
    tools = getattr(pymupdf, "TOOLS", None)  # damaged files are reported through statuses, not stderr noise
    if tools is not None and hasattr(tools, "mupdf_display_errors"):
        tools.mupdf_display_errors(False)
    return pymupdf


class PdfParser(BaseDocumentParser):
    name = "pdf/pymupdf"
    file_types = (FileType.PDF,)

    def parse(self, source: DocumentSource) -> ParsedDocument:
        fitz = _load_pymupdf()
        try:
            document = fitz.open(stream=source.data, filetype="pdf")
        except Exception as exc:
            raise DocumentParsingError(DocumentStatus.EXTRACTION_FAILED, DocumentErrorCode.MALFORMED_PDF,
                                       f"'{source.filename}' could not be opened as a PDF.", cause=exc) from None
        try:
            return self._parse_document(fitz, document, source)
        finally:
            document.close()

    # ------------------------------------------------------------------ document
    def _parse_document(self, fitz, document, source: DocumentSource) -> ParsedDocument:
        if document.needs_pass:
            raise DocumentParsingError(DocumentStatus.EXTRACTION_FAILED, DocumentErrorCode.PASSWORD_PROTECTED,
                                       f"'{source.filename}' is password-protected.")
        total = document.page_count
        if total == 0:
            raise DocumentParsingError(DocumentStatus.EXTRACTION_FAILED, DocumentErrorCode.MALFORMED_PDF,
                                       f"'{source.filename}' contains no readable pages.")
        builder = DocumentBuilder(source, self.name, map_arabic_presentation_forms=True)
        builder.page_count = total
        partial_reasons: list[str] = []
        if getattr(document, "is_repaired", False):
            builder.warn("The PDF structure was damaged and repaired while opening; some content may be missing.")
            partial_reasons.append("repaired")
        limit = min(total, self.settings.max_pdf_pages)
        if total > limit:
            builder.warn(f"Only the first {limit} of {total} pages were processed (page limit).")
            partial_reasons.append("page_limit")

        flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP
        page_dicts = []
        for index in range(limit):
            page = document.load_page(index)
            page_dicts.append((page, page.get_text("dict", flags=flags, sort=True)))
        body_size = self._body_font_size(page_dicts)

        ocr_pages: list[int] = []
        for page, content in page_dicts:
            number = page.number + 1
            chars = self._add_page_content(fitz, page, content, number, builder, body_size)
            has_images = bool(page.get_images())
            if chars >= self.settings.min_page_text_chars:
                builder.add_page(DocumentPage(page_number=number, status=PageStatus.TEXT, text_char_count=chars,
                                              has_images=has_images, extraction_method=ExtractionMethod.PDF_TEXT_LAYER))
                continue
            blank = self._appears_blank(fitz, page)
            if blank and not has_images:
                builder.add_page(DocumentPage(page_number=number, status=PageStatus.NO_TEXT, text_char_count=chars,
                                              has_images=has_images, appears_blank=True))
                continue
            ocr_pages.append(number)  # reported, never OCR'd
            builder.add_page(DocumentPage(page_number=number, status=PageStatus.OCR_REQUIRED,
                                          text_char_count=chars, has_images=has_images, appears_blank=blank))

        builder.ocr = OcrInfo(required=bool(ocr_pages), pages=ocr_pages)
        return self._finish(builder, partial_reasons)

    def _finish(self, builder: DocumentBuilder, partial_reasons: list[str]) -> ParsedDocument:
        pages = builder.pages
        text_pages = [p for p in pages if p.status is PageStatus.TEXT]
        needs_ocr = [p.page_number for p in pages if p.status is PageStatus.OCR_REQUIRED]
        if needs_ocr:
            builder.warn(f"Page(s) {_pages(needs_ocr)} have no usable text layer and need OCR; "
                         "OCR is not part of Sanad's MVP, so their content was not extracted.")
        if text_pages:
            status = DocumentStatus.PARTIAL if (needs_ocr or partial_reasons) else DocumentStatus.SUCCESS
            return builder.build(status)
        if pages and all(p.appears_blank for p in pages):
            builder.warn("No text layer was found and the pages appear blank when rendered.")
        else:
            builder.warn("No usable text layer was found in this PDF.")
        if builder.sections or builder.tables:
            builder.warn("A few characters were present but below the text threshold; they were not treated as document text.")
        builder.ocr.required = True
        builder.ocr.pages = [p.page_number for p in pages]
        return builder.build(DocumentStatus.OCR_REQUIRED)

    # ------------------------------------------------------------------ page content
    @staticmethod
    def _body_font_size(page_dicts) -> float:
        sizes: Counter[float] = Counter()
        for _, content in page_dicts:
            for block in content.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        sizes[round(span.get("size", 0) * 2) / 2] += len(span.get("text", "").strip())
        return sizes.most_common(1)[0][0] if sizes else 0.0

    def _add_page_content(self, fitz, page, content, number: int, builder: DocumentBuilder, body_size: float) -> int:
        tables = self._find_tables(page, builder)
        items = []
        for table in tables:
            items.append((table.bbox[1], table.bbox[0], "table", table))
        for block in content.get("blocks", []):
            if block.get("type") != 0 or not block.get("lines"):
                continue
            if any(_mostly_inside(block["bbox"], table.bbox) for table in tables):
                continue  # this text is already captured by the table
            items.append((block["bbox"][1], block["bbox"][0], "block", block))
        items.sort(key=lambda item: (round(item[0], 1), item[1]))

        chars = 0
        for _, _, kind, item in items:
            if kind == "table":
                try:
                    rows = item.extract()
                except Exception:  # table extraction is best-effort
                    builder.warn(f"A table on page {number} could not be extracted.")
                    continue
                table = builder.add_table(rows, ExtractionMethod.PDF_TEXT_LAYER, page_number=number)
                if table is not None:
                    chars += sum(count_text_chars(cell) for row in table.rows for cell in row)
            else:
                chars += self._add_block(item, number, builder, body_size)
        return chars

    @staticmethod
    def _find_tables(page, builder: DocumentBuilder) -> list:
        finder = getattr(page, "find_tables", None)
        if finder is None:
            builder.warn("Table detection is not available in this PyMuPDF version; tables were read as text.")
            return []
        try:
            return list(finder().tables)
        except Exception:
            builder.warn(f"Table detection failed on page {page.number + 1}; tables were read as text.")
            return []

    def _add_block(self, block, number: int, builder: DocumentBuilder, body_size: float) -> int:
        lines = []
        for line in block["lines"]:
            spans = [span for span in line.get("spans", []) if span.get("text")]
            text = "".join(span["text"] for span in spans)
            if text.strip():
                lines.append((text, spans))
        if not lines:
            return 0
        block_text = "\n".join(text for text, _ in lines)
        spans = [span for _, line_spans in lines for span in line_spans if span["text"].strip()]
        max_size = max((span.get("size", 0) for span in spans), default=0)
        all_bold = bool(spans) and all(span.get("flags", 0) & _BOLD_FLAG for span in spans)
        stripped = block_text.strip()
        is_heading = len(lines) <= 2 and len(stripped) <= 150 and (
            (body_size and max_size >= body_size * 1.15) or (all_bold and len(stripped) <= 80)
        )
        chars = 0
        if is_heading:
            section = builder.add_section(block_text.replace("\n", " "), SectionType.HEADING,
                                          ExtractionMethod.PDF_TEXT_LAYER, page_number=number)
            return count_text_chars(section.text) if section else 0

        paragraph: list[str] = []
        bullet: list[str] = []

        def flush() -> int:
            added = 0
            if paragraph:
                section = builder.add_section("\n".join(paragraph), SectionType.PARAGRAPH,
                                              ExtractionMethod.PDF_TEXT_LAYER, page_number=number)
                added += count_text_chars(section.text) if section else 0
                paragraph.clear()
            if bullet:
                section = builder.add_section(" ".join(bullet), SectionType.LIST_ITEM,
                                              ExtractionMethod.PDF_TEXT_LAYER, page_number=number)
                added += count_text_chars(section.text) if section else 0
                bullet.clear()
            return added

        for text, _ in lines:
            if text.lstrip().startswith(_BULLETS) and len(text.strip()) > 1:
                chars += flush()
                bullet.append(text.strip())
            elif bullet:
                bullet.append(text.strip())
            else:
                paragraph.append(text)
        chars += flush()
        return chars

    @staticmethod
    def _appears_blank(fitz, page) -> bool:
        try:
            pixmap = page.get_pixmap(dpi=40, colorspace=fitz.csGRAY, alpha=False)
            return min(pixmap.samples) >= 250
        except Exception:
            return False


def _mostly_inside(inner, outer, threshold: float = 0.8) -> bool:
    x0, y0, x1, y1 = inner
    ox0, oy0, ox1, oy1 = outer
    width, height = max(0.0, min(x1, ox1) - max(x0, ox0)), max(0.0, min(y1, oy1) - max(y0, oy0))
    area = max((x1 - x0) * (y1 - y0), 1e-6)
    return (width * height) / area >= threshold


def _pages(numbers: list[int]) -> str:
    return ", ".join(str(n) for n in numbers)
