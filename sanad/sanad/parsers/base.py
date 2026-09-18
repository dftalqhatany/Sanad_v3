"""Shared parser contract and the builder every format parser uses to assemble a ParsedDocument."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from sanad.config import DocumentProcessingSettings
from sanad.models.common import ErrorInfo
from sanad.models.documents import (
    DocumentPage,
    DocumentSection,
    DocumentStatus,
    DocumentTable,
    ExtractionMethod,
    FileType,
    OcrInfo,
    ParsedDocument,
    SectionType,
    row_text,
)
from sanad.parsers.detection import DocumentSource
from sanad.parsers.normalization import NormalizationReport, detect_language, normalize_text


class BaseDocumentParser(ABC):
    """A format parser. Implementations must not execute content, write files or import the RAG."""

    name: str = "base"
    file_types: tuple[FileType, ...] = ()

    def __init__(self, settings: DocumentProcessingSettings) -> None:
        self.settings = settings

    @abstractmethod
    def parse(self, source: DocumentSource) -> ParsedDocument:
        """Parse in memory. Raise DocumentParsingError for explicit failures."""


class DocumentBuilder:
    def __init__(self, source: DocumentSource, parser_name: str, *, map_arabic_presentation_forms: bool = False) -> None:
        self.source = source
        self.parser_name = parser_name
        self.map_presentation_forms = map_arabic_presentation_forms
        self.sections: list[DocumentSection] = []
        self.tables: list[DocumentTable] = []
        self.pages: list[DocumentPage] = []
        self.warnings: list[str] = []
        self.ocr = OcrInfo()
        self.report = NormalizationReport()
        self.page_count: int | None = None
        self._order = 0
        self._heading: DocumentSection | None = None

    # ------------------------------------------------------------------ content
    def normalize(self, text: str) -> str:
        return normalize_text(text, map_arabic_presentation_forms=self.map_presentation_forms, report=self.report)

    def add_section(
        self,
        text: str,
        section_type: SectionType,
        method: ExtractionMethod,
        *,
        page_number: int | None = None,
        heading_level: int | None = None,
        style: str | None = None,
        text_direction: str | None = None,
    ) -> DocumentSection | None:
        text = self.normalize(text)
        if not text:
            return None
        is_heading = section_type is SectionType.HEADING
        section_id = f"s{len(self.sections) + 1:04d}"
        section = DocumentSection(
            section_id=section_id,
            order=self._next_order(),
            section_type=section_type,
            title=text if is_heading else (self._heading.text if self._heading else None),
            text=text,
            page_number=page_number,
            heading_level=heading_level,
            heading_id=section_id if is_heading else (self._heading.section_id if self._heading else None),
            style=style,
            text_direction=text_direction,
            extraction_method=method,
        )
        self.sections.append(section)
        if is_heading:
            self._heading = section
        return section

    def add_table(
        self,
        rows: Sequence[Sequence[str | None]],
        method: ExtractionMethod,
        *,
        page_number: int | None = None,
        parent_table_id: str | None = None,
    ) -> DocumentTable | None:
        cleaned = [[self.normalize(cell or "").replace("\n", " ") for cell in row] for row in rows]
        cleaned = [row for row in cleaned if any(cell for cell in row)]
        if not cleaned:
            return None
        table = DocumentTable(
            table_id=f"t{len(self.tables) + 1:03d}",
            order=self._next_order(),
            page_number=page_number,
            title=self._heading.text if self._heading else None,
            heading_id=self._heading.section_id if self._heading else None,
            rows=cleaned,
            parent_table_id=parent_table_id,
            extraction_method=method,
        )
        self.tables.append(table)
        return table

    def add_page(self, page: DocumentPage) -> None:
        self.pages.append(page)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def _next_order(self) -> int:
        order = self._order
        self._order += 1
        return order

    # ------------------------------------------------------------------ result
    def full_text(self) -> str:
        parts = []
        for block in sorted([*self.sections, *self.tables], key=lambda b: b.order):
            if isinstance(block, DocumentTable):
                parts.append("\n".join(row_text(row) for row in block.rows))
            else:
                parts.append(block.text)
        return "\n\n".join(parts)

    def build(self, status: DocumentStatus, errors: Sequence[ErrorInfo] = ()) -> ParsedDocument:
        full_text = self.full_text() if status.has_text else ""
        return ParsedDocument(
            document_id=self.source.document_id,
            filename=self.source.filename,
            file_type=self.source.file_type,
            mime_type=self.source.mime_type,
            size_bytes=self.source.size_bytes,
            sha256=self.source.sha256,
            status=status,
            language=detect_language(full_text),
            page_count=self.page_count if self.page_count is not None else (len(self.pages) or None),
            pages=self.pages,
            sections=self.sections if status.has_text else [],
            tables=self.tables if status.has_text else [],
            full_text=full_text,
            ocr=self.ocr,
            extraction_warnings=[*self.warnings, *self.report.warnings()],
            errors=list(errors),
            parser=self.parser_name,
        )
