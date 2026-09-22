"""Structured representation of an uploaded document (Phase 3).

A ParsedDocument only contains what was read from the file. Nothing is inferred or generated:
text that could not be read (e.g. a scanned PDF page; Sanad's MVP performs no OCR) is reported through an explicit
status, never replaced by guessed content.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from models.common import ErrorInfo


class DocumentStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    EMPTY_DOCUMENT = "empty_document"
    OCR_REQUIRED = "ocr_required"
    UNSUPPORTED_FILE_TYPE = "unsupported_file_type"
    FILE_NOT_FOUND = "file_not_found"
    FILE_TOO_LARGE = "file_too_large"
    PATH_NOT_ALLOWED = "path_not_allowed"
    EXTRACTION_FAILED = "extraction_failed"

    @property
    def has_text(self) -> bool:
        return self in (DocumentStatus.SUCCESS, DocumentStatus.PARTIAL)


class FileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    PDF_TEXT_LAYER = "pdf_text_layer"
    DOCX_XML = "docx_xml"


class SectionType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    HEADER = "header"
    FOOTER = "footer"


class PageStatus(str, Enum):
    TEXT = "text"
    OCR_REQUIRED = "ocr_required"
    NO_TEXT = "no_text"


class DocumentSection(BaseModel):
    section_id: str
    order: int = Field(ge=0, description="Position in reading order, shared with tables.")
    section_type: SectionType
    title: str | None = Field(default=None, description="Nearest preceding heading (or the heading itself).")
    text: str
    page_number: int | None = Field(default=None, description="1-based; None for DOCX (no reliable pages).")
    heading_level: int | None = None
    heading_id: str | None = Field(default=None, description="section_id of the nearest preceding heading.")
    style: str | None = None
    text_direction: Literal["rtl", "ltr"] | None = None
    extraction_method: ExtractionMethod


class DocumentTable(BaseModel):
    table_id: str
    order: int = Field(ge=0)
    page_number: int | None = None
    title: str | None = Field(default=None, description="Nearest preceding heading.")
    heading_id: str | None = None
    rows: list[list[str]]
    parent_table_id: str | None = Field(default=None, description="Set for tables nested inside a table cell.")
    extraction_method: ExtractionMethod

    @property
    def row_texts(self) -> list[str]:
        """Each row as it appears in ParsedDocument.full_text."""
        return [row_text(row) for row in self.rows]


class DocumentPage(BaseModel):
    page_number: int = Field(ge=1)
    status: PageStatus
    text_char_count: int = Field(ge=0, description="Letters and digits read from this page.")
    has_images: bool = False
    appears_blank: bool | None = Field(default=None, description="Only checked for pages without a text layer.")
    extraction_method: ExtractionMethod | None = None


class OcrInfo(BaseModel):
    """Whether the document would need OCR. Sanad's MVP never performs OCR: such content is reported, not read."""

    required: bool = False
    pages: list[int] = Field(default_factory=list, description="PDF pages without a usable text layer.")


class ParsedDocument(BaseModel):
    document_id: str = Field(description="doc_ + first 16 hex chars of the content SHA-256.")
    filename: str = Field(description="Base name only; directory components are never kept.")
    file_type: FileType
    mime_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    status: DocumentStatus
    language: Literal["ar", "en", "mixed", "unknown"] = "unknown"
    page_count: int | None = None
    pages: list[DocumentPage] = Field(default_factory=list)
    sections: list[DocumentSection] = Field(default_factory=list)
    tables: list[DocumentTable] = Field(default_factory=list)
    full_text: str = ""
    ocr: OcrInfo = Field(default_factory=OcrInfo)
    extraction_warnings: list[str] = Field(default_factory=list)
    errors: list[ErrorInfo] = Field(default_factory=list)
    parser: str | None = None

    @model_validator(mode="after")
    def _status_invariants(self):
        if self.status.has_text and not self.full_text.strip():
            raise ValueError(f"status '{self.status.value}' requires extracted text")
        if not self.status.has_text and (self.sections or self.tables or self.full_text.strip()):
            raise ValueError(f"status '{self.status.value}' must not carry extracted content")
        failure = {
            DocumentStatus.UNSUPPORTED_FILE_TYPE, DocumentStatus.FILE_NOT_FOUND, DocumentStatus.FILE_TOO_LARGE,
            DocumentStatus.PATH_NOT_ALLOWED, DocumentStatus.EXTRACTION_FAILED,
        }
        if self.status in failure and not self.errors:
            raise ValueError(f"status '{self.status.value}' requires an ErrorInfo")
        return self

    def blocks(self) -> list[DocumentSection | DocumentTable]:
        """Sections and tables in reading order."""
        return sorted([*self.sections, *self.tables], key=lambda block: block.order)


def row_text(row: list[str]) -> str:
    return " | ".join(cell for cell in row)
