"""Entry point of the parser layer: validate an upload, pick the parser, always return a ParsedDocument.

Document problems (unsupported type, missing file, scanned PDF, corrupt file...) are returned as
explicit statuses with sanitized ErrorInfo; they are never raised to the caller and never logged
with document content, filenames or personal data.

Accepted uploads (MVP): PDF and DOCX. JPEG/PNG and other types return UNSUPPORTED_FILE_TYPE; no OCR is
performed anywhere.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path

from config import DocumentProcessingSettings
from models.documents import DocumentStatus, FileType, ParsedDocument
from parsers.base import BaseDocumentParser
from parsers.detection import EXTENSIONS, DocumentSource, identify, read_path, safe_filename
from parsers.docx_parser import DocxParser
from parsers.errors import DocumentErrorCode, DocumentParsingError
from parsers.pdf_parser import PdfParser

logger = logging.getLogger(f"sanad.{__name__}")  # one "sanad" logging namespace, as before the flattening

DEFAULT_PARSERS: tuple[type[BaseDocumentParser], ...] = (PdfParser, DocxParser)


class DocumentProcessor:
    def __init__(
        self,
        settings: DocumentProcessingSettings | None = None,
        parser_classes: Iterable[type[BaseDocumentParser]] = DEFAULT_PARSERS,
    ) -> None:
        self.settings = settings or DocumentProcessingSettings.from_env()
        self._parsers: dict[FileType, BaseDocumentParser] = {}
        for parser_class in parser_classes:
            parser = parser_class(self.settings)
            for file_type in parser.file_types:
                self._parsers[file_type] = parser

    @property
    def supported_file_types(self) -> tuple[FileType, ...]:
        return tuple(self._parsers)

    def parse_path(self, path: str | os.PathLike) -> ParsedDocument:
        try:
            data, filename = read_path(path, self.settings)
        except DocumentParsingError as exc:
            return self._failed(exc, filename=safe_filename(path))
        return self.parse_bytes(data, filename)

    def parse_bytes(self, data: bytes, filename: str, declared_mime_type: str | None = None) -> ParsedDocument:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes")
        started = time.perf_counter()
        data = bytes(data)
        try:
            source = identify(data, filename, self.settings, declared_mime_type)
        except DocumentParsingError as exc:
            return self._failed(exc, filename=safe_filename(filename), data=data)

        parser = self._parsers.get(source.file_type)
        if parser is None:
            error = DocumentParsingError(DocumentStatus.UNSUPPORTED_FILE_TYPE, DocumentErrorCode.UNSUPPORTED_FILE_TYPE,
                                         f"No parser is registered for {source.file_type.value} files.", stage="validation")
            return self._failed(error, source=source)
        try:
            document = parser.parse(source)
        except DocumentParsingError as exc:
            document = self._failed(exc, source=source)
        except Exception as exc:  # unexpected library failure -> explicit, sanitized status
            logger.warning("Unexpected %s while parsing document %s", type(exc).__name__, source.document_id)
            logger.debug("Parser failure for document %s", source.document_id, exc_info=True)
            error = DocumentParsingError(DocumentStatus.EXTRACTION_FAILED, DocumentErrorCode.EXTRACTION_FAILED,
                                         f"Text could not be extracted from '{source.filename}'.", cause=exc)
            document = self._failed(error, source=source)
        logger.info("Parsed document %s type=%s status=%s duration_ms=%.1f", document.document_id,
                    document.file_type.value, document.status.value, (time.perf_counter() - started) * 1000)
        return document

    @staticmethod
    def _failed(error: DocumentParsingError, *, filename: str | None = None, data: bytes | None = None,
                source: DocumentSource | None = None) -> ParsedDocument:
        if source is not None:
            return ParsedDocument(
                document_id=source.document_id, filename=source.filename, file_type=source.file_type,
                mime_type=source.mime_type, size_bytes=source.size_bytes, sha256=source.sha256,
                status=error.status, errors=[error.to_error_info()],
            )
        name = filename or "unnamed"
        digest = hashlib.sha256(data if data is not None else f"unavailable:{name}".encode()).hexdigest()
        return ParsedDocument(
            document_id=f"doc_{digest[:16]}", filename=name,
            file_type=EXTENSIONS.get(Path(name).suffix.lower(), FileType.UNKNOWN),
            size_bytes=len(data) if data is not None else None, sha256=digest if data is not None else None,
            status=error.status, errors=[error.to_error_info()],
        )
