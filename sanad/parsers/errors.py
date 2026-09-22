"""Parser error codes and the internal exception used to stop parsing with an explicit status.

User-facing ErrorInfo objects never contain stack traces, absolute paths or document content.
"""

from __future__ import annotations

import re
from enum import Enum

from models.common import ErrorInfo
from models.documents import DocumentStatus


class DocumentErrorCode(str, Enum):
    UNSUPPORTED_FILE_TYPE = "unsupported_file_type"
    CONTENT_TYPE_MISMATCH = "content_type_mismatch"
    MACRO_ENABLED_DOCUMENT = "macro_enabled_document"
    FILE_NOT_FOUND = "file_not_found"
    NOT_A_REGULAR_FILE = "not_a_regular_file"
    PATH_NOT_ALLOWED = "path_not_allowed"
    FILE_TOO_LARGE = "file_too_large"
    ARCHIVE_TOO_LARGE = "archive_too_large"
    EMPTY_FILE = "empty_file"
    PASSWORD_PROTECTED = "password_protected"
    MALFORMED_PDF = "malformed_pdf"
    MALFORMED_DOCX = "malformed_docx"
    DEPENDENCY_MISSING = "dependency_missing"
    EXTRACTION_FAILED = "extraction_failed"


class DocumentParsingError(Exception):
    """Raised inside parsers; converted by the processor into a ParsedDocument with an explicit status."""

    def __init__(
        self,
        status: DocumentStatus,
        code: DocumentErrorCode,
        message: str,
        *,
        stage: str = "parsing",
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.stage = stage
        self.cause = cause

    def to_error_info(self) -> ErrorInfo:
        return ErrorInfo(
            code=self.code.value,
            stage=self.stage,
            message=sanitize_message(self.message),
            exception_type=type(self.cause).__name__ if self.cause is not None else None,
        )


_ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:\\|/)(?:[^\s'\"<>|:]+[/\\])+[^\s'\"<>|:]*")


def sanitize_message(message: str, limit: int = 300) -> str:
    """Remove absolute paths and truncate; library messages may otherwise leak server paths."""
    cleaned = _ABSOLUTE_PATH.sub("<path>", str(message))
    cleaned = " ".join(cleaned.split())
    return cleaned[:limit]
