"""Safe loading and content-based type detection for uploaded files.

Accepted uploads (MVP): PDF and DOCX. Standalone images (JPEG/PNG) are rejected as unsupported.

Rules:
  * files are only READ, into memory, with a size limit; nothing is executed or written
  * paths are resolved (symlinks followed) and must stay inside the allowed roots, when configured
  * the extension must be a supported one AND the file content must match it (magic bytes /
    package structure); the extension alone is never trusted
  * DOCX packages are checked for macro-enabled content and for archive-bomb sizes before parsing
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from sanad.config import DocumentProcessingSettings
from sanad.models.documents import DocumentStatus, FileType
from sanad.parsers.errors import DocumentErrorCode, DocumentParsingError

EXTENSIONS: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
}
MIME_TYPES: dict[FileType, str] = {
    FileType.PDF: "application/pdf",
    FileType.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
_DOCX_MAIN = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
_MACRO_MAIN = "application/vnd.ms-word.document.macroenabled.main+xml"
_FILENAME_UNSAFE = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class DocumentSource:
    data: bytes
    filename: str
    file_type: FileType
    mime_type: str
    sha256: str

    @property
    def document_id(self) -> str:
        return f"doc_{self.sha256[:16]}"

    @property
    def size_bytes(self) -> int:
        return len(self.data)


def safe_filename(name: str | os.PathLike | None) -> str:
    """Base name only (both / and \\ separators), control characters removed, length-limited."""
    raw = str(name or "")
    base = PureWindowsPath(PurePosixPath(raw).name).name
    base = _FILENAME_UNSAFE.sub("", base).strip()
    if base in ("", ".", ".."):
        base = "unnamed"
    return base[:255]


def read_path(path: str | os.PathLike, settings: DocumentProcessingSettings) -> tuple[bytes, str]:
    display_name = safe_filename(path)
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError):
        raise DocumentParsingError(DocumentStatus.FILE_NOT_FOUND, DocumentErrorCode.FILE_NOT_FOUND,
                                   f"File '{display_name}' was not found.", stage="loading") from None
    except (OSError, RuntimeError) as exc:  # e.g. symlink loops, permission errors
        raise DocumentParsingError(DocumentStatus.FILE_NOT_FOUND, DocumentErrorCode.FILE_NOT_FOUND,
                                   f"File '{display_name}' could not be accessed.", stage="loading", cause=exc) from None

    if settings.allowed_roots is not None and not any(
        resolved == root or root in resolved.parents for root in settings.allowed_roots
    ):
        raise DocumentParsingError(DocumentStatus.PATH_NOT_ALLOWED, DocumentErrorCode.PATH_NOT_ALLOWED,
                                   f"File '{display_name}' is outside the allowed upload directory.", stage="loading")

    try:
        info = resolved.stat()
    except OSError as exc:
        raise DocumentParsingError(DocumentStatus.FILE_NOT_FOUND, DocumentErrorCode.FILE_NOT_FOUND,
                                   f"File '{display_name}' could not be accessed.", stage="loading", cause=exc) from None
    if not stat.S_ISREG(info.st_mode):
        raise DocumentParsingError(DocumentStatus.FILE_NOT_FOUND, DocumentErrorCode.NOT_A_REGULAR_FILE,
                                   f"'{display_name}' is not a regular file.", stage="loading")
    if info.st_size > settings.max_file_size_bytes:
        raise _too_large(display_name, settings)

    with open(resolved, "rb") as handle:  # read-only; the upload is never executed or modified
        data = handle.read(settings.max_file_size_bytes + 1)
    if len(data) > settings.max_file_size_bytes:
        raise _too_large(display_name, settings)
    return data, display_name


def _too_large(display_name: str, settings: DocumentProcessingSettings) -> DocumentParsingError:
    limit_mb = settings.max_file_size_bytes / (1024 * 1024)
    return DocumentParsingError(DocumentStatus.FILE_TOO_LARGE, DocumentErrorCode.FILE_TOO_LARGE,
                                f"File '{display_name}' exceeds the {limit_mb:g} MB limit.", stage="loading")


def _sniff(data: bytes) -> FileType:
    if b"%PDF-" in data[:1024]:
        return FileType.PDF
    if data.startswith(b"PK\x03\x04"):
        return FileType.DOCX  # confirmed by _check_docx_package
    return FileType.UNKNOWN


def _check_docx_package(data: bytes, display_name: str, settings: DocumentProcessingSettings) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > settings.max_docx_entries:
                raise DocumentParsingError(DocumentStatus.FILE_TOO_LARGE, DocumentErrorCode.ARCHIVE_TOO_LARGE,
                                           f"'{display_name}' contains too many package entries.", stage="validation")
            if sum(entry.file_size for entry in entries) > settings.max_docx_uncompressed_bytes:
                raise DocumentParsingError(DocumentStatus.FILE_TOO_LARGE, DocumentErrorCode.ARCHIVE_TOO_LARGE,
                                           f"'{display_name}' expands beyond the allowed size.", stage="validation")
            names = {entry.filename for entry in entries}
            if "[Content_Types].xml" not in names:
                raise _mismatch(display_name, "is a ZIP archive but not a Word document")
            content_types = archive.read("[Content_Types].xml").decode("utf-8", errors="replace").lower()
    except zipfile.BadZipFile as exc:
        raise DocumentParsingError(DocumentStatus.EXTRACTION_FAILED, DocumentErrorCode.MALFORMED_DOCX,
                                   f"'{display_name}' is not a valid DOCX package.", stage="validation", cause=exc) from None
    if _MACRO_MAIN in content_types or "vbaproject" in content_types:
        raise DocumentParsingError(DocumentStatus.UNSUPPORTED_FILE_TYPE, DocumentErrorCode.MACRO_ENABLED_DOCUMENT,
                                   f"'{display_name}' is a macro-enabled document, which is not accepted.", stage="validation")
    if _DOCX_MAIN not in content_types:
        raise _mismatch(display_name, "is a ZIP-based file but not a Word .docx document")


def _mismatch(display_name: str, detail: str) -> DocumentParsingError:
    return DocumentParsingError(DocumentStatus.UNSUPPORTED_FILE_TYPE, DocumentErrorCode.CONTENT_TYPE_MISMATCH,
                                f"'{display_name}' {detail}.", stage="validation")


def identify(data: bytes, filename: str | None, settings: DocumentProcessingSettings,
             declared_mime_type: str | None = None) -> DocumentSource:
    display_name = safe_filename(filename)
    if len(data) > settings.max_file_size_bytes:
        raise _too_large(display_name, settings)
    extension = Path(display_name).suffix.lower()
    expected = EXTENSIONS.get(extension)
    if expected is None:
        supported = ", ".join(sorted(EXTENSIONS))
        raise DocumentParsingError(DocumentStatus.UNSUPPORTED_FILE_TYPE, DocumentErrorCode.UNSUPPORTED_FILE_TYPE,
                                   f"'{display_name}': unsupported file type. Supported: {supported}.", stage="validation")
    if not data:
        raise DocumentParsingError(DocumentStatus.EMPTY_DOCUMENT, DocumentErrorCode.EMPTY_FILE,
                                   f"'{display_name}' is empty.", stage="validation")
    detected = _sniff(data)
    if detected is not expected:
        raise _mismatch(display_name, f"has a {extension} extension but its content is not a {expected.value.upper()} file")
    if expected is FileType.DOCX:
        _check_docx_package(data, display_name, settings)
    mime_type = MIME_TYPES[expected]
    if declared_mime_type is not None:
        declared = declared_mime_type.split(";")[0].strip().lower()
        if declared not in (mime_type, "application/octet-stream"):
            raise _mismatch(display_name, f"was declared as '{declared}' but its content is {mime_type}")
    return DocumentSource(data=data, filename=display_name, file_type=expected, mime_type=mime_type,
                          sha256=hashlib.sha256(data).hexdigest())
