"""Document processing layer: PDF / DOCX -> ParsedDocument.

Sanad's MVP accepts PDF and DOCX only. Standalone images (JPEG/PNG) are rejected as unsupported, and no
OCR is performed: PDF pages without a text layer are reported as OCR_REQUIRED.
Independent of the regulatory RAG: nothing in this package imports sanad.rag.
"""

from parsers.errors import DocumentErrorCode
from parsers.factory import DocumentProcessor

__all__ = ["DocumentErrorCode", "DocumentProcessor"]
