"""Conservative normalisation (artifacts only; Arabic legal text preserved) and ParsedDocument invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from config import DocumentProcessingSettings
from models.common import ErrorInfo
from models.documents import DocumentStatus, FileType, ParsedDocument
from parsers import DocumentProcessor
from parsers.errors import sanitize_message
from parsers.normalization import NormalizationReport, detect_language, normalize_text

ARABIC_LEGAL = "المادةُ التاسعة بعد المائة: يســتحق العامل إجازةً سنويةً لا تقل عن واحد وعشرين يوماً (٢١)، وتُزاد إلى ثلاثين."


def test_whitespace_line_endings_and_invisible_artifacts_are_cleaned():
    raw = "Basic\u00a0Salary:\u200b  12,000\u00adSAR \r\n\r\n\r\n\r\n\ufeffNotice\u200f period\x00 \t\t 30 days  "
    report = NormalizationReport()
    assert normalize_text(raw, report=report) == "Basic Salary: 12,000SAR\n\nNotice period\t30 days"
    assert report.removed == {"zero_width_space": 1, "soft_hyphen": 1, "byte_order_mark": 1, "bidi_mark": 1}


def test_arabic_legal_text_numbers_and_punctuation_are_preserved_exactly():
    assert normalize_text(ARABIC_LEGAL) == ARABIC_LEGAL  # tashkeel, tatweel, hamza, taa marbuta, ٢١ unchanged
    for text in ("Article 109.", "المادة (109)", "1.2.3", "البند 4-أ", "25%", "SAR 12,000.50"):
        assert normalize_text(text) == text


def test_arabic_presentation_forms_are_mapped_only_when_requested():
    presentation = "\ufe8d\ufedf\ufecc\ufee4\ufedf"  # glyph codes for "العمل" as some PDFs emit them
    assert normalize_text(presentation) == presentation
    report = NormalizationReport()
    assert normalize_text(presentation, map_arabic_presentation_forms=True, report=report) == "العمل"
    assert report.presentation_forms_mapped == 5 and report.warnings()


@pytest.mark.parametrize("text, language", [
    ("Employment contract for a data analyst", "en"), (ARABIC_LEGAL, "ar"),
    ("عقد عمل Employment Contract بين الطرفين between parties", "mixed"), ("12345 ---", "unknown"),
])
def test_language_detection(text, language):
    assert detect_language(text) == language


def test_error_messages_are_sanitized():
    message = sanitize_message("cannot open /Users/someone/uploads/private/contract.pdf or C:\\Users\\x\\a.docx\n  now")
    assert "/Users/someone" not in message and "C:\\Users" not in message and "\n" not in message


def test_parsed_document_status_invariants_prevent_silent_results():
    base = dict(document_id="doc_x", filename="a.pdf", file_type=FileType.PDF)
    with pytest.raises(ValidationError, match="requires extracted text"):
        ParsedDocument(status=DocumentStatus.SUCCESS, **base)
    with pytest.raises(ValidationError, match="must not carry extracted content"):
        ParsedDocument(status=DocumentStatus.OCR_REQUIRED, full_text="guessed text", **base)
    with pytest.raises(ValidationError, match="requires an ErrorInfo"):
        ParsedDocument(status=DocumentStatus.EXTRACTION_FAILED, **base)
    ok = ParsedDocument(status=DocumentStatus.EXTRACTION_FAILED, errors=[ErrorInfo(code="x", stage="y", message="z")], **base)
    assert ok.model_dump(mode="json")["status"] == "extraction_failed"


def test_document_ids_are_deterministic_and_results_serialise(sample_bytes):
    processor = DocumentProcessor(DocumentProcessingSettings())
    first = processor.parse_bytes(sample_bytes["sample_contract_en.docx"], "a.docx")
    second = processor.parse_bytes(sample_bytes["sample_contract_en.docx"], "renamed.docx")
    assert first.document_id == second.document_id
    assert ParsedDocument.model_validate_json(first.model_dump_json()) == first
    assert set(processor.supported_file_types) == {FileType.PDF, FileType.DOCX}


def test_settings_validate_limits_and_read_environment(tmp_path):
    with pytest.raises(ValueError):
        DocumentProcessingSettings(max_file_size_bytes=0)
    settings = DocumentProcessingSettings.from_env(
        {"SANAD_MAX_UPLOAD_MB": "2", "SANAD_MAX_PDF_PAGES": "7", "SANAD_UPLOAD_ROOT": str(tmp_path)})
    assert (settings.max_file_size_bytes, settings.max_pdf_pages) == (2 * 1024 * 1024, 7)
    assert settings.allowed_roots == (tmp_path.resolve(),)
