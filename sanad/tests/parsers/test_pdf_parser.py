"""PDF: text layer -> structured sections/tables; no text layer -> OCR_REQUIRED; broken -> explicit error."""

from __future__ import annotations

import json

import pytest

from config import DocumentProcessingSettings
from models.documents import DocumentStatus, ExtractionMethod, FileType, PageStatus, SectionType
from parsers import DocumentProcessor
from tests.fixtures.documents.builders import (
    contract_text_pdf,
    load_pymupdf,
    pdf_with_text_pages_and_scanned_page,
)
from tests.fixtures.documents.synthetic_content import CONTRACT_EN

pymupdf = load_pymupdf()


def test_text_pdf_is_extracted_into_structured_sections_with_page_provenance(parsed):
    document = parsed["sample_contract_text.pdf"]
    assert document.status is DocumentStatus.SUCCESS
    assert (document.file_type, document.mime_type, document.language) == (FileType.PDF, "application/pdf", "en")
    assert document.document_id.startswith("doc_") and len(document.sha256) == 64
    assert document.page_count == 2 and [p.status for p in document.pages] == [PageStatus.TEXT, PageStatus.TEXT]
    headings = {s.text: s for s in document.sections if s.section_type is SectionType.HEADING}
    assert {"Employment Contract", "Compensation", "Probation Period", "Termination"} <= set(headings)
    assert headings["Probation Period"].page_number == 1 and headings["Termination"].page_number == 2
    employee = next(s for s in document.sections if s.text == "Employee Name: Jordan Sample")
    assert employee.page_number == 1 and employee.title == "Employment Contract"
    assert all(s.extraction_method is ExtractionMethod.PDF_TEXT_LAYER for s in document.sections)
    assert "Either party may terminate this contract by giving sixty (60) days written notice." in document.full_text
    assert not document.errors and not document.ocr.required


def test_text_pdf_table_is_extracted_as_rows_without_duplicate_text(parsed):
    document = parsed["sample_contract_text.pdf"]
    [table] = document.tables
    assert table.rows == [list(row) for row in CONTRACT_EN["compensation_rows"]]
    assert table.page_number == 1 and table.title == "Compensation"
    assert not any("Housing Allowance" in s.text for s in document.sections)
    assert "Basic Salary | 12,000 SAR per month" in document.full_text


def test_blank_pdf_without_text_layer_returns_ocr_required(parsed):
    document = parsed["sample_empty.pdf"]
    assert document.status is DocumentStatus.OCR_REQUIRED
    assert document.sections == [] and document.tables == [] and document.full_text == ""
    assert document.pages[0].appears_blank is True and document.pages[0].has_images is False
    assert document.ocr.required is True and document.ocr.pages == [1]
    assert any("appear blank" in w for w in document.extraction_warnings)


def test_scanned_pdf_returns_ocr_required_and_no_text(parsed):
    document = parsed["sample_scanned.pdf"]
    assert document.status is DocumentStatus.OCR_REQUIRED
    [page] = document.pages
    assert (page.status, page.has_images, page.appears_blank) == (PageStatus.OCR_REQUIRED, True, False)
    assert document.full_text == "" and document.sections == []
    assert document.ocr.required is True and document.ocr.pages == [1]
    assert any("OCR is not part of Sanad's MVP" in w for w in document.extraction_warnings)


def test_pdf_with_a_scanned_page_is_partial_and_names_the_page(processor):
    document = processor.parse_bytes(pdf_with_text_pages_and_scanned_page(), "mixed.pdf")
    assert document.status is DocumentStatus.PARTIAL
    assert [p.status for p in document.pages] == [PageStatus.TEXT, PageStatus.TEXT, PageStatus.OCR_REQUIRED]
    assert any("Page(s) 3" in w and "need OCR" in w for w in document.extraction_warnings)
    assert "Jordan Sample" in document.full_text
    assert document.ocr.required is True and document.ocr.pages == [3]


@pytest.mark.parametrize("payload", [
    b"%PDF-1.7\n1 0 obj <<>> this is not a pdf body\n%%EOF",
    b"%PDF-" + b"\x00" * 200,
])
def test_malformed_pdf_returns_structured_extraction_error(processor, payload):
    document = processor.parse_bytes(payload, "broken.pdf")
    assert document.status is DocumentStatus.EXTRACTION_FAILED
    [error] = document.errors
    assert error.code == "malformed_pdf" and error.stage == "parsing"
    assert "broken.pdf" in error.message
    dumped = document.model_dump_json()
    assert "Traceback" not in dumped and error.exception_chain == []
    assert document.full_text == "" and document.sections == []


def test_truncated_pdf_is_reported_as_repaired_partial_content(processor):
    data = contract_text_pdf()
    document = processor.parse_bytes(data[: len(data) // 2], "truncated.pdf")
    assert document.status in (DocumentStatus.PARTIAL, DocumentStatus.EXTRACTION_FAILED)
    if document.status is DocumentStatus.PARTIAL:
        assert any("repaired" in w for w in document.extraction_warnings)


def test_password_protected_pdf_is_explicit(processor):
    source = pymupdf.open(stream=contract_text_pdf(), filetype="pdf")
    encrypted = source.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="user-secret", owner_pw="owner-secret")
    document = processor.parse_bytes(encrypted, "locked.pdf")
    assert document.status is DocumentStatus.EXTRACTION_FAILED
    assert [e.code for e in document.errors] == ["password_protected"]
    assert "secret" not in json.dumps(document.model_dump(mode="json"))


def test_page_limit_is_explicit_partial():
    processor = DocumentProcessor(DocumentProcessingSettings(max_pdf_pages=1))
    document = processor.parse_bytes(contract_text_pdf(), "contract.pdf")
    assert document.status is DocumentStatus.PARTIAL
    assert document.page_count == 2 and [p.page_number for p in document.pages] == [1]
    assert any("first 1 of 2 pages" in w for w in document.extraction_warnings)
    assert "Termination" not in document.full_text
