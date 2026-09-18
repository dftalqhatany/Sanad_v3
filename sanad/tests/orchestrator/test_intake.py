"""Document intake: the only Phase 6 step that uses the existing parser and extraction layers."""

from __future__ import annotations

from sanad.models.common import ResultStatus
from sanad.models.orchestration import DocumentRole, UploadedDocument
from sanad.orchestrator import DocumentIntake
from tests.fixtures.documents.builders import docx_from_paragraphs
from tests.orchestrator.conftest import CONTRACT_LINES


def test_a_file_on_disk_is_parsed_and_extracted(sample_paths):
    item = DocumentIntake().load(UploadedDocument(path=sample_paths["sample_contract_en.docx"]))
    assert item.usable and item.contract is not None and item.cv is None
    assert item.summary.filename == "sample_contract_en.docx" and item.summary.document_id.startswith("doc_")
    assert item.contract.salary.value.amount == 12000
    assert item.summary.extraction_status is ResultStatus.SUCCESS


def test_a_declared_role_is_used_as_given():
    upload = UploadedDocument(filename="c.docx", content=docx_from_paragraphs(CONTRACT_LINES), role=DocumentRole.CV)
    item = DocumentIntake().load(upload)
    assert item.summary.role is DocumentRole.CV and item.summary.role_source == "declared"
    assert item.cv is not None and item.contract is None


def test_role_detection_counts_the_fields_that_were_actually_found(sample_bytes):
    intake = DocumentIntake()
    contract = intake.load(UploadedDocument(filename="a.docx", content=sample_bytes["sample_contract_en.docx"]))
    cv = intake.load(UploadedDocument(filename="b.docx", content=sample_bytes["sample_cv_en.docx"]))
    assert contract.summary.role is DocumentRole.CONTRACT and cv.summary.role is DocumentRole.CV
    assert all("Role detected from the extracted fields" in item.summary.notes[0] for item in (contract, cv))
    assert contract.contract is not None and cv.cv is not None


def test_a_document_with_no_recognisable_fields_is_undetermined():
    item = DocumentIntake().load(UploadedDocument(filename="x.docx", content=docx_from_paragraphs(["Hello there."])))
    assert not item.usable and item.summary.role is None and item.summary.role_source == "undetermined"
    assert item.errors[0].code == "document_role_undetermined"


def test_an_unreadable_document_carries_its_status_and_reason(sample_bytes):
    item = DocumentIntake().load(UploadedDocument(filename="s.pdf", content=sample_bytes["sample_scanned.pdf"]))
    assert not item.usable and item.contract is None and item.cv is None
    assert item.summary.document_status.value == "ocr_required"
    assert item.errors[0].code == "unusable_document" and "OCR" in item.errors[0].message


def test_partially_readable_documents_stay_usable_and_keep_their_warnings(sample_bytes):
    from tests.fixtures.documents.builders import pdf_with_text_pages_and_scanned_page

    item = DocumentIntake().load(UploadedDocument(filename="mixed.pdf", content=pdf_with_text_pages_and_scanned_page()))
    assert item.usable and item.summary.document_status.value == "partial"
    assert item.summary.extraction_status is ResultStatus.PARTIAL and item.summary.warnings
    assert item.summary.page_count and item.contract is not None
