"""Document intake: uploaded file -> existing parser -> existing extraction -> typed item.

This is the only place in Phase 6 that touches the parser and extraction layers; the agents never do.
A document whose role was not declared is classified deterministically by counting how many contract
fields versus CV fields the existing extractors actually found - no model and no keyword guessing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config import DocumentProcessingSettings
from extraction import extract_contract, extract_cv
from models.common import ErrorInfo, ResultStatus
from models.documents import DocumentStatus, ParsedDocument
from models.extraction import ContractExtraction, CvExtraction, FieldStatus
from models.orchestration import DocumentIntakeSummary, DocumentRole, UploadedDocument
from orchestrator.errors import OrchestratorErrorCode, orchestrator_error
from parsers import DocumentProcessor

logger = logging.getLogger(f"sanad.{__name__}")  # one "sanad" logging namespace, as before the flattening

CONTRACT_MARKERS = ("employer_name", "job_title", "contract_type", "salary", "total_salary", "housing_allowance",
                    "probation_period", "working_hours", "working_days", "annual_leave", "notice_period",
                    "termination_terms", "contract_duration")
CV_MARKERS = ("skills", "education", "work_experience", "certifications", "languages", "summary")

UNREADABLE_MESSAGES = {
    DocumentStatus.UNSUPPORTED_FILE_TYPE: "Sanad accepts PDF and DOCX files only.",
    DocumentStatus.OCR_REQUIRED: "The pages carry no text layer (a scan); Sanad's MVP does not perform OCR.",
    DocumentStatus.EMPTY_DOCUMENT: "No text could be read from the file.",
    DocumentStatus.FILE_NOT_FOUND: "The file was not found.",
    DocumentStatus.FILE_TOO_LARGE: "The file is larger than the configured limit.",
    DocumentStatus.PATH_NOT_ALLOWED: "The file is outside the directory Sanad may read.",
    DocumentStatus.EXTRACTION_FAILED: "The file could not be read.",
}


@dataclass
class IntakeItem:
    summary: DocumentIntakeSummary
    parsed: ParsedDocument | None = None
    contract: ContractExtraction | None = None
    cv: CvExtraction | None = None
    errors: list[ErrorInfo] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.summary.usable


class DocumentIntake:
    """Parses and extracts uploaded documents using the existing Phase 3 components."""

    def __init__(self, processor: DocumentProcessor | None = None,
                 settings: DocumentProcessingSettings | None = None) -> None:
        self.processor = processor or DocumentProcessor(settings or DocumentProcessingSettings.from_env())

    def load(self, upload: UploadedDocument) -> IntakeItem:
        filename = upload.filename or (str(upload.path).rsplit("/", 1)[-1] if upload.path else "document")
        if upload.content is not None:
            parsed = self.processor.parse_bytes(upload.content, filename, upload.declared_mime_type)
        else:
            parsed = self.processor.parse_path(upload.path)

        summary = DocumentIntakeSummary(
            document_id=parsed.document_id, filename=parsed.filename, label=upload.label, file_type=parsed.file_type,
            document_status=parsed.status, role=None if upload.role is DocumentRole.AUTO else upload.role,
            role_source="declared" if upload.role is not DocumentRole.AUTO else "detected",
            page_count=parsed.page_count, warnings=list(parsed.extraction_warnings), errors=list(parsed.errors),
        )
        if not parsed.status.has_text:
            message = UNREADABLE_MESSAGES.get(parsed.status, "The file could not be read.")
            summary.notes.append(message)
            summary.role_source = "declared" if upload.role is not DocumentRole.AUTO else "undetermined"
            error = orchestrator_error(OrchestratorErrorCode.UNUSABLE_DOCUMENT, "intake",
                                       f"'{parsed.filename}' could not be used: {message}")
            return IntakeItem(summary=summary, parsed=parsed, errors=[error, *parsed.errors])

        item = self._extract(parsed, upload.role, summary)
        logger.info("Intake %s: status=%s role=%s (%s)", parsed.document_id, parsed.status.value,
                    summary.role.value if summary.role else "undetermined", summary.role_source)
        return item

    # ------------------------------------------------------------------ internals
    def _extract(self, parsed: ParsedDocument, declared: DocumentRole, summary: DocumentIntakeSummary) -> IntakeItem:
        if declared is DocumentRole.CONTRACT:
            contract = extract_contract(parsed)
            return self._item(summary, parsed, contract=contract, extraction=contract)
        if declared is DocumentRole.CV:
            cv = extract_cv(parsed)
            return self._item(summary, parsed, cv=cv, extraction=cv)

        contract, cv = extract_contract(parsed), extract_cv(parsed)
        contract_score, cv_score = _found(contract, CONTRACT_MARKERS), _found(cv, CV_MARKERS)
        summary.notes.append(f"Role detected from the extracted fields: {contract_score} contract field(s) versus "
                             f"{cv_score} CV section(s) were found.")
        if contract_score > cv_score:
            summary.role = DocumentRole.CONTRACT
            return self._item(summary, parsed, contract=contract, extraction=contract)
        if cv_score > contract_score:
            summary.role = DocumentRole.CV
            return self._item(summary, parsed, cv=cv, extraction=cv)
        summary.role, summary.role_source = None, "undetermined"
        summary.usable = False
        error = orchestrator_error(
            OrchestratorErrorCode.DOCUMENT_ROLE_UNDETERMINED, "intake",
            f"'{parsed.filename}' could not be identified as an employment contract or a CV "
            f"({contract_score} contract fields, {cv_score} CV sections). Upload it with an explicit role.")
        summary.errors.append(error)
        return IntakeItem(summary=summary, parsed=parsed, errors=[error])

    @staticmethod
    def _item(summary: DocumentIntakeSummary, parsed: ParsedDocument, *, extraction, contract=None, cv=None) -> IntakeItem:
        summary.extraction_status = extraction.status
        summary.warnings += [w for w in extraction.warnings if w not in summary.warnings]
        summary.usable = extraction.status is not ResultStatus.ERROR
        errors = []
        if not summary.usable:
            summary.notes.append("No fields could be extracted from the document text.")
            errors = [orchestrator_error(OrchestratorErrorCode.UNUSABLE_DOCUMENT, "intake",
                                         f"'{parsed.filename}' produced no usable fields."), *extraction.errors]
        return IntakeItem(summary=summary, parsed=parsed, contract=contract, cv=cv, errors=errors)


def _found(extraction, names: tuple[str, ...]) -> int:
    fields = extraction.fields()
    return sum(1 for name in names if name in fields and fields[name].status is FieldStatus.FOUND)
