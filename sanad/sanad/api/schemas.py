"""Edge validation and the API's own small payloads.

Nothing here analyses, parses or routes: it only checks what arrives over HTTP and turns it into the
existing SanadRequest. Business rules stay in the Orchestrator and the agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

from pydantic import BaseModel, Field, ValidationError

from sanad.config import ApiSettings, DocumentProcessingSettings
from sanad.models.analysis import AnalysisStatus, SalaryQuery, TargetJob
from sanad.models.orchestration import DocumentRole, SanadRequest, TaskHint, UploadedDocument

# Orchestrator status -> HTTP status. A structured result is always returned in the body.
HTTP_STATUS = {
    AnalysisStatus.SUCCESS: 200,
    AnalysisStatus.PARTIAL: 200,
    AnalysisStatus.INSUFFICIENT_EVIDENCE: 200,
    AnalysisStatus.INVALID_INPUT: 422,
    AnalysisStatus.RAG_ERROR: 502,
    AnalysisStatus.ANALYSIS_ERROR: 500,
}


class ApiErrorBody(BaseModel):
    code: str
    message: str
    detail: str | None = None


class ApiErrorResponse(BaseModel):
    error: ApiErrorBody


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class ApiConfig(BaseModel):
    """What the user interface needs to validate a request before sending it."""

    allowed_file_types: list[str]
    max_file_size_mb: float
    max_total_size_mb: float
    max_files: int
    max_contracts: int
    tasks: list[str]
    document_roles: list[str]
    comparison_priorities: list[str]
    answer_generation_enabled: bool
    salary_benchmarking_enabled: bool = False


class ApiHealth(BaseModel):
    status: str = "ok"
    service: str = "sanad-api"
    entry_point: str = "SanadOrchestrator.handle"


class ApiValidationError(Exception):
    """Rejected at the edge, before the Orchestrator is called."""

    def __init__(self, status_code: int, code: str, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.detail = detail


@dataclass(frozen=True)
class UploadLimits:
    api: ApiSettings
    documents: DocumentProcessingSettings

    @property
    def max_file_bytes(self) -> int:
        return self.documents.max_file_size_bytes


def safe_filename(name: str | None, position: int) -> str:
    """Base name only: an upload never decides where anything is written."""
    if not name:
        return f"upload_{position}"
    base = PureWindowsPath(PurePosixPath(name).name).name.strip()
    return base or f"upload_{position}"


def check_upload(filename: str, data: bytes, limits: UploadLimits) -> None:
    suffix = PurePosixPath(filename.lower()).suffix
    if suffix not in limits.api.allowed_suffixes:
        raise ApiValidationError(415, "unsupported_file_type",
                                 f"'{filename}' is not a supported file type. Sanad accepts "
                                 f"{', '.join(limits.api.allowed_suffixes)} files only.")
    if not data:
        raise ApiValidationError(400, "empty_file", f"'{filename}' is empty.")
    if len(data) > limits.max_file_bytes:
        raise ApiValidationError(413, "file_too_large",
                                 f"'{filename}' is larger than the {limits.max_file_bytes // (1024 * 1024)} MB limit.")


def build_salary_query(job_title: str | None, location: str | None, years_experience: str | None,
                       seniority: str | None) -> SalaryQuery | None:
    """Optional salary hints from the form; anything unparsable is rejected rather than ignored."""
    if not any(value and str(value).strip() for value in (job_title, location, years_experience, seniority)):
        return None
    years: float | None = None
    if years_experience and str(years_experience).strip():
        try:
            years = float(str(years_experience).strip())
        except ValueError:
            raise ApiValidationError(400, "invalid_request",
                                     "years_experience must be a number of years.") from None
    try:
        return SalaryQuery(job_title=(job_title or None), location=(location or None), years_experience=years,
                           seniority=(seniority or None))
    except ValidationError as exc:
        raise ApiValidationError(400, "invalid_request", "The salary details are not valid.",
                                 str(exc.errors()[0].get("msg", ""))) from None


def build_request(uploads: list[tuple[str, bytes]], *, roles: list[str], labels: list[str], question: str | None,
                  task: str, priorities: list[str], target_job: str | None, limits: UploadLimits,
                  salary_query: SalaryQuery | None = None) -> SanadRequest:
    """Validated HTTP input -> the existing SanadRequest. Raises ApiValidationError for bad input."""
    if len(uploads) > limits.api.max_files:
        raise ApiValidationError(413, "too_many_files",
                                 f"At most {limits.api.max_files} files can be uploaded in one request.")
    total = sum(len(data) for _, data in uploads)
    if total > limits.api.max_total_bytes:
        raise ApiValidationError(413, "upload_too_large",
                                 f"The upload is larger than the {limits.api.max_total_bytes // (1024 * 1024)} MB limit.")
    if roles and len(roles) != len(uploads):
        raise ApiValidationError(400, "invalid_request", "Give one role per uploaded file, or none at all.")
    if labels and len(labels) != len(uploads):
        raise ApiValidationError(400, "invalid_request", "Give one label per uploaded file, or none at all.")
    if not uploads and not (question or "").strip():
        raise ApiValidationError(400, "nothing_to_do", "Upload a contract or CV, or ask a regulatory question.")

    for filename, data in uploads:
        check_upload(filename, data, limits)

    documents = []
    for position, (filename, data) in enumerate(uploads):
        role = roles[position] if roles else DocumentRole.AUTO.value
        label = (labels[position].strip() if labels else "") or None
        try:
            documents.append(UploadedDocument(filename=filename, content=data, role=DocumentRole(role), label=label))
        except (ValueError, ValidationError) as exc:
            raise ApiValidationError(400, "invalid_request", f"'{role}' is not a valid document role.", str(exc)) from None

    try:
        job = TargetJob.model_validate_json(target_job) if target_job and target_job.strip() else None
    except ValidationError as exc:
        raise ApiValidationError(400, "invalid_request", "target_job is not a valid job description.",
                                 str(exc.errors()[0].get("msg", ""))) from None
    try:
        return SanadRequest(question=(question or None), documents=documents, task=TaskHint(task),
                            priorities=[p for p in priorities if p], target_job=job, salary_query=salary_query)
    except (ValueError, ValidationError) as exc:
        raise ApiValidationError(400, "invalid_request", "The request could not be built.",
                                 str(exc.errors()[0].get("msg", "")) if isinstance(exc, ValidationError) else str(exc)
                                 ) from None
