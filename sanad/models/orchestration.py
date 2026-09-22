"""Typed request and response of the Phase 6 Orchestrator.

The Orchestrator is the single entry point: it turns uploaded PDF/DOCX files and an optional question
into a routing decision and delegates to the existing components. It carries their structured results
unchanged (analysis bundle, comparison result, regulatory answer, salary benchmark) so no evidence,
source text, page number, finding id, risk or caveat is lost on the way out.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from models.analysis import (
    AnalysisBundle,
    AnalysisStatus,
    SalaryBenchmark,
    SalaryQuery,
    TargetJob,
    require_explicit_errors,
)
from models.common import ErrorInfo, ResultStatus
from models.comparison import ContractComparisonResult
from models.documents import DocumentStatus, FileType
from models.regulatory import RegulatoryAnswerResult

ORCHESTRATOR_DISCLAIMER = (
    "Sanad is an AI assistant, not a legal authority. Every conclusion comes from the uploaded documents and the "
    "evidence retrieved from the existing Saudi Labor Law knowledge base, and is not legal advice."
)


class DocumentRole(str, Enum):
    CONTRACT = "contract"
    CV = "cv"
    AUTO = "auto"  # decided by the intake step from the extracted fields


class TaskHint(str, Enum):
    AUTO = "auto"  # routed from what was uploaded
    REGULATORY_QUESTION = "regulatory_question"
    CONTRACT_ANALYSIS = "contract_analysis"
    CV_ANALYSIS = "cv_analysis"
    CONTRACT_COMPARISON = "contract_comparison"
    SALARY_BENCHMARK = "salary_benchmark"


class Route(str, Enum):
    REGULATORY_QUESTION = "regulatory_question"
    CONTRACT_ANALYSIS = "contract_analysis"
    CV_ANALYSIS = "cv_analysis"
    CONTRACT_COMPARISON = "contract_comparison"
    SALARY_BENCHMARK = "salary_benchmark"
    NONE = "none"  # nothing could be routed; the errors say why


class UploadedDocument(BaseModel):
    """One uploaded PDF or DOCX, given either as bytes or as a path the parser is allowed to read."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    filename: str | None = None
    content: bytes | None = Field(default=None, repr=False, exclude=True)
    path: str | Path | None = None
    role: DocumentRole = DocumentRole.AUTO
    label: str | None = None
    declared_mime_type: str | None = None

    @model_validator(mode="after")
    def _one_source(self):
        if (self.content is None) == (self.path is None):
            raise ValueError("give either content or path for an uploaded document")
        if self.content is not None and not self.filename:
            raise ValueError("a document given as bytes needs a filename")
        return self


class SanadRequest(BaseModel):
    """What the user asked for: an optional question, the uploaded documents and optional hints."""

    question: str | None = None
    documents: list[UploadedDocument] = Field(default_factory=list)
    task: TaskHint = TaskHint.AUTO
    target_job: TargetJob | None = None
    priorities: list[str] = Field(default_factory=list)
    salary_query: SalaryQuery | None = Field(default=None, description="Job title, location, years of experience or "
                                                                       "seniority for salary benchmarking.")

    @model_validator(mode="after")
    def _blank_question(self):
        if self.question is not None and not self.question.strip():
            raise ValueError("question must not be blank")
        return self


class DocumentIntakeSummary(BaseModel):
    """What the parser and extraction layer made of one uploaded file (no document content is copied here)."""

    document_id: str | None = None
    filename: str
    label: str | None = None
    file_type: FileType | None = None
    document_status: DocumentStatus
    role: DocumentRole | None = None
    role_source: Literal["declared", "detected", "undetermined"] = "declared"
    extraction_status: ResultStatus | None = None
    page_count: int | None = None
    usable: bool = False
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[ErrorInfo] = Field(default_factory=list)


class RoutingDecision(BaseModel):
    route: Route
    agent: str = Field(description="Component the request was delegated to, e.g. 'AnalysisAgent'.")
    rule: str = Field(description="The deterministic rule that selected the route (no model decides this).")
    reason: str
    task_hint: TaskHint = TaskHint.AUTO
    contract_count: int = 0
    cv_count: int = 0
    has_question: bool = False
    fallback_from: Route | None = Field(default=None, description="Set when unusable uploads changed the route.")


class OrchestratorResult(BaseModel):
    status: AnalysisStatus
    routing: RoutingDecision
    documents: list[DocumentIntakeSummary] = Field(default_factory=list)
    analysis: AnalysisBundle | None = None
    comparison: ContractComparisonResult | None = None
    regulatory_answer: RegulatoryAnswerResult | None = None
    salary_benchmark: SalaryBenchmark | None = None
    salary_contract_id: str | None = None
    summary: str
    warnings: list[str] = Field(default_factory=list)
    errors: list[ErrorInfo] = Field(default_factory=list)
    disclaimer: str = ORCHESTRATOR_DISCLAIMER

    @model_validator(mode="after")
    def _errors_are_explicit(self):
        require_explicit_errors(self.status, self.errors)
        if self.routing.route is Route.NONE and self.status is AnalysisStatus.SUCCESS:
            raise ValueError("an unrouted request cannot be a success")
        return self
