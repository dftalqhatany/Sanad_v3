"""Structured output of the Phase 4 Contract & CV Analysis Agent.

Three kinds of information are kept apart on purpose:
  * DocumentFact         what the document says (Phase 3 extraction, with provenance)
  * EvidenceReference    what the existing regulatory RAG returned (unchanged adapter evidence)
  * Interpretation       what the agent concluded, by which method, and whether it is grounded

Sanad is an AI assistant, not a legal authority: compliance labels are only allowed when an
interpretation is grounded in retrieved regulatory evidence (enforced by validators below).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, model_validator

from sanad.models.common import ErrorInfo, ResultStatus
from sanad.models.documents import DocumentStatus
from sanad.models.extraction import (
    CertificationEntry,
    EducationEntry,
    ExperienceEntry,
    FieldCandidate,
    FieldStatus,
    LanguageEntry,
    ListItem,
    MoneyValue,
    SourceSpan,
)
from sanad.models.regulatory import RegulatoryEvidence

DISCLAIMER = (
    "Sanad is an AI assistant, not a court or legal authority. Findings are based only on the extracted document "
    "and the regulatory evidence retrieved from the existing Saudi Labor Law knowledge base; they are not legal advice."
)


class AnalysisStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVALID_INPUT = "invalid_input"
    RAG_ERROR = "rag_error"
    ANALYSIS_ERROR = "analysis_error"


class FindingStatus(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    REQUIRES_REVIEW = "requires_review"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_APPLICABLE = "not_applicable"
    NOT_FOUND = "not_found"
    ERROR = "error"  # the regulatory check could not be performed (e.g. RAG unavailable)


class InterpretationMethod(str, Enum):
    NONE = "none"  # evidence retrieved, no automated interpretation configured
    LLM = "llm"


# --------------------------------------------------------------------------- facts
class DocumentFact(BaseModel):
    """A value exactly as Phase 3 extracted it (never re-parsed, never inferred)."""

    field: str
    extraction_status: FieldStatus
    value: Any = None
    raw_value: str | None = None
    source_text: str | None = None
    page_number: int | None = None
    section_id: str | None = None
    table_id: str | None = None
    sources: list[SourceSpan] = Field(default_factory=list)
    candidates: list[FieldCandidate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- regulatory evidence
class EvidenceRetrieval(BaseModel):
    topic: str
    query: str
    rank: int
    score: float
    score_is_rounded: bool


class EvidenceReference(BaseModel):
    """One article returned by RegulatoryRAGAdapter, kept unchanged, plus how it was retrieved."""

    evidence_id: str
    evidence: RegulatoryEvidence
    retrievals: list[EvidenceRetrieval] = Field(default_factory=list)
    relevant_topics: list[str] = Field(default_factory=list, description="Topics whose keywords the article text contains.")

    @computed_field
    @property
    def source(self) -> str:
        return f"{self.evidence.source.title_en} ({self.evidence.source.title_ar}), {self.evidence.source.publisher}"

    @computed_field
    @property
    def article_number(self) -> int | None:
        return self.evidence.reference.article_number

    @computed_field
    @property
    def article_name(self) -> str | None:
        return self.evidence.reference.article_name_ar

    @computed_field
    @property
    def part(self) -> str | None:
        return self.evidence.reference.part_title_ar

    @computed_field
    @property
    def chapter(self) -> str | None:
        return self.evidence.reference.chapter_title_ar

    @computed_field
    @property
    def arabic_text(self) -> str | None:
        return self.evidence.arabic_content

    @computed_field
    @property
    def english_text(self) -> str | None:
        return self.evidence.english_content

    @computed_field
    @property
    def citation(self) -> str:
        return self.evidence.citation

    @computed_field
    @property
    def rank(self) -> int:
        return min((r.rank for r in self.retrievals), default=self.evidence.rank)

    @computed_field
    @property
    def score(self) -> float:
        return max((r.score for r in self.retrievals), default=self.evidence.score)


class RegulatoryCheck(BaseModel):
    """The focused questions asked for one regulatory topic, and what came back."""

    topic: str
    triggered_by_fields: list[str]
    questions: list[str]
    status: Literal["evidence_found", "insufficient_evidence", "error", "not_run"]
    relevant_evidence_ids: list[str] = Field(default_factory=list)
    retrieved_evidence_ids: list[str] = Field(default_factory=list)
    errors: list[ErrorInfo] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- interpretation
class EvidenceQuote(BaseModel):
    evidence_id: str
    quote: str
    verified: bool = False
    language: Literal["ar", "en"] | None = None


class Interpretation(BaseModel):
    method: InterpretationMethod
    assessment: FindingStatus
    explanation: str
    cited_evidence_ids: list[str] = Field(default_factory=list)
    evidence_quotes: list[EvidenceQuote] = Field(default_factory=list)
    contract_quote: str | None = None
    grounded: bool = False
    model: str | None = None
    notes: list[str] = Field(default_factory=list)


class AnalysisFinding(BaseModel):
    finding_id: str
    field: str
    topic: str | None = None
    status: FindingStatus
    assessment: str
    explanation: str
    contract_fact: DocumentFact
    regulatory_evidence: list[EvidenceReference] = Field(default_factory=list)
    interpretation: Interpretation | None = None
    source_span: SourceSpan | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status_rules(self):
        extraction = self.contract_fact.extraction_status
        if (self.status is FindingStatus.NOT_FOUND) != (extraction is FieldStatus.NOT_FOUND):
            raise ValueError(f"{self.field}: 'not_found' is used exactly when the field was not found in the extraction")
        if self.status in (FindingStatus.COMPLIANT, FindingStatus.NON_COMPLIANT):
            if extraction is not FieldStatus.FOUND:
                raise ValueError(f"{self.field}: a compliance label requires a found contract value")
            if not self.regulatory_evidence:
                raise ValueError(f"{self.field}: a compliance label requires regulatory evidence")
            if self.interpretation is None or not self.interpretation.grounded:
                raise ValueError(f"{self.field}: a compliance label requires a grounded interpretation")
            if not any(q.verified for q in self.interpretation.evidence_quotes):
                raise ValueError(f"{self.field}: a compliance label requires a verified evidence quote")
        if self.status is FindingStatus.NOT_APPLICABLE and self.topic is not None:
            raise ValueError(f"{self.field}: 'not_applicable' is only used for fields without a regulatory topic")
        if self.status is FindingStatus.INSUFFICIENT_EVIDENCE and self.topic is None:
            raise ValueError(f"{self.field}: 'insufficient_evidence' requires a regulatory topic")
        return self


# --------------------------------------------------------------------------- salary benchmark (interface only)
SourceTier = Literal["official", "market", "supplementary", "lead_only"]
SalaryBasis = Literal["base_salary", "total_compensation", "unspecified"]


class SalarySource(BaseModel):
    name: str
    url: str
    source_type: Literal["official_statistics", "market_survey", "job_posting", "other"]
    retrieved_at: str | None = None
    tier: SourceTier | None = Field(default=None, description="Official > market > supplementary; 'lead_only' is never "
                                                              "used as salary evidence on its own.")


class SalaryObservation(BaseModel):
    """One salary figure read from one page, kept with the text it came from."""

    source_name: str
    url: str
    tier: SourceTier
    retrieved_at: str
    quote: str = Field(description="The text the numbers were read from, unchanged.")
    title: str | None = None
    minimum: float = Field(gt=0)
    maximum: float = Field(gt=0)
    currency: str
    period: Literal["monthly", "annual"]
    monthly_min: float = Field(gt=0, description="minimum normalised to SAR per month (annual / 12, rates below).")
    monthly_max: float = Field(gt=0)
    basis: SalaryBasis = "unspecified"
    converted_from: str | None = Field(default=None, description="Original currency when a configured rate was applied.")
    usable: bool = Field(default=True, description="False when it may not drive the range (lead-only source, or an "
                                                   "unconvertible currency).")
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ordered(self):
        if self.maximum < self.minimum or self.monthly_max < self.monthly_min:
            raise ValueError("a salary observation's maximum must not be below its minimum")
        return self


class SalaryQuery(BaseModel):
    """What the user (or the contract) says about the job being benchmarked."""

    job_title: str | None = None
    location: str | None = None
    years_experience: float | None = Field(default=None, ge=0, le=60)
    seniority: str | None = None
    currency: str = "SAR"


class SalaryBenchmark(BaseModel):
    status: Literal["not_configured", "success", "insufficient_data", "error"]
    provider: str
    message: str
    contract_salary: MoneyValue | None = None
    market_min: float | None = None
    market_max: float | None = None
    currency: str | None = None
    location: str | None = None
    job_title: str | None = None
    sources: list[SalarySource] = Field(default_factory=list)
    period: Literal["monthly", "annual"] | None = None
    basis: SalaryBasis | None = Field(default=None, description="Base salary and total compensation are never mixed.")
    data_quality: Literal["official", "market", "supplementary", "mixed", "conflicting", "insufficient"] | None = None
    limitations: list[str] = Field(default_factory=list)
    observations: list[SalaryObservation] = Field(default_factory=list)
    query: str | None = Field(default=None, description="The search query that was actually sent.")
    searched_urls: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_unsourced_market_data(self):
        has_range = self.market_min is not None or self.market_max is not None
        if has_range and not self.sources:
            raise ValueError("a market salary range must cite at least one source")
        if self.status == "success" and not has_range:
            raise ValueError("'success' requires a sourced market range")
        if has_range:
            if self.market_min is None or self.market_max is None:
                raise ValueError("a market range needs both a minimum and a maximum")
            if self.market_max < self.market_min:
                raise ValueError("the market maximum must not be below its minimum")
            cited = {source.url for source in self.sources}
            evidence = [o for o in self.observations if o.usable and o.tier != "lead_only" and o.url in cited]
            if not evidence:
                raise ValueError("a market range must come from usable salary observations of the cited sources")
            if self.market_min < min(o.monthly_min for o in evidence) or self.market_max > max(o.monthly_max for o in evidence):
                raise ValueError("the market range must stay within the observed figures")
            if self.period is None or self.basis is None:
                raise ValueError("a market range must state its period and whether it is base or total pay")
        return self


# --------------------------------------------------------------------------- results
class _AnalysisResult(BaseModel):
    status: AnalysisStatus
    document_id: str
    filename: str
    document_status: DocumentStatus
    extraction_status: ResultStatus
    overall_summary: str
    status_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[ErrorInfo] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER

    @model_validator(mode="after")
    def _errors_are_explicit(self):
        if self.status in (AnalysisStatus.INVALID_INPUT, AnalysisStatus.RAG_ERROR, AnalysisStatus.ANALYSIS_ERROR):
            if not self.errors:
                raise ValueError(f"status '{self.status.value}' requires at least one ErrorInfo")
        return self


class ContractAnalysisResult(_AnalysisResult):
    document_type: Literal["employment_contract"] = "employment_contract"
    findings: list[AnalysisFinding] = Field(default_factory=list)
    regulatory_checks: list[RegulatoryCheck] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    interpreter: str = "none"
    salary_benchmark: SalaryBenchmark | None = None

    def finding(self, field: str) -> AnalysisFinding:
        return next(f for f in self.findings if f.field == field)


# --------------------------------------------------------------------------- CV
class CvObservation(BaseModel):
    code: Literal["section_not_found", "ambiguous_field", "experience_without_dates", "experience_title_not_explicit"]
    field: str
    message: str
    sources: list[SourceSpan] = Field(default_factory=list)


class CvProfile(BaseModel):
    """CV content exactly as extracted in Phase 3 (items keep their own source spans)."""

    name: DocumentFact
    email: DocumentFact
    phone: DocumentFact
    location: DocumentFact
    summary: DocumentFact
    skills: list[ListItem] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    certifications: list[CertificationEntry] = Field(default_factory=list)
    work_experience: list[ExperienceEntry] = Field(default_factory=list)
    languages: list[LanguageEntry] = Field(default_factory=list)


class CvFieldReview(BaseModel):
    field: str
    extraction_status: FieldStatus
    item_count: int | None = None
    notes: list[str] = Field(default_factory=list)


class CompatibilityStatus(str, Enum):
    EXPLICIT_MATCH = "explicit_match"
    PARTIAL_MATCH = "partial_match"
    MISSING_REQUIREMENT = "missing_requirement"
    INSUFFICIENT_INFORMATION = "insufficient_information"


class TargetJob(BaseModel):
    """An explicit job to compare the CV with. It is never inferred from the CV."""

    model_config = {"extra": "forbid"}

    title: str = Field(min_length=1)
    source: Literal["user_provided", "contract_job_title"] = "user_provided"
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    required_education: list[str] = Field(default_factory=list)
    required_certifications: list[str] = Field(default_factory=list)
    required_languages: list[str] = Field(default_factory=list)
    minimum_years_experience: float | None = Field(default=None, ge=0)
    source_span: SourceSpan | None = None


class CvEvidence(BaseModel):
    cv_field: str
    text: str
    source: SourceSpan


class RequirementAssessment(BaseModel):
    requirement_type: Literal["job_title", "skill", "education", "certification", "language", "experience_years"]
    requirement: str
    importance: Literal["required", "preferred"]
    status: CompatibilityStatus
    explanation: str
    cv_evidence: list[CvEvidence] = Field(default_factory=list)
    searched_fields: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _matches_need_evidence(self):
        if self.status in (CompatibilityStatus.EXPLICIT_MATCH, CompatibilityStatus.PARTIAL_MATCH) and not self.cv_evidence:
            raise ValueError("a match must cite the CV evidence that supports it")
        return self


class JobCompatibilityResult(BaseModel):
    target_job: TargetJob
    requirements: list[RequirementAssessment]
    status_counts: dict[str, int]
    overall: Literal["required_requirements_explicitly_met", "gaps_identified", "partially_evidenced",
                     "insufficient_information"]
    summary: str
    note: str = ("This compares explicit CV statements with the stated requirements. It is not a hiring decision "
                 "and does not judge whether the person is qualified.")


class CvAnalysisResult(_AnalysisResult):
    document_type: Literal["cv"] = "cv"
    profile: CvProfile | None = None
    field_reviews: list[CvFieldReview] = Field(default_factory=list)
    observations: list[CvObservation] = Field(default_factory=list)
    job_compatibility: JobCompatibilityResult | None = None


class AnalysisBundle(BaseModel):
    status: AnalysisStatus
    contract_analysis: ContractAnalysisResult | None = None
    cv_analysis: CvAnalysisResult | None = None
