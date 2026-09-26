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

from models.common import ErrorInfo, ResultStatus
from models.documents import DocumentStatus
from models.extraction import (
    CertificationEntry,
    ClauseType,
    EducationEntry,
    ExperienceEntry,
    FieldCandidate,
    FieldStatus,
    LanguageEntry,
    ListItem,
    MoneyValue,
    SourceSpan,
)
from models.legal_rules import ComparisonResult, ContractFact, LegalRule, RuleType
from models.regulatory import RegulatoryEvidence

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


# A failure is never reported as an empty result. Every Sanad result model applies this one rule.
FAILURE_STATUSES = (AnalysisStatus.INVALID_INPUT, AnalysisStatus.RAG_ERROR, AnalysisStatus.ANALYSIS_ERROR)


def require_explicit_errors(status: AnalysisStatus, errors: list[ErrorInfo]) -> None:
    if status in FAILURE_STATUSES and not errors:
        raise ValueError(f"status '{status.value}' requires at least one ErrorInfo")


class FindingStatus(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    AMBIGUOUS = "ambiguous"  # the CONTRACT itself is conflicting or unclear (Stage 3)
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
class ClauseCheck(BaseModel):
    """One segmented clause, the regulatory question asked about it, and what the existing RAG returned.

    This is the Stage 2 output: it records what the contract says and which articles are relevant to
    it. It deliberately carries no verdict - no compliant/non-compliant field exists here - because
    deciding that needs rule semantics (minimum / maximum / conditional) that Stage 2 does not have.
    Stage 3's ClauseLegalFinding (below) is what turns this into a verdict.

    The chain clause -> query -> evidence is kept whole so a later stage can cite it without
    re-running retrieval.
    """

    clause_id: str
    clause_type: ClauseType
    clause_name: str | None = None
    clause_text: str = Field(description="The contract's own words, copied from the document.")
    source_span: SourceSpan | None = None
    regulatory_topic: str | None = Field(
        default=None, description="Name of the existing RegulatoryTopic, or None when no existing topic fits.")
    queries: list[str] = Field(default_factory=list, description="Questions sent to the existing RAG for this clause.")
    retrieval_queries: list[str] = Field(
        default_factory=list, description="The composed text the retriever actually saw (question + clause).")
    check: RegulatoryCheck | None = Field(
        default=None, description="The existing RegulatoryCheck record for this clause's retrieval.")
    evidence: list[EvidenceReference] = Field(default_factory=list)

    @property
    def page_number(self) -> int | None:
        return self.source_span.page_number if self.source_span else None


class ClauseLegalFinding(BaseModel):
    """Stage 3: one segmented clause, compared deterministically against a structured legal rule.

    `status` is always the output of agents.legal_rules.compare_fact_to_rule() plus the small set of
    deterministic rules in agents.legal_rules._status_from_comparison() - never an LLM's opinion. An
    optional LLM may later be plugged in to word `explanation`, but nothing is allowed to touch
    `status`, `comparison` or `legal_rule` after they are computed (see agents/legal_rules.py).
    """

    finding_id: str
    clause_id: str
    clause_type: ClauseType
    clause_name: str | None = None
    clause_text: str = Field(description="The contract's own words, copied from the document.")
    field: str | None = Field(default=None, description="The Stage 1 extracted field this clause's fact came from.")
    topic: str | None = Field(default=None, description="Name of the existing RegulatoryTopic checked, if any.")
    status: FindingStatus
    assessment: str
    explanation: str
    contract_fact: ContractFact | None = None
    legal_rule: LegalRule | None = None
    comparison: ComparisonResult | None = None
    regulatory_evidence: list[EvidenceReference] = Field(default_factory=list)
    source_span: SourceSpan | None = None
    confidence: Literal["high", "medium", "low"] | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def page_number(self) -> int | None:
        return self.source_span.page_number if self.source_span else None

    @model_validator(mode="after")
    def _status_rules(self):
        if self.status in (FindingStatus.COMPLIANT, FindingStatus.NON_COMPLIANT):
            if self.legal_rule is None or self.legal_rule.rule_type is RuleType.UNKNOWN:
                raise ValueError(f"{self.clause_id}: a compliance label requires a deterministic legal rule")
            if self.contract_fact is None or self.contract_fact.normalized_value is None:
                raise ValueError(f"{self.clause_id}: a compliance label requires a numeric contract fact")
            if self.comparison is None:
                raise ValueError(f"{self.clause_id}: a compliance label requires a recorded comparison")
        return self


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
    """Stage 4 field-level finding: informational only.

    `status` here is NEVER allowed to be COMPLIANT/NON_COMPLIANT (enforced by the validator below,
    unconditionally) - those two statuses are reserved for the deterministic clause-level
    ClauseLegalFinding (see models.analysis.ClauseLegalFinding and agents.legal_rules). An LLM may
    still populate `interpretation` with its own read of the evidence for a human reviewer, but that
    reading is never promoted to this finding's `status`.
    """

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
            # Stage 4 hardening: a compliance label is never valid on a field-level AnalysisFinding,
            # regardless of grounding or evidence. Compliant/non_compliant may only be produced by the
            # deterministic clause-level comparison (agents.legal_rules.build_clause_legal_finding),
            # recorded separately on ClauseLegalFinding / ContractAnalysisResult.clause_findings. An
            # LLM interpretation of a field is informational only (see agents/contract_analysis.py).
            raise ValueError(
                f"{self.field}: a compliance label is not allowed on a field-level finding; "
                "compliant/non_compliant must come only from the deterministic clause-level "
                "ClauseLegalFinding, never from field-level interpretation."
            )
        if self.status is FindingStatus.NOT_APPLICABLE:
            # Two distinct reasons are both legitimate: (a) structural - this field has no regulatory
            # topic at all, so Sanad never checks it; or (b) the CONTRACT ITSELF explicitly says the
            # field does not apply (e.g. "not subject to a probationary period"), regardless of whether
            # the field normally has a topic. Anything else claiming not_applicable is rejected.
            structural = self.topic is None
            stated_by_the_contract = extraction is FieldStatus.NOT_APPLICABLE
            if not (structural or stated_by_the_contract):
                raise ValueError(
                    f"{self.field}: 'not_applicable' requires either no regulatory topic for this field, "
                    "or the contract's own extraction already marking the field not applicable"
                )
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
        require_explicit_errors(self.status, self.errors)
        return self


class ContractAnalysisResult(_AnalysisResult):
    document_type: Literal["employment_contract"] = "employment_contract"
    findings: list[AnalysisFinding] = Field(default_factory=list)
    regulatory_checks: list[RegulatoryCheck] = Field(default_factory=list)
    clause_checks: list[ClauseCheck] = Field(
        default_factory=list,
        description="Segmented clauses with the regulatory evidence retrieved for each. Stage 2 output: "
                    "evidence only, never a compliance conclusion.")
    clause_findings: list[ClauseLegalFinding] = Field(
        default_factory=list,
        description="Stage 3 output: each eligible clause compared, deterministically, against a structured "
                    "legal rule derived from its retrieved evidence. One entry per segmented clause, in the "
                    "same order as clause_checks.")
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
