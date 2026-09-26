"""Structured fields extracted from a ParsedDocument (Phase 3).

These models describe what a document *says*, with provenance. They contain no legal assessment,
no compatibility scoring and no salary benchmarking (Phase 4).

Field statuses:
  found           exactly one consistent value was read from the document
  not_found       the document was readable but did not state this field
  ambiguous       the document states conflicting values, or a value that cannot be read reliably;
                  value stays None and every candidate is listed
  not_applicable  the document explicitly states this topic does not apply (e.g. "not subject to a
                  probationary period") - a positive fact, not silence; value stays None but a source
                  (the statement itself) is always carried, unlike not_found/not_extracted
  not_extracted   the document text was not available (e.g. OCR required), so nothing could be checked
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field, model_validator

from models.common import ErrorInfo, ResultStatus
from models.documents import DocumentStatus

T = TypeVar("T")


class FieldStatus(str, Enum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    NOT_APPLICABLE = "not_applicable"
    NOT_EXTRACTED = "not_extracted"


class ExtractionMethodName(str, Enum):
    LABELED_LINE = "labeled_line"
    LABELED_NEXT_LINE = "labeled_next_line"
    TABLE_ROW = "table_row"
    CLAUSE_SENTENCE = "clause_sentence"
    SECTION_CONTENT = "section_content"
    DOCUMENT_TITLE = "document_title"
    PATTERN = "pattern"


class SourceSpan(BaseModel):
    """Where a value was read. `text` is copied verbatim from the parsed document."""

    text: str
    page_number: int | None = None
    section_id: str | None = None
    table_id: str | None = None
    method: ExtractionMethodName


class FieldCandidate(BaseModel):
    value: Any = None
    raw_value: str
    source: SourceSpan
    notes: list[str] = Field(default_factory=list)


class ExtractedField(BaseModel, Generic[T]):
    name: str
    status: FieldStatus
    value: T | None = None
    raw_value: str | None = Field(default=None, description="Value as written in the document.")
    normalized_value: str | None = Field(default=None, description="Canonical one-line form of `value` (e.g. '180 day', 'SAR 1800 monthly', '2025-07-24'). Derived from `value`; never a new fact.")
    source_text: str | None = Field(default=None, description="Verbatim text the value was read from.")
    page_number: int | None = None
    confidence: Literal["high", "medium", "low"] | None = None
    sources: list[SourceSpan] = Field(default_factory=list)
    candidates: list[FieldCandidate] = Field(default_factory=list, description="All readings, listed when ambiguous.")
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _invariants(self):
        if self.status is FieldStatus.FOUND:
            if self.value is None or not self.sources or self.raw_value is None:
                raise ValueError(f"{self.name}: 'found' requires a value, raw_value and at least one source")
        else:
            if self.value is not None:
                raise ValueError(f"{self.name}: only 'found' fields may carry a value")
        if self.status is FieldStatus.AMBIGUOUS and not self.candidates:
            raise ValueError(f"{self.name}: 'ambiguous' requires the conflicting candidates")
        if self.status in (FieldStatus.NOT_FOUND, FieldStatus.NOT_EXTRACTED) and (self.sources or self.candidates):
            raise ValueError(f"{self.name}: '{self.status.value}' must not carry sources")
        if self.status is FieldStatus.NOT_APPLICABLE and not self.sources:
            raise ValueError(f"{self.name}: 'not_applicable' requires the source stating it does not apply")
        return self


# --------------------------------------------------------------------------- typed values
class MoneyValue(BaseModel):
    amount: float
    currency: str | None = Field(default=None, description="Only when written (e.g. SAR for 'ريال').")
    period: Literal["monthly", "annual"] | None = None


class AllowanceValue(BaseModel):
    name: str | None = Field(default=None, description="Allowance label as written.")
    amount: float | None = None
    currency: str | None = None
    period: Literal["monthly", "annual"] | None = None
    percentage: float | None = Field(default=None, description="When written as a percentage, e.g. 25.")
    percentage_basis: str | None = Field(default=None, description="What the percentage applies to, as written.")


class DurationValue(BaseModel):
    count: float
    unit: Literal["day", "week", "month", "year"]
    qualifier: str | None = Field(default=None, description="e.g. 'working', 'calendar' when written.")


class DateValue(BaseModel):
    raw: str
    iso_date: str | None = Field(default=None, description="Only when the day/month order is unambiguous.")
    calendar: Literal["gregorian", "hijri"] = "gregorian"


class WorkingHoursValue(BaseModel):
    hours_per_day: float | None = None
    hours_per_week: float | None = None
    hours: float | None = Field(default=None, description="Hours stated without a period.")


class WorkingDaysValue(BaseModel):
    days_per_week: int | None = None
    days: list[str] = Field(default_factory=list, description="Named days (English), when written as names or a range.")


class ContractTypeValue(BaseModel):
    raw: str
    normalized: Literal["fixed_term", "indefinite"] | None = None


class ClauseType(str, Enum):
    """Controlled vocabulary for what a contract clause is about.

    OTHER is a clause that states a rule Sanad has no category for; UNKNOWN is one whose subject
    could not be established. Neither is ever guessed into a more specific type.
    """

    EMPLOYMENT = "employment"
    JOB_TITLE = "job_title"
    WORK_LOCATION = "work_location"
    START_DATE = "start_date"
    CONTRACT_DURATION = "contract_duration"
    PROBATION = "probation"
    NOTICE = "notice"
    TERMINATION = "termination"
    WORKING_HOURS = "working_hours"
    WEEKLY_REST = "weekly_rest"
    WORKING_DAYS = "working_days"
    ANNUAL_LEAVE = "annual_leave"
    SICK_LEAVE = "sick_leave"
    SALARY = "salary"
    ALLOWANCES = "allowances"
    BENEFITS = "benefits"
    OVERTIME = "overtime"
    TRANSPORTATION = "transportation"
    HOUSING = "housing"
    HEALTH_INSURANCE = "health_insurance"
    DUTIES = "duties"
    CONFIDENTIALITY = "confidentiality"
    NON_COMPETE = "non_compete"
    DISCIPLINARY = "disciplinary"
    OTHER = "other"
    UNKNOWN = "unknown"


class ClauseValue(BaseModel):
    """One contractual rule, quoted from the document.

    `text` is always copied from the parsed document - never paraphrased, summarised or rewritten -
    so a clause can always be shown to the reader as the contract's own words.
    """

    clause_id: str | None = Field(default=None, description="Stable id within one segmentation, e.g. 'C07'.")
    clause_name: str | None = Field(default=None, description="Short human label, e.g. 'Notice during probation'.")
    clause_type: ClauseType = ClauseType.UNKNOWN
    heading: str | None = None
    text: str
    source_span: SourceSpan | None = None
    confidence: Literal["high", "medium", "low"] | None = None

    @property
    def page_number(self) -> int | None:
        return self.source_span.page_number if self.source_span else None

    @property
    def section_id(self) -> str | None:
        return self.source_span.section_id if self.source_span else None

    @property
    def extraction_method(self) -> ExtractionMethodName | None:
        return self.source_span.method if self.source_span else None


class ExperienceEntry(BaseModel):
    header: str = Field(description="The entry's first line as written.")
    title: str | None = Field(default=None, description="Only from an explicit 'X at Y' / 'X في Y' form.")
    organization: str | None = None
    start: str | None = Field(default=None, description="As written.")
    end: str | None = Field(default=None, description="As written.")
    is_current: bool | None = Field(default=None, description="True only when 'Present' / 'حتى الآن' is written.")
    details: list[str] = Field(default_factory=list)
    source: SourceSpan


class EducationEntry(BaseModel):
    text: str
    degree: str | None = None
    institution: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    year: int | None = Field(default=None, description="A single year written without a range.")
    source: SourceSpan


class CertificationEntry(BaseModel):
    name: str
    year: int | None = None
    source: SourceSpan


class LanguageEntry(BaseModel):
    language: str
    level: str | None = None
    source: SourceSpan


class ListItem(BaseModel):
    text: str
    source: SourceSpan


# --------------------------------------------------------------------------- results
class _ExtractionResult(BaseModel):
    document_id: str
    filename: str
    document_status: DocumentStatus
    status: ResultStatus
    warnings: list[str] = Field(default_factory=list)
    errors: list[ErrorInfo] = Field(default_factory=list)

    def fields(self) -> dict[str, ExtractedField]:
        return {name: value for name, value in self if isinstance(value, ExtractedField)}

    def fields_with_status(self, status: FieldStatus) -> list[str]:
        return [name for name, field in self.fields().items() if field.status is status]


class ContractExtraction(_ExtractionResult):
    document_type: Literal["employment_contract"] = "employment_contract"
    clauses: list[ClauseValue] = Field(
        default_factory=list,
        description="Segmented, classified clauses in document order. Not an ExtractedField: clauses are "
                    "spans of the document, not one value per field, so fields() does not report them.")
    employee_name: ExtractedField[str]
    employer_name: ExtractedField[str]
    job_title: ExtractedField[str]
    contract_type: ExtractedField[ContractTypeValue]
    start_date: ExtractedField[DateValue]
    end_date: ExtractedField[DateValue]
    contract_duration: ExtractedField[DurationValue]
    probation_period: ExtractedField[DurationValue]
    probation_status: ExtractedField[str] = Field(
        default_factory=lambda: ExtractedField(name="probation_status", status=FieldStatus.NOT_FOUND),
        description="Whether a probation period applies, derived from `probation_period` (FOUND with value "
                    "'applicable' when a period is stated, NOT_APPLICABLE when the document explicitly waives "
                    "it, mirroring `probation_period`'s own status otherwise). `probation_period` keeps "
                    "carrying the duration itself; this field never replaces it.")
    salary: ExtractedField[MoneyValue]
    total_salary: ExtractedField[MoneyValue]
    net_salary: ExtractedField[MoneyValue] = Field(
        default_factory=lambda: ExtractedField(name="net_salary", status=FieldStatus.NOT_FOUND),
        description="Take-home wage after statutory deductions, when the document states one as such "
                    "(distinct from `salary` (basic) and `total_salary` (gross)).")
    housing_allowance: ExtractedField[AllowanceValue]
    transportation_allowance: ExtractedField[AllowanceValue]
    other_allowances: ExtractedField[list[AllowanceValue]]
    working_hours: ExtractedField[WorkingHoursValue]
    working_days: ExtractedField[WorkingDaysValue]
    annual_leave: ExtractedField[DurationValue]
    notice_period: ExtractedField[DurationValue]
    notice_period_during_probation: ExtractedField[DurationValue] = Field(
        default_factory=lambda: ExtractedField(name="notice_period_during_probation", status=FieldStatus.NOT_FOUND),
        description="Notice while the worker is still on probation. Never merged with the notice period that "
                    "applies after confirmation, nor with the probation period itself.")
    notice_period_after_confirmation: ExtractedField[DurationValue] = Field(
        default_factory=lambda: ExtractedField(name="notice_period_after_confirmation", status=FieldStatus.NOT_FOUND),
        description="Notice once the worker is confirmed.")
    weekly_rest: ExtractedField[DurationValue] = Field(
        default_factory=lambda: ExtractedField(name="weekly_rest", status=FieldStatus.NOT_FOUND),
        description="Weekly rest / days off, as stated for the worker.")
    in_hand_salary: ExtractedField[MoneyValue] = Field(
        default_factory=lambda: ExtractedField(name="in_hand_salary", status=FieldStatus.NOT_FOUND),
        description="Take-home pay when the document states one separately from basic/gross.")
    benefits: ExtractedField[list[ListItem]] = Field(
        default_factory=lambda: ExtractedField(name="benefits", status=FieldStatus.NOT_FOUND),
        description="Non-wage benefits and perks listed in the document, as written.")
    termination_terms: ExtractedField[list[ClauseValue]]
    work_location: ExtractedField[str]
    nationality: ExtractedField[str]
    employee_id: ExtractedField[str]


class CvExtraction(_ExtractionResult):
    document_type: Literal["cv"] = "cv"
    name: ExtractedField[str]
    email: ExtractedField[str]
    phone: ExtractedField[str]
    location: ExtractedField[str]
    summary: ExtractedField[str]
    skills: ExtractedField[list[ListItem]]
    education: ExtractedField[list[EducationEntry]]
    certifications: ExtractedField[list[CertificationEntry]]
    work_experience: ExtractedField[list[ExperienceEntry]]
    languages: ExtractedField[list[LanguageEntry]]
