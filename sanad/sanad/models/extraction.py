"""Structured fields extracted from a ParsedDocument (Phase 3).

These models describe what a document *says*, with provenance. They contain no legal assessment,
no compatibility scoring and no salary benchmarking (Phase 4).

Field statuses:
  found          exactly one consistent value was read from the document
  not_found      the document was readable but did not state this field
  ambiguous      the document states conflicting values, or a value that cannot be read reliably;
                 value stays None and every candidate is listed
  not_extracted  the document text was not available (e.g. OCR required), so nothing could be checked
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field, model_validator

from sanad.models.common import ErrorInfo, ResultStatus
from sanad.models.documents import DocumentStatus

T = TypeVar("T")


class FieldStatus(str, Enum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
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
    source_text: str | None = Field(default=None, description="Verbatim text the value was read from.")
    page_number: int | None = None
    confidence: Literal["high", "medium"] | None = None
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


class ClauseValue(BaseModel):
    heading: str | None = None
    text: str


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
    employee_name: ExtractedField[str]
    employer_name: ExtractedField[str]
    job_title: ExtractedField[str]
    contract_type: ExtractedField[ContractTypeValue]
    start_date: ExtractedField[DateValue]
    end_date: ExtractedField[DateValue]
    contract_duration: ExtractedField[DurationValue]
    probation_period: ExtractedField[DurationValue]
    salary: ExtractedField[MoneyValue]
    total_salary: ExtractedField[MoneyValue]
    housing_allowance: ExtractedField[AllowanceValue]
    transportation_allowance: ExtractedField[AllowanceValue]
    other_allowances: ExtractedField[list[AllowanceValue]]
    working_hours: ExtractedField[WorkingHoursValue]
    working_days: ExtractedField[WorkingDaysValue]
    annual_leave: ExtractedField[DurationValue]
    notice_period: ExtractedField[DurationValue]
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
