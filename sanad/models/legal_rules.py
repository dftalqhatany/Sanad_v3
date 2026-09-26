"""Structured legal rules, contract facts and deterministic comparison results (Stage 3).

This module has no dependency on models.analysis: it describes the legal-domain shapes only
(a rule read from one article, a fact read from one clause, the arithmetic result of comparing
them). models.analysis imports from here to build the ClauseLegalFinding that ties a rule+fact+
comparison to a clause and its retrieved evidence - never the other way round.

Nothing here decides compliance. `LegalRule` is a reading of retrieved regulatory evidence; the
validators below only stop an invented number from being smuggled in as a "value" - they say
nothing about the contract.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RuleType(str, Enum):
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    EXACT = "exact"
    RANGE = "range"
    CONDITIONAL = "conditional"
    PROHIBITION = "prohibition"
    REQUIREMENT = "requirement"
    INFORMATIONAL = "informational"
    UNKNOWN = "unknown"


class Comparator(str, Enum):
    GE = ">="
    LE = "<="
    EQ = "=="
    GT = ">"
    LT = "<"
    RANGE = "range"
    CONDITIONAL = "conditional"
    PROHIBITED = "prohibited"
    REQUIRED = "required"
    UNKNOWN = "unknown"


class ConditionStatus(str, Enum):
    SATISFIED = "satisfied"
    NOT_SATISFIED = "not_satisfied"
    NOT_STATED = "not_stated"
    AMBIGUOUS = "ambiguous"
    NOT_APPLICABLE = "not_applicable"  # e.g. a unit this contract fact cannot be checked against


class RuleCondition(BaseModel):
    """One condition, exception or sub-threshold a rule depends on.

    `value`/`unit`/`comparator` are set only when the condition is itself a numeric sub-threshold
    (e.g. Article 75's separate 30-day and 60-day notice minimums). A condition that depends on a
    fact outside the numbers - "only if agreed in writing" - leaves them unset; the comparison
    engine can then only ever mark it satisfied when the contract's own words say so, or leave it
    'not_stated' - never 'non_compliant' merely because it is silent.
    """

    description: str
    status: ConditionStatus = ConditionStatus.NOT_STATED
    value: float | None = None
    unit: str | None = None
    comparator: Comparator | None = None
    notes: list[str] = Field(default_factory=list)


class LegalRule(BaseModel):
    """One deterministic reading of a retrieved regulatory article.

    Extraction is pattern-based, never an LLM guess: `extraction_method` is 'pattern' when a known,
    testable regex produced the value, and 'unknown' (rule_type=UNKNOWN, value=None) when the
    retrieved text did not match a pattern this module can safely convert into a number.
    """

    rule_id: str
    topic: str
    article_citation: str | None = None
    article_number: int | None = None
    rule_type: RuleType
    comparator: Comparator
    value: float | None = None
    value_max: float | None = None
    unit: str | None = None
    conditions: list[RuleCondition] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    source_evidence_id: str | None = None
    source_text: str | None = Field(default=None, description="The article text the rule was parsed from, verbatim.")
    confidence: Literal["high", "medium", "low"] = "medium"
    extraction_method: Literal["pattern", "unknown"] = "pattern"
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _never_invents_a_value(self):
        if self.rule_type in (RuleType.MINIMUM, RuleType.MAXIMUM, RuleType.EXACT):
            if self.value is None or self.unit is None:
                raise ValueError(f"{self.rule_type.value} requires a value and a unit read from the evidence")
        if self.rule_type is RuleType.RANGE:
            if self.value is None or self.value_max is None or self.unit is None:
                raise ValueError("range requires a minimum, a maximum and a unit read from the evidence")
            if self.value_max < self.value:
                raise ValueError("a range's maximum must not be below its minimum")
        if self.rule_type is RuleType.CONDITIONAL and not self.conditions:
            raise ValueError("a conditional rule must record at least one condition")
        if self.rule_type is RuleType.UNKNOWN and (self.value is not None or self.value_max is not None):
            raise ValueError("an unknown rule must not carry an invented numeric value")
        return self


class ContractFact(BaseModel):
    """A normalized fact read from one clause, kept with the provenance it came from.

    Two clauses stating the same number (e.g. two separate 90-day notice clauses) are always two
    separate ContractFacts, one per clause_id - they are never merged into one reading.
    """

    clause_id: str
    field: str | None = Field(default=None, description="The Stage 1 extracted field this fact was read from, if any.")
    raw_value: str
    normalized_value: float | None = Field(default=None, description="Numeric magnitude, when the value is one.")
    unit: str | None = None
    source_text: str
    page_number: int | None = None
    confidence: Literal["high", "medium", "low"] | None = None


class ComparisonResult(BaseModel):
    """The arithmetic result of comparing one ContractFact with one LegalRule. No verdict lives here:

    the finding-builder turns this into a FindingStatus; this record only carries what was compared,
    with what, and whether it held - so the 'why' can always be re-derived without re-running
    anything.
    """

    comparator: Comparator
    contract_value: float | None = None
    legal_value: float | None = None
    legal_value_max: float | None = None
    unit: str | None = None
    satisfied: bool | None = Field(default=None, description="True/False for a plain numeric check; None when no "
                                                              "single yes/no answer applies (e.g. a qualitative rule "
                                                              "or incompatible units).")
    condition_results: list[RuleCondition] = Field(default_factory=list)
    explanation: str
    notes: list[str] = Field(default_factory=list)
