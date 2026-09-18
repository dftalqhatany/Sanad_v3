"""Structured output of the Phase 5 Comparison & Recommendation Agent.

A comparison is built ONLY from the individual Analysis Agent results (one per contract). Every
compared value points back to the contract field, finding and source text it came from, and a
preferred contract is only named by a documented deterministic rule whose decision factors are listed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from sanad.models.analysis import AnalysisStatus, ContractAnalysisResult, CvAnalysisResult
from sanad.models.common import ErrorInfo
from sanad.models.documents import DocumentStatus

COMPARISON_DISCLAIMER = (
    "Sanad is an AI assistant, not a legal or financial adviser. The comparison uses only the terms extracted from "
    "each contract, each contract's individual analysis and the regulatory evidence retrieved for it. It is not "
    "legal advice or a hiring decision; please review the source clauses before deciding."
)

Category = Literal["salary", "compliance", "cv_compatibility", "benefits", "working_conditions", "contract_terms"]
Ranking = Literal["higher_is_better", "lower_is_better", "not_ranked"]
ValueStatus = Literal["available", "not_found", "ambiguous", "not_extracted", "not_comparable", "not_available",
                      "analysis_unavailable"]


class ComparedContract(BaseModel):
    """One contract and the individual analysis the Analysis Agent produced for it."""

    contract_id: str
    label: str
    document_id: str
    filename: str
    document_status: DocumentStatus
    analysis_status: AnalysisStatus
    contract_analysis: ContractAnalysisResult
    cv_analysis: CvAnalysisResult | None = None


class ContractValue(BaseModel):
    """What one contract says for one comparison dimension, with its provenance."""

    contract_id: str
    status: ValueStatus
    value: float | None = Field(default=None, description="Normalised number used for ranking (None if not ranked).")
    unit: str | None = None
    display: str
    fields_used: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)
    raw_values: list[str] = Field(default_factory=list)
    source_texts: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ranked_values_are_available(self):
        if self.value is not None and self.status != "available":
            raise ValueError("only an 'available' value can carry a number used for ranking")
        return self


class DimensionComparison(BaseModel):
    dimension: str
    label: str
    category: Category
    ranking: Ranking
    rationale: str = Field(description="Why this dimension is (or is not) ranked, and in which direction.")
    values: list[ContractValue]
    comparable: bool = Field(description="True when every contract has an available value in the same unit.")
    all_equal: bool = False
    best_contract_ids: list[str] = Field(default_factory=list)
    explanation: str

    @model_validator(mode="after")
    def _consistency(self):
        if self.best_contract_ids and (self.ranking == "not_ranked" or not self.comparable):
            raise ValueError(f"{self.dimension}: only a comparable ranked dimension can have leading contracts")
        known = {v.contract_id for v in self.values}
        if not set(self.best_contract_ids) <= known:
            raise ValueError(f"{self.dimension}: leading contracts must be among the compared contracts")
        return self


class ComparisonSections(BaseModel):
    salary: list[DimensionComparison] = Field(default_factory=list)
    compliance: list[DimensionComparison] = Field(default_factory=list)
    cv_compatibility: list[DimensionComparison] = Field(default_factory=list)
    benefits: list[DimensionComparison] = Field(default_factory=list)
    working_conditions: list[DimensionComparison] = Field(default_factory=list)
    contract_terms: list[DimensionComparison] = Field(default_factory=list)

    def all(self) -> list[DimensionComparison]:
        return [*self.salary, *self.compliance, *self.cv_compatibility, *self.benefits, *self.working_conditions,
                *self.contract_terms]


class ContractRisk(BaseModel):
    contract_id: str
    severity: Literal["high", "medium", "info"]
    code: Literal["non_compliant_finding", "regulatory_check_error", "insufficient_regulatory_evidence",
                  "requires_review", "ambiguous_term", "key_term_not_found", "analysis_failed", "cv_requirement_gap"]
    field: str | None = None
    finding_id: str | None = None
    message: str


class DecisionFactor(BaseModel):
    dimension: str
    label: str
    favours: list[str] = Field(description="Contract ids that lead on this dimension.")
    explanation: str
    evidence: list[ContractValue]


class Recommendation(BaseModel):
    status: Literal["preferred_contract", "no_clear_preference", "not_possible"]
    method: Literal["dominance", "priorities", "none"]
    preferred_contract_id: str | None = None
    explanation: str
    priorities_used: list[str] = Field(default_factory=list)
    decision_factors: list[DecisionFactor] = Field(default_factory=list)
    trade_offs: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    basis: str = ("Deterministic rules over the compared values; no language model chooses or ranks contracts. "
                  "Ranking directions are stated per dimension from the employee's point of view.")

    @model_validator(mode="after")
    def _explained_choice(self):
        if (self.status == "preferred_contract") != (self.preferred_contract_id is not None):
            raise ValueError("a preferred contract id is given exactly when status is 'preferred_contract'")
        if self.status == "preferred_contract":
            if not self.decision_factors:
                raise ValueError("a preferred contract must be explained by at least one decision factor")
            if not any(self.preferred_contract_id in factor.favours for factor in self.decision_factors):
                raise ValueError("the preferred contract must lead on at least one decision factor")
        return self


class ContractComparisonResult(BaseModel):
    status: AnalysisStatus
    contracts: list[ComparedContract] = Field(default_factory=list)
    comparison: ComparisonSections = Field(default_factory=ComparisonSections)
    risks: list[ContractRisk] = Field(default_factory=list)
    recommendation: Recommendation | None = None
    summary: str
    cv_document_id: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[ErrorInfo] = Field(default_factory=list)
    disclaimer: str = COMPARISON_DISCLAIMER

    @model_validator(mode="after")
    def _errors_are_explicit(self):
        if self.status in (AnalysisStatus.INVALID_INPUT, AnalysisStatus.RAG_ERROR, AnalysisStatus.ANALYSIS_ERROR):
            if not self.errors:
                raise ValueError(f"status '{self.status.value}' requires at least one ErrorInfo")
        return self

    def contract(self, contract_id: str) -> ComparedContract:
        return next(c for c in self.contracts if c.contract_id == contract_id)

    def dimension(self, name: str) -> DimensionComparison:
        return next(d for d in self.comparison.all() if d.dimension == name)
