"""Typed schema for the Sanad Golden Dataset.

Evaluation only: nothing here is imported by `sanad/` or `frontend/`, and nothing here changes
product behaviour. The schema exists so that an invalid evaluation case fails immediately, at load
time, instead of quietly producing a meaningless metric later.

Three rules are enforced structurally rather than by convention:

  1. A case may only call itself 'reviewed' when it carries review metadata naming a person.
  2. A regulatory case must carry BOTH the knowledge-base index and the legal article number.
     They are not interchangeable: in the source dataset they differ in 95.6% of rows.
  3. A salary case may not declare a single "correct salary". Salary is evaluated on methodology
     (sources, tiers, units, period, basis, conflict handling), never against an invented figure.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CASE_ID = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")

Task = Literal["regulatory_question", "contract_analysis", "contract_comparison",
               "salary_benchmark", "robustness"]

ReviewStatus = Literal[
    "source_derived",          # taken from an existing dataset; label inherited, not re-verified
    "generated_from_fixture",  # derived mechanically from a repository fixture whose content is known
    "needs_review",            # a human must confirm the expectation before it counts
    "reviewed",                # a named person verified it against the primary source
    "quarantined",             # kept for provenance, excluded from every metric
]

Language = Literal["ar", "en", "mixed", "not_applicable"]

# Keys a salary case may never contain: they would assert a single correct wage.
FORBIDDEN_SALARY_KEYS = ("expected_salary_value", "correct_salary", "true_salary", "salary_answer")


class Provenance(BaseModel):
    """Where the expectation came from. Unknown facts are recorded as None, never invented."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="File the case was derived from, repo-relative.")
    source_row: int | None = Field(default=None, description="1-based row in the source file (header = 1).")
    source_sha256: str | None = Field(default=None, description="Hash of the source file at build time.")
    authority: str | None = Field(default=None, description="Legal authority, when the case asserts law.")
    law_snapshot: str | None = Field(default=None, description="Hash of labor_law_parsed.json at build time.")
    law_version: str | None = Field(default=None, description="None: the repository holds no law-version metadata.")
    effective_date: str | None = Field(default=None, description="None: not available in the repository.")
    unavailable_fields: list[str] = Field(default_factory=list,
                                          description="Provenance the repository cannot supply, named explicitly.")


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ReviewStatus
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    method: Literal["human_from_primary_source", "human_adjudicated", "mechanical_from_fixture",
                    "inherited_from_source_dataset", None] = None
    reason: str | None = Field(default=None, description="Required for 'quarantined' and 'needs_review'.")

    @model_validator(mode="after")
    def _honest_status(self):
        if self.status == "reviewed" and not (self.reviewed_by and self.reviewed_at and self.method):
            raise ValueError("a 'reviewed' case needs reviewed_by, reviewed_at and method")
        if self.status in ("quarantined", "needs_review") and not self.reason:
            raise ValueError(f"a '{self.status}' case needs a reason")
        return self


class Gold(BaseModel):
    """The expected regulatory target. Index and article number are kept separately, on purpose."""

    model_config = ConfigDict(extra="forbid")

    article_indices: list[int] = Field(default_factory=list, description="1-based knowledge-base index (1..249).")
    article_numbers: list[str] = Field(default_factory=list, description="Legal article number as printed.")

    @model_validator(mode="after")
    def _paired(self):
        if self.article_indices and self.article_numbers and len(self.article_indices) != len(self.article_numbers):
            raise ValueError("article_indices and article_numbers must be the same length")
        if any(i < 1 or i > 249 for i in self.article_indices):
            raise ValueError("article_indices must be within the knowledge base (1..249)")
        return self


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    task: Task
    title: str
    language: Language = "ar"
    input: dict[str, Any] = Field(description="What is sent to Sanad: question, documents, salary_query or scenario.")
    expected: dict[str, Any] = Field(default_factory=dict, description="Task-specific expectations.")
    gold: Gold | None = None
    provenance: Provenance
    review: Review
    duplicate_group: str | None = Field(default=None, description="Set when several cases share one question string.")
    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _case_rules(self):
        if not CASE_ID.match(self.case_id):
            raise ValueError(f"case_id '{self.case_id}' must be lower_snake_case")

        if self.task == "regulatory_question":
            if not self.input.get("question"):
                raise ValueError("a regulatory case needs input.question")
            if self.review.status != "quarantined":
                if self.gold is None or not self.gold.article_indices:
                    raise ValueError("a regulatory case needs gold.article_indices")
                if not self.gold.article_numbers:
                    raise ValueError("a regulatory case needs gold.article_numbers as well as indices")

        if self.task in ("contract_analysis", "contract_comparison") and not self.input.get("documents"):
            raise ValueError(f"a {self.task} case needs input.documents")

        if self.task == "salary_benchmark":
            found = [k for k in FORBIDDEN_SALARY_KEYS if k in self.expected]
            if found:
                raise ValueError(f"a salary case must not assert a single correct salary: {found}")

        # An expectation that asserts a legal conclusion cannot be mechanical.
        if self.expected.get("findings") and self.review.status == "generated_from_fixture":
            raise ValueError("compliance expectations are a legal judgement: use 'needs_review'")
        return self


def load_cases(path) -> list[GoldenCase]:
    """Read one .jsonl file into validated cases. Any invalid line raises immediately."""
    import json
    from pathlib import Path

    cases = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(GoldenCase.model_validate(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"{Path(path).name}:{number}: {exc}") from None
    return cases
