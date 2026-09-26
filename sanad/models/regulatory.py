"""Structured output of the Regulatory RAG adapter.

Everything a downstream agent needs to trace a regulatory conclusion back to its source is
preserved: source document, part, chapter, article identity, Arabic and English text,
retrieval rank/score, and the untouched legacy metadata.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from models.common import ErrorInfo, ResultStatus


class RegulatoryQuery(BaseModel):
    """A programmatic request to the existing RAG."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    contract_context: str | None = Field(
        default=None, description="Contract clause or excerpt the question is about."
    )
    clause_name: str | None = None
    employee_data: dict[str, Any] | None = Field(
        default=None,
        description="Passed unchanged to the legacy answer_policy_question(); only used by ask().",
    )

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value

    @field_validator("contract_context", "clause_name")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class SourceDocument(BaseModel):
    title_ar: str
    title_en: str
    publisher: str
    url: str | None = None
    file_path: str | None = None
    knowledge_base_file: str
    vector_collection: str | None = None
    english_content_is_machine_translation: bool = True


class ArticleReference(BaseModel):
    kb_index: int | None = Field(default=None, description="1-based position in labor_law_parsed.json ('index').")
    article_number: int | None = Field(default=None, description="Legal article number ('article_number').")
    article_number_ar: str | None = None
    article_name_ar: str | None = None
    english_number: str | None = None
    part_number: int | None = None
    part_number_ar: str | None = None
    part_title_ar: str | None = None
    chapter_number: int | None = None
    chapter_number_ar: str | None = None
    chapter_title_ar: str | None = None


class RegulatoryEvidence(BaseModel):
    reference: ArticleReference
    citation: str
    arabic_content: str | None = None
    english_content: str | None = None
    rank: int = Field(ge=1)
    score: float = Field(description="Legacy hybrid score (see RetrievalConfig.score_description).")
    score_is_rounded: bool = Field(description="True when taken from answer_policy_question (rounded to 3 dp).")
    source: SourceDocument
    metadata_complete: bool
    raw_metadata: dict[str, Any] = Field(description="Legacy article metadata, unmodified.")
    legacy_reference: dict[str, Any] | None = Field(
        default=None, description="The reference dict exactly as answer_policy_question returned it."
    )

    @computed_field(description="`score` as a whole percentage, for display. This is a SIMILARITY score - how "
                                "closely the retrieved article matches the query - and never a statement about "
                                "how correct the answer is. Derived, not stored: the single source of truth "
                                "stays `score`.")
    @property
    def similarity_percentage(self) -> int:
        return round(self.score * 100)


class RetrievalConfig(BaseModel):
    """Retrieval settings read from rag/retriever.py (not redefined anywhere else in Sanad)."""

    method: Literal["hybrid_dense_bm25"] = "hybrid_dense_bm25"
    qdrant_url: str
    collection: str
    embedding_model: str
    top_k: int
    alpha: float
    dense_similarity_top_k: int | None = None
    score_description: str = (
        "alpha * minmax(dense cosine over the dense top-k) + (1 - alpha) * minmax(BM25 over all articles); "
        "relative within one query, not a calibrated probability"
    )
    source_file: str


class RegulatoryEvidenceResult(BaseModel):
    status: ResultStatus
    query: RegulatoryQuery
    retrieval_query: str | None = Field(default=None, description="Exact text sent to the existing RAG.")
    detected_language: Literal["ar", "en"] | None = None
    evidence: list[RegulatoryEvidence] = Field(default_factory=list)
    retrieval_config: RetrievalConfig | None = None
    legacy_entry_point: str
    errors: list[ErrorInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    duration_ms: float | None = None

    @model_validator(mode="after")
    def _status_invariants(self):
        if self.status is ResultStatus.ERROR and not self.errors:
            raise ValueError("status 'error' requires at least one ErrorInfo")
        if self.status in (ResultStatus.SUCCESS, ResultStatus.PARTIAL) and not self.evidence:
            raise ValueError(f"status '{self.status.value}' requires evidence")
        return self

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence)


class RegulatoryAnswerResult(RegulatoryEvidenceResult):
    answer: str | None = Field(default=None, description="Answer generated by the existing RAG prompt, unchanged.")
    answer_model: str | None = None
    message: str | None = Field(default=None, description="Non-answer message returned by the legacy RAG, if any.")


class RagHealth(BaseModel):
    status: ResultStatus
    legacy_rag_dir: str
    legacy_files_present: bool
    legacy_backend_loaded: bool
    knowledge_base_article_count: int | None = None
    retrieval_config: RetrievalConfig | None = None
    vector_db_url: str | None = None
    vector_db_reachable: bool | None = None
    collection_exists: bool | None = None
    collection_points_count: int | None = None
    collection_vector_size: int | None = None
    collection_distance: str | None = None
    errors: list[ErrorInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
