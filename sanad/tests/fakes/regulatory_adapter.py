"""Double of RegulatoryRAGAdapter.retrieve_evidence(), the only RAG method the analysis agent uses.

Evidence items are built by the REAL sanad.rag.mapping.evidence_from_retriever_result() from REAL
knowledge-base articles, so their shape, citations and metadata are exactly what the adapter returns.
Only the choice of articles and scores per question is scripted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sanad.agents.regulatory import CONTRACT_TOPICS
from sanad.config import SanadSettings
from sanad.models.common import ErrorInfo, ResultStatus
from sanad.models.regulatory import (
    RegulatoryAnswerResult,
    RegulatoryEvidenceResult,
    RegulatoryQuery,
    SourceDocument,
)
from sanad.rag.mapping import evidence_from_retriever_result

ENTRY_POINT = "fake RegulatoryRAGAdapter.retrieve_evidence"

# knowledge-base index (1-based "index" in labor_law_parsed.json) -> legal article number
KB_ARTICLE_1, KB_ARTICLE_2, KB_ARTICLE_3 = 1, 2, 3  # 1 and 3 mention none of the contract topics
KB_PROBATION_ART_53 = 54
KB_CONTRACT_TERM_ART_55 = 56
KB_TERMINATION_ART_74 = 75
KB_NOTICE_ART_75 = 76
KB_WAGE_ART_90 = 92
KB_WORKING_HOURS_ART_98 = 100
KB_WEEKLY_REST_ART_104 = 106
KB_ANNUAL_LEAVE_ART_109 = 111

NOISE = [(KB_ARTICLE_1, 0.41), (KB_ARTICLE_3, 0.37)]
DEFAULT_TOPIC_HITS: dict[str, list[tuple[int, float]]] = {
    "probation": [(KB_PROBATION_ART_53, 0.83), *NOISE],
    "annual_leave": [(KB_ANNUAL_LEAVE_ART_109, 0.88), *NOISE],
    "working_hours": [(KB_WORKING_HOURS_ART_98, 0.86), *NOISE],
    "weekly_rest": [(KB_WEEKLY_REST_ART_104, 0.8), *NOISE],
    "notice": [(KB_NOTICE_ART_75, 0.79), *NOISE],
    "contract_term": [(KB_CONTRACT_TERM_ART_55, 0.81), *NOISE],
    "termination": [(KB_TERMINATION_ART_74, 0.78), *NOISE],
    "wage": [(KB_WAGE_ART_90, 0.77), *NOISE],
    "allowances": [(KB_ARTICLE_2, 0.74), *NOISE],
}
TOPIC_BY_QUESTION = {question: topic.name for topic in CONTRACT_TOPICS for question in topic.questions}


def source_document() -> SourceDocument:
    settings = SanadSettings()
    return SourceDocument(
        title_ar=settings.source_title_ar, title_en=settings.source_title_en, publisher=settings.source_publisher,
        url=settings.source_url, file_path=str(settings.source_document_path),
        knowledge_base_file=str(settings.knowledge_base_path), vector_collection="saudi_labor_law",
    )


@dataclass
class FakeRegulatoryAdapter:
    knowledge_base: list[dict]
    hits_by_topic: dict[str, list[tuple[int, float]]] = field(default_factory=lambda: dict(DEFAULT_TOPIC_HITS))
    default_hits: list[tuple[int, float]] = field(default_factory=lambda: list(NOISE))
    error_code: str | None = None  # every query fails with this adapter error code
    error_stage: str = "retrieval"
    error_topics: tuple[str, ...] = ()  # only these topics fail (when error_code is set)
    raise_exc: BaseException | None = None
    answer: str | None = "Scripted answer from the existing RAG."  # None means no API key is configured
    calls: list[RegulatoryQuery] = field(default_factory=list)
    ask_calls: list[RegulatoryQuery] = field(default_factory=list)

    def retrieve_evidence(self, query: RegulatoryQuery | str, top_k: int | None = None) -> RegulatoryEvidenceResult:
        query = query if isinstance(query, RegulatoryQuery) else RegulatoryQuery(question=query)
        self.calls.append(query)
        topic = TOPIC_BY_QUESTION.get(query.question)
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.error_code and (not self.error_topics or topic in self.error_topics):
            return RegulatoryEvidenceResult(
                status=ResultStatus.ERROR, query=query, retrieval_query=query.question, legacy_entry_point=ENTRY_POINT,
                errors=[ErrorInfo(code=self.error_code, stage=self.error_stage, message="scripted adapter failure")],
            )
        hits = self.hits_by_topic.get(topic, self.default_hits)
        source = source_document()
        evidence = [
            evidence_from_retriever_result(
                {"content": self.knowledge_base[kb - 1]["arabic_content"], "metadata": self.knowledge_base[kb - 1],
                 "score": score},
                rank=rank, source=source)
            for rank, (kb, score) in enumerate(hits, 1)
        ]
        status = ResultStatus.SUCCESS if any(item.score > 0 for item in evidence) else ResultStatus.INSUFFICIENT_EVIDENCE
        return RegulatoryEvidenceResult(status=status, query=query, retrieval_query=query.question, evidence=evidence,
                                        legacy_entry_point=ENTRY_POINT)

    def ask(self, query: RegulatoryQuery | str, api_key: str | None = None) -> RegulatoryAnswerResult:
        """Mirrors RegulatoryRAGAdapter.ask(): the legacy answer plus the evidence it was based on."""
        query = query if isinstance(query, RegulatoryQuery) else RegulatoryQuery(question=query)
        self.ask_calls.append(query)
        evidence = self.retrieve_evidence(query)
        if self.answer is None:
            return RegulatoryAnswerResult(
                status=ResultStatus.ERROR, query=query, retrieval_query=query.question, legacy_entry_point=ENTRY_POINT,
                errors=[ErrorInfo(code="missing_api_key", stage="answer_generation",
                                  message="OPENAI_API_KEY is not set, so the existing RAG cannot generate an answer.")])
        return RegulatoryAnswerResult(**evidence.model_dump(), answer=self.answer, answer_model="gpt-4o-mini")

    @property
    def questions(self) -> list[str]:
        return [call.question for call in self.calls]

    def topics_queried(self) -> set[str]:
        return {TOPIC_BY_QUESTION[q] for q in self.questions if q in TOPIC_BY_QUESTION}
