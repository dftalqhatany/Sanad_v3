"""Focused regulatory questions and evidence collection through the existing RAG adapter.

The only regulatory data source is RegulatoryRAGAdapter.retrieve_evidence() (duck-typed as
RegulatoryEvidenceSource so tests can supply a double). This module never touches Qdrant,
embeddings, BM25 or the legacy retriever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Protocol

from sanad.agents.errors import AnalysisErrorCode, analysis_error
from sanad.extraction.text import matching_key
from sanad.models.analysis import EvidenceReference, EvidenceRetrieval, RegulatoryCheck
from sanad.models.common import ErrorInfo, ResultStatus
from sanad.models.regulatory import RegulatoryEvidence, RegulatoryEvidenceResult, RegulatoryQuery


class RegulatoryEvidenceSource(Protocol):
    """The subset of sanad.rag.adapter.RegulatoryRAGAdapter used by the analysis agent."""

    def retrieve_evidence(self, query: RegulatoryQuery | str, top_k: int | None = None) -> RegulatoryEvidenceResult: ...


@dataclass(frozen=True)
class RegulatoryTopic:
    name: str
    fields: tuple[str, ...]
    questions: tuple[str, ...]  # Arabic first: the knowledge base's legal reference text is Arabic
    keyword_groups: tuple[tuple[str, ...], ...]  # an article is relevant when every group has a keyword in its Arabic text
    related_fields: tuple[str, ...] = ()

    @cached_property
    def keyword_patterns(self) -> tuple[re.Pattern[str], ...]:
        """One pattern per group; a keyword must start a word (after optional و/ف, ب/ل/ك and ال/ل clitics)."""
        return tuple(
            re.compile(_WORD_START + "(?:" + "|".join(re.escape(matching_key(k)) for k in group) + ")")
            for group in self.keyword_groups
        )

    def mentioned_in(self, arabic_text: str | None) -> bool:
        text = matching_key(arabic_text or "")
        return all(pattern.search(text) for pattern in self.keyword_patterns)


# e.g. "للتجربة" and "بالإجازة" match, "صراحة" does not match "راحة"
_WORD_START = r"(?<!\S)(?:[وف])?(?:[بلك])?(?:ال|ل)?"


CONTRACT_TOPICS: tuple[RegulatoryTopic, ...] = (
    RegulatoryTopic(
        "probation", ("probation_period",),
        ("ما هي المدة القصوى لفترة التجربة في عقد العمل؟",
         "What is the maximum probation period allowed in an employment contract?"),
        (("تجربة",),),
    ),
    RegulatoryTopic(
        "annual_leave", ("annual_leave",),
        ("ما هي مدة الإجازة السنوية التي يستحقها العامل؟",
         "What is the minimum paid annual leave a worker is entitled to?"),
        (("إجازة",), ("سنوية",)),
    ),
    RegulatoryTopic(
        "working_hours", ("working_hours",),
        ("ما هو الحد الأقصى لساعات العمل في اليوم وفي الأسبوع؟",
         "What are the maximum daily and weekly working hours?"),
        (("ساعات", "ساعة"),),
    ),
    RegulatoryTopic(
        "weekly_rest", ("working_days",),
        ("ما هي أحكام يوم الراحة الأسبوعية للعامل؟",
         "What are the rules for the worker's weekly rest day?"),
        (("راحة",),),
    ),
    RegulatoryTopic(
        "notice", ("notice_period",),
        ("ما هي مدة الإشعار المطلوبة لإنهاء عقد العمل؟",
         "What notice period is required to terminate an employment contract?"),
        (("إشعار", "إنذار"),),
        related_fields=("contract_type", "salary"),
    ),
    RegulatoryTopic(
        "contract_term", ("contract_type", "contract_duration", "end_date"),
        ("ما هي أحكام عقد العمل محدد المدة وتجديده؟",
         "What are the rules for fixed-term employment contracts and their renewal?"),
        (("محدد المدة", "محددة المدة", "غير محدد المدة"),),
        related_fields=("contract_type", "contract_duration", "start_date", "end_date", "nationality"),
    ),
    RegulatoryTopic(
        "termination", ("termination_terms",),
        ("متى ينتهي عقد العمل وما هي أحكام إنهائه؟",
         "When does an employment contract end and what are the rules for terminating it?"),
        (("إنهاء", "فسخ", "انتهاء"),),
        related_fields=("contract_type",),
    ),
    RegulatoryTopic(
        "wage", ("salary", "total_salary"),
        ("ما هي أحكام أجر العامل وطريقة دفعه؟",
         "What are the rules governing the worker's wage and its payment?"),
        (("أجر",),),
    ),
    RegulatoryTopic(
        "allowances", ("housing_allowance", "transportation_allowance", "other_allowances"),
        ("هل تدخل البدلات مثل بدل السكن وبدل النقل ضمن أجر العامل؟",
         "Are allowances such as housing and transportation part of the worker's wage?"),
        (("بدل",), ("أجر",)),
        related_fields=("salary",),
    ),
)
TOPIC_BY_FIELD: dict[str, RegulatoryTopic] = {name: topic for topic in CONTRACT_TOPICS for name in topic.fields}

# Adapter error codes that mean the existing RAG itself cannot be used (as opposed to one failed query).
RAG_UNAVAILABLE_CODES = frozenset({
    "legacy_rag_not_found", "legacy_interface_changed", "dependency_missing", "knowledge_base_missing",
    "embedding_model_unavailable", "vector_db_unreachable", "collection_not_found", "rag_initialization_failed",
})


@dataclass
class TopicEvidence:
    check: RegulatoryCheck
    relevant: list[EvidenceReference] = field(default_factory=list)


class RegulatoryEvidenceCollector:
    def __init__(self, source: RegulatoryEvidenceSource, *, max_evidence_per_topic: int = 6) -> None:
        self.source = source
        self.max_evidence_per_topic = max_evidence_per_topic
        self.registry: dict[str, EvidenceReference] = {}
        self.errors: list[ErrorInfo] = []
        self.rag_unavailable: ErrorInfo | None = None
        self.queries_sent = 0

    def collect(self, topic: RegulatoryTopic, triggered_by_fields: list[str]) -> TopicEvidence:
        check = RegulatoryCheck(topic=topic.name, triggered_by_fields=triggered_by_fields,
                                questions=list(topic.questions), status="not_run")
        if self.rag_unavailable is not None:
            check.status = "error"
            check.errors.append(self.rag_unavailable)
            check.notes.append("Not queried: the existing RAG was unavailable for an earlier question.")
            return TopicEvidence(check)

        topic_refs: dict[str, EvidenceReference] = {}
        for question in topic.questions:
            try:
                self.queries_sent += 1
                result = self.source.retrieve_evidence(RegulatoryQuery(question=question))
            except Exception as exc:  # the adapter contract is to never raise; treat a raise as RAG unavailable
                error = analysis_error(AnalysisErrorCode.RAG_UNAVAILABLE, "regulatory_retrieval",
                                       "The regulatory RAG adapter raised an unexpected exception.", exc)
                self._record(check, error, unavailable=True)
                break
            if result.status is ResultStatus.ERROR:
                unavailable = any(e.code in RAG_UNAVAILABLE_CODES or e.stage == "initialization" for e in result.errors)
                for error in result.errors:
                    self._record(check, error, unavailable=unavailable)
                if unavailable:
                    break
                continue
            for item in result.evidence:
                reference = self._register(item, topic, result.retrieval_query or question)
                topic_refs[reference.evidence_id] = reference

        retrieved = list(topic_refs.values())
        check.retrieved_evidence_ids = [ref.evidence_id for ref in retrieved]
        relevant = [ref for ref in retrieved if topic.name in ref.relevant_topics]
        relevant.sort(key=lambda ref: (self._best_rank(ref, topic.name), -self._best_score(ref, topic.name)))
        if len(relevant) > self.max_evidence_per_topic:
            check.notes.append(f"{len(relevant)} relevant articles retrieved; the {self.max_evidence_per_topic} best ranked are used.")
            relevant = relevant[: self.max_evidence_per_topic]
        check.relevant_evidence_ids = [ref.evidence_id for ref in relevant]
        if relevant:
            check.status = "evidence_found"
        elif check.errors:
            check.status = "error"
        else:
            check.status = "insufficient_evidence"
            check.notes.append(
                f"{len(retrieved)} articles were retrieved, but none mentions this topic "
                f"({' + '.join('/'.join(group) for group in topic.keyword_groups)}) with a positive retrieval score."
            )
        return TopicEvidence(check, relevant)

    # ------------------------------------------------------------------ internals
    def _record(self, check: RegulatoryCheck, error: ErrorInfo, *, unavailable: bool) -> None:
        check.errors.append(error)
        if error not in self.errors:
            self.errors.append(error)
        if unavailable and self.rag_unavailable is None:
            self.rag_unavailable = error

    def _register(self, item: RegulatoryEvidence, topic: RegulatoryTopic, query: str) -> EvidenceReference:
        kb_index = item.reference.kb_index
        evidence_id = f"kb-{kb_index}" if kb_index is not None else f"{topic.name}-{len(self.registry) + 1}"
        reference = self.registry.get(evidence_id)
        if reference is None:
            reference = EvidenceReference(evidence_id=evidence_id, evidence=item)
            self.registry[evidence_id] = reference
        reference.retrievals.append(EvidenceRetrieval(topic=topic.name, query=query, rank=item.rank, score=item.score,
                                                      score_is_rounded=item.score_is_rounded))
        if item.score > 0 and topic.mentioned_in(item.arabic_content) and topic.name not in reference.relevant_topics:
            reference.relevant_topics.append(topic.name)
        return reference

    @staticmethod
    def _best_rank(reference: EvidenceReference, topic: str) -> int:
        return min(r.rank for r in reference.retrievals if r.topic == topic)

    @staticmethod
    def _best_score(reference: EvidenceReference, topic: str) -> float:
        return max(r.score for r in reference.retrievals if r.topic == topic)
