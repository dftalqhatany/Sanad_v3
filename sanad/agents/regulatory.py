"""Focused regulatory questions and evidence collection through the existing RAG adapter.

The only regulatory data source is RegulatoryRAGAdapter.retrieve_evidence() (duck-typed as
RegulatoryEvidenceSource so tests can supply a double). This module never touches Qdrant,
embeddings, BM25 or rag.retriever directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Protocol

from agents.errors import AnalysisErrorCode, analysis_error
from extraction.text import matching_key
from models.analysis import ClauseCheck, EvidenceReference, EvidenceRetrieval, RegulatoryCheck
from models.common import ErrorInfo, ResultStatus
from models.extraction import ClauseType, ClauseValue
from models.regulatory import RegulatoryEvidence, RegulatoryEvidenceResult, RegulatoryQuery


class RegulatoryEvidenceSource(Protocol):
    """The subset of sanad.rag.adapter.RegulatoryRAGAdapter used by the analysis agent.

    `evidence_for_clause` is optional: the adapter provides it, and a source that does not is called
    through `retrieve_evidence` with the same RegulatoryQuery the adapter would have built. Either
    way there is exactly one retrieval implementation - the existing one.
    """

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

# Clause type -> the name of an EXISTING topic above. No new labor-law topic is invented here: a
# clause type with no suitable existing topic is simply kept without one, and is not sent to the RAG.
CLAUSE_TYPE_TOPIC_NAMES: dict[ClauseType, str] = {
    ClauseType.PROBATION: "probation",
    ClauseType.NOTICE: "notice",
    ClauseType.TERMINATION: "termination",
    ClauseType.WORKING_HOURS: "working_hours",
    ClauseType.WEEKLY_REST: "weekly_rest",
    ClauseType.WORKING_DAYS: "weekly_rest",
    ClauseType.ANNUAL_LEAVE: "annual_leave",
    ClauseType.SALARY: "wage",
    ClauseType.ALLOWANCES: "allowances",
    ClauseType.HOUSING: "allowances",
    ClauseType.TRANSPORTATION: "allowances",
    ClauseType.CONTRACT_DURATION: "contract_term",
    ClauseType.START_DATE: "contract_term",
}
TOPIC_BY_NAME: dict[str, RegulatoryTopic] = {topic.name: topic for topic in CONTRACT_TOPICS}


def topic_for_clause(clause: ClauseValue,
                     topics: tuple[RegulatoryTopic, ...] = CONTRACT_TOPICS) -> RegulatoryTopic | None:
    """The existing regulatory topic a clause should be checked against, or None."""
    name = CLAUSE_TYPE_TOPIC_NAMES.get(clause.clause_type)
    if name is None:
        return None
    return {topic.name: topic for topic in topics}.get(name)


# Retrieval eligibility ---------------------------------------------------------------------------
# Asking the regulatory RAG about a clause is a claim that the clause is about that area of the law.
# A weakly classified clause does not support that claim, so it is kept in the extraction result and
# left unqueried rather than being checked against an area it may have nothing to do with.
RETRIEVAL_CONFIDENCE = frozenset({"high", "medium"})
UNRETRIEVABLE_TYPES = frozenset({ClauseType.UNKNOWN, ClauseType.OTHER})


def retrieval_ineligibility(clause: ClauseValue,
                            topics: tuple[RegulatoryTopic, ...] = CONTRACT_TOPICS) -> str | None:
    """Why this clause must not be sent to the RAG, or None when it may be."""
    if clause.clause_type in UNRETRIEVABLE_TYPES:
        return (f"The clause is recorded as '{clause.clause_type.value}': its subject was not established, "
                "so no regulatory question can honestly be asked about it.")
    if clause.confidence not in RETRIEVAL_CONFIDENCE:
        return (f"The clause was classified with {clause.confidence or 'no'} confidence; that is too weak to "
                "assert what area of the law it belongs to.")
    if topic_for_clause(clause, topics) is None:
        return (f"No existing regulatory topic covers a '{clause.clause_type.value}' clause; it was kept "
                "without one and no question was asked.")
    return None


def is_eligible_for_retrieval(clause: ClauseValue,
                              topics: tuple[RegulatoryTopic, ...] = CONTRACT_TOPICS) -> bool:
    return retrieval_ineligibility(clause, topics) is None

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
    def __init__(self, source: RegulatoryEvidenceSource, *, max_evidence_per_topic: int = 6,
                 clause_questions: int = 1) -> None:
        self.source = source
        self.max_evidence_per_topic = max_evidence_per_topic
        # One question per clause by default: the clause text is already part of the retrieval query,
        # and the Arabic question is the legal reference language. A document has far more clauses
        # than topics, so asking every phrasing of every question per clause is mostly duplicate work.
        self.clause_questions = max(1, clause_questions)
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

    # ------------------------------------------------------------------ clause level
    def skip_clause(self, clause: ClauseValue, reason: str) -> ClauseCheck:
        """Record a clause that is preserved but not queried. Makes no call to the RAG."""
        return ClauseCheck(
            clause_id=clause.clause_id or "C00", clause_type=clause.clause_type,
            clause_name=clause.clause_name, clause_text=clause.text, source_span=clause.source_span,
            regulatory_topic=None,
            check=RegulatoryCheck(topic="", triggered_by_fields=[], questions=[], status="not_run",
                                  notes=[reason]),
        )

    def collect_for_clause(self, clause: ClauseValue, topic: RegulatoryTopic | None) -> ClauseCheck:
        """Regulatory evidence for one clause, through the existing adapter hook.

        Stage 2 stops here: the clause, the questions asked about it and the articles that came back
        are recorded together, and nothing is concluded from them.
        """
        record = ClauseCheck(
            clause_id=clause.clause_id or "C00", clause_type=clause.clause_type,
            clause_name=clause.clause_name, clause_text=clause.text, source_span=clause.source_span,
            regulatory_topic=topic.name if topic else None,
        )
        if topic is None:
            record.check = RegulatoryCheck(
                topic="", triggered_by_fields=[], questions=[], status="not_run",
                notes=[f"No existing regulatory topic covers a '{clause.clause_type.value}' clause; "
                       "it was kept without one and no question was asked."])
            return record

        questions = list(topic.questions[: self.clause_questions])
        check = RegulatoryCheck(topic=topic.name, triggered_by_fields=[], questions=questions, status="not_run")
        record.check = check
        record.queries = questions
        if self.rag_unavailable is not None:
            check.status = "error"
            check.errors.append(self.rag_unavailable)
            check.notes.append("Not queried: the existing RAG was unavailable for an earlier question.")
            return record

        refs: dict[str, EvidenceReference] = {}
        for question in questions:
            try:
                self.queries_sent += 1
                result = self._clause_evidence(question, clause)
            except Exception as exc:
                error = analysis_error(AnalysisErrorCode.RAG_UNAVAILABLE, "regulatory_retrieval",
                                       "The regulatory RAG adapter raised an unexpected exception.", exc)
                self._record(check, error, unavailable=True)
                break
            if result.status is ResultStatus.ERROR:
                unavailable = any(e.code in RAG_UNAVAILABLE_CODES or e.stage == "initialization"
                                  for e in result.errors)
                for error in result.errors:
                    self._record(check, error, unavailable=unavailable)
                if unavailable:
                    break
                continue
            if result.retrieval_query:
                record.retrieval_queries.append(result.retrieval_query)
            for item in result.evidence:
                reference = self._register(item, topic, result.retrieval_query or question)
                refs[reference.evidence_id] = reference

        retrieved = list(refs.values())
        check.retrieved_evidence_ids = [ref.evidence_id for ref in retrieved]
        relevant = [ref for ref in retrieved if topic.name in ref.relevant_topics]
        relevant.sort(key=lambda ref: (self._best_rank(ref, topic.name), -self._best_score(ref, topic.name)))
        if len(relevant) > self.max_evidence_per_topic:
            relevant = relevant[: self.max_evidence_per_topic]
        check.relevant_evidence_ids = [ref.evidence_id for ref in relevant]
        if relevant:
            check.status = "evidence_found"
        elif check.errors:
            check.status = "error"
        else:
            check.status = "insufficient_evidence"
            check.notes.append(f"{len(retrieved)} articles were retrieved, but none mentions this topic.")
        record.evidence = relevant or retrieved
        return record

    def _clause_evidence(self, question: str, clause: ClauseValue) -> RegulatoryEvidenceResult:
        """The existing hook when the source offers it, otherwise the query the hook would have built."""
        hook = getattr(self.source, "evidence_for_clause", None)
        if callable(hook):
            return hook(question, clause.text, clause.clause_name)
        return self.source.retrieve_evidence(
            RegulatoryQuery(question=question, contract_context=clause.text, clause_name=clause.clause_name))

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
