"""Clause-level regulatory retrieval: the existing hook, the existing retriever, no verdicts."""

from __future__ import annotations

import pytest

from agents.regulatory import CLAUSE_TYPE_TOPIC_NAMES, RegulatoryEvidenceCollector, topic_for_clause
from models.extraction import ClauseType, ClauseValue, ExtractionMethodName, SourceSpan
from models.regulatory import RegulatoryQuery
from tests.fakes.regulatory_adapter import FakeRegulatoryAdapter


def clause(text: str, clause_type: ClauseType, name: str = "Probation", clause_id: str = "C01") -> ClauseValue:
    return ClauseValue(
        clause_id=clause_id, clause_name=name, clause_type=clause_type, text=text, confidence="high",
        source_span=SourceSpan(text=text, page_number=3, section_id="s0010",
                               method=ExtractionMethodName.CLAUSE_SENTENCE),
    )


class HookSource:
    """The existing fake adapter, plus the convenience hook the real adapter offers.

    Evidence still comes from FakeRegulatoryAdapter, which builds it from the real knowledge base
    through the real mapping code, so these tests see the true result shape.
    """

    def __init__(self, knowledge_base) -> None:
        self.inner = FakeRegulatoryAdapter(knowledge_base)
        self.clause_calls: list[tuple] = []

    @property
    def query_calls(self) -> list[RegulatoryQuery]:
        return self.inner.calls

    def evidence_for_clause(self, question, clause_text, clause_name=None, top_k=None):
        self.clause_calls.append((question, clause_text, clause_name))
        return self.inner.retrieve_evidence(
            RegulatoryQuery(question=question, contract_context=clause_text, clause_name=clause_name))

    def retrieve_evidence(self, query, top_k=None):
        return self.inner.retrieve_evidence(query)


@pytest.fixture
def hook_source(knowledge_base) -> HookSource:
    return HookSource(knowledge_base)


@pytest.fixture
def plain_source(knowledge_base) -> FakeRegulatoryAdapter:
    """Only the documented protocol method, as the existing doubles have."""
    return FakeRegulatoryAdapter(knowledge_base)


# --------------------------------------------------------------------------- the mapping
def test_clause_types_map_only_onto_existing_topics():
    from agents.regulatory import CONTRACT_TOPICS
    existing = {topic.name for topic in CONTRACT_TOPICS}
    assert set(CLAUSE_TYPE_TOPIC_NAMES.values()) <= existing, "no new labor-law topic may be invented"


@pytest.mark.parametrize("clause_type, topic_name", [
    (ClauseType.PROBATION, "probation"),
    (ClauseType.NOTICE, "notice"),
    (ClauseType.TERMINATION, "termination"),
    (ClauseType.WORKING_HOURS, "working_hours"),
    (ClauseType.WEEKLY_REST, "weekly_rest"),
    (ClauseType.ANNUAL_LEAVE, "annual_leave"),
    (ClauseType.SALARY, "wage"),
    (ClauseType.ALLOWANCES, "allowances"),
])
def test_a_clause_type_resolves_to_its_existing_topic(clause_type, topic_name):
    assert topic_for_clause(clause("text", clause_type)).name == topic_name


@pytest.mark.parametrize("clause_type", [ClauseType.DUTIES, ClauseType.CONFIDENTIALITY,
                                         ClauseType.NON_COMPETE, ClauseType.OTHER, ClauseType.UNKNOWN])
def test_a_clause_with_no_suitable_topic_is_kept_without_one(clause_type):
    assert topic_for_clause(clause("text", clause_type)) is None


# --------------------------------------------------------------------------- the hook
def test_the_existing_evidence_for_clause_hook_is_used_when_offered(hook_source):
    item = clause("The initial period of probation is 180 days.", ClauseType.PROBATION)

    RegulatoryEvidenceCollector(hook_source).collect_for_clause(item, topic_for_clause(item))

    assert hook_source.clause_calls, "the adapter's evidence_for_clause() must be the entry point"
    question, clause_text, clause_name = hook_source.clause_calls[0]
    assert clause_text == item.text and clause_name == "Probation" and question
    # Everything the hook forwarded still went to the one existing retrieval implementation.
    assert len(hook_source.query_calls) == len(hook_source.clause_calls)


def test_a_source_without_the_hook_gets_the_query_the_hook_would_have_built(plain_source):
    item = clause("The initial period of probation is 180 days.", ClauseType.PROBATION)

    RegulatoryEvidenceCollector(plain_source).collect_for_clause(item, topic_for_clause(item))

    assert len(plain_source.calls) == 1
    query = plain_source.calls[0]
    assert query.contract_context == item.text
    assert query.clause_name == "Probation"


def test_a_clause_without_a_topic_is_never_sent_to_the_rag(hook_source):
    source = hook_source
    item = clause("You shall not disclose confidential information.", ClauseType.CONFIDENTIALITY, "Confidentiality")

    record = RegulatoryEvidenceCollector(source).collect_for_clause(item, topic_for_clause(item))

    assert source.clause_calls == [] and source.query_calls == []
    assert record.regulatory_topic is None
    assert record.check.status == "not_run"
    assert record.check.notes


# --------------------------------------------------------------------------- traceability
def test_the_record_keeps_the_clause_the_query_and_the_evidence_together(hook_source):
    source = hook_source
    item = clause("The initial period of probation is 180 days.", ClauseType.PROBATION, clause_id="C09")

    record = RegulatoryEvidenceCollector(source).collect_for_clause(item, topic_for_clause(item))

    assert record.clause_id == "C09"
    assert record.clause_type is ClauseType.PROBATION
    assert record.clause_text == item.text
    assert record.source_span is not None and record.page_number == 3
    assert record.regulatory_topic == "probation"
    assert record.queries and all(q for q in record.queries)
    assert record.check is not None and record.check.topic == "probation"
    # clause -> query -> evidence stays joined up for Stage 3.
    assert record.evidence, "the probation article should have come back"
    assert record.check.retrieved_evidence_ids
    first = record.evidence[0]
    assert first.evidence_id and first.retrievals
    assert first.retrievals[0].rank >= 1 and first.retrievals[0].score is not None


def test_stage_two_reaches_no_compliance_conclusion():
    """A verdict needs rule semantics Stage 2 does not have, so the record has nowhere to put one."""
    from models.analysis import ClauseCheck
    forbidden = {"status", "assessment", "compliant", "verdict", "finding_status"}
    assert forbidden.isdisjoint(ClauseCheck.model_fields)


# --------------------------------------------------------------------------- no second retriever
def test_the_clause_layer_introduces_no_retrieval_machinery():
    """Segmentation must not reach for Qdrant, embeddings, BM25 or rag.retriever."""
    import inspect

    import extraction.clauses as clauses_module

    source = inspect.getsource(clauses_module)
    for forbidden in ("qdrant", "llama_index", "rag.retriever", "BM25", "SentenceTransformer", "embedding"):
        assert forbidden not in source, f"{forbidden} must not appear in the clause layer"


def test_the_collector_has_exactly_one_way_to_reach_the_rag():
    """Checked on the imports, not the prose: the docstrings legitimately name Qdrant to say they
    never touch it."""
    import ast
    import inspect

    import agents.regulatory as regulatory

    tree = ast.parse(inspect.getsource(regulatory))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(m.startswith(("qdrant", "llama_index", "rag.retriever", "sentence_transformers"))
                   for m in imported), imported
    source = inspect.getsource(regulatory)
    assert "evidence_for_clause" in source and "retrieve_evidence" in source


# --------------------------------------------------------------------------- retrieval eligibility
def _clause(text, clause_type, confidence="high", clause_id="C01"):
    return ClauseValue(clause_id=clause_id, clause_name="X", clause_type=clause_type, text=text,
                       confidence=confidence,
                       source_span=SourceSpan(text=text, page_number=1, section_id="s1",
                                              method=ExtractionMethodName.CLAUSE_SENTENCE))


@pytest.mark.parametrize("clause_type, confidence, eligible", [
    (ClauseType.PROBATION, "high", True),
    (ClauseType.PROBATION, "medium", True),
    (ClauseType.PROBATION, "low", False),      # too weak to assert the area of law
    (ClauseType.PROBATION, None, False),
    (ClauseType.UNKNOWN, "high", False),       # subject never established
    (ClauseType.OTHER, "high", False),
    (ClauseType.CONFIDENTIALITY, "high", False),  # classified, but no existing topic covers it
])
def test_retrieval_eligibility(clause_type, confidence, eligible):
    from agents.regulatory import is_eligible_for_retrieval
    assert is_eligible_for_retrieval(_clause("text here", clause_type, confidence)) is eligible


def test_an_ineligible_clause_records_why_and_makes_no_call(hook_source):
    from agents.regulatory import retrieval_ineligibility

    item = _clause("Something the classifier could not place.", ClauseType.UNKNOWN)
    reason = retrieval_ineligibility(item)
    assert reason

    record = RegulatoryEvidenceCollector(hook_source).skip_clause(item, reason)

    assert hook_source.clause_calls == [] and hook_source.query_calls == []
    assert record.clause_text == item.text
    assert record.regulatory_topic is None
    assert record.queries == []
    assert record.check.status == "not_run" and reason in record.check.notes


def test_skipping_introduces_no_alternative_retrieval_path():
    import ast
    import inspect

    import agents.regulatory as regulatory

    tree = ast.parse(inspect.getsource(regulatory))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(m.startswith(("qdrant", "llama_index", "rag.retriever", "sentence_transformers"))
                   for m in imported), imported
