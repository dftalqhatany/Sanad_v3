"""Mandatory clause regression: the real Aramco appointment letter, end to end.

parse -> semantic extraction -> clause segmentation -> classification -> clause-level retrieval.
Nothing in the chain is mocked except the infrastructure underneath the existing RAG (Qdrant, the
embedding model), which is unavailable in the test environment; the adapter, the retriever module
and the segmentation logic are all the real ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.regulatory import RegulatoryEvidenceCollector, topic_for_clause
from extraction.contract import extract_contract
from extraction.text import flat
from models.extraction import ClauseType
from models.regulatory import RegulatoryQuery
from tests.fakes.regulatory_adapter import FakeRegulatoryAdapter

FIXTURE = "aramco_appointment_letter.pdf"
LETTER_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "documents" / FIXTURE


@pytest.fixture(scope="module")
def extraction(processor):
    if not LETTER_PATH.exists():
        pytest.skip(f"{FIXTURE} is not checked in at {LETTER_PATH}")
    document = processor.parse_bytes(LETTER_PATH.read_bytes(), FIXTURE)
    assert document.status.has_text
    return extract_contract(document)


@pytest.fixture(scope="module")
def clauses(extraction):
    return extraction.clauses


def _of_type(clauses, clause_type):
    return [c for c in clauses if c.clause_type is clause_type]


def _named(clauses, name):
    return [c for c in clauses if c.clause_name == name]


# --------------------------------------------------------------------------- the nine required clauses
def test_the_letter_is_segmented_into_classified_clauses(clauses):
    assert len(clauses) >= 20
    assert all(c.clause_id and c.text for c in clauses)


REQUIRED = [
    (ClauseType.PROBATION, "The initial period of probation is 180 days"),
    (ClauseType.NOTICE, "During the period of probation"),
    (ClauseType.NOTICE, "After confirmation"),
    (ClauseType.WORKING_HOURS, "required to work up to 40 hours a week"),
    (ClauseType.WEEKLY_REST, "weekly off would be for 2 days"),
    (ClauseType.ANNUAL_LEAVE, "24 paid leave per annum"),
    (ClauseType.SALARY, "In hand monthly- 1800 Riyal"),
    (ClauseType.BENEFITS, "Free Transportation"),
    (ClauseType.DUTIES, "Liaising with clients"),
]


@pytest.mark.parametrize("clause_type, quote", REQUIRED)
def test_a_required_clause_is_present_with_its_own_words(clauses, clause_type, quote):
    matching = [c for c in _of_type(clauses, clause_type) if flat(quote) in flat(c.text)]
    assert matching, f"no {clause_type.value} clause quoting {quote!r}"


# --------------------------------------------------------------------------- subject binding
def test_probation_is_its_own_clause_and_holds_180_days(clauses):
    probation = _of_type(clauses, ClauseType.PROBATION)
    assert len(probation) == 1
    assert "180 days" in probation[0].text


def test_the_probation_clause_does_not_contain_the_90_day_notice_sentence(clauses):
    """The bug this whole layer is built to prevent."""
    probation = _of_type(clauses, ClauseType.PROBATION)[0]
    assert "90 days" not in probation.text
    assert "During the period of probation, either side can terminate" not in probation.text


def test_both_notice_clauses_exist_separately_with_90_days_each(clauses):
    during = _named(clauses, "Notice during probation")
    after = _named(clauses, "Notice after confirmation")
    assert len(during) == 1 and len(after) == 1
    assert "90 days" in during[0].text and "90 days" in after[0].text
    assert during[0].clause_id != after[0].clause_id
    assert during[0].text != after[0].text


def test_probation_and_notice_clauses_never_share_text(clauses):
    probation = _of_type(clauses, ClauseType.PROBATION)[0]
    for notice in _of_type(clauses, ClauseType.NOTICE):
        assert notice.text != probation.text
        assert flat(notice.text) not in flat(probation.text)


# --------------------------------------------------------------------------- provenance
def test_every_clause_is_traceable_to_the_document(extraction, clauses, processor):
    document = processor.parse_bytes(LETTER_PATH.read_bytes(), FIXTURE)
    full = flat(document.full_text)
    for c in clauses:
        assert c.source_span is not None, c.clause_id
        assert c.page_number and 1 <= c.page_number <= 7, c.clause_id
        assert c.section_id, c.clause_id
        assert c.extraction_method is not None, c.clause_id
        assert c.confidence in ("high", "medium", "low"), c.clause_id
        assert flat(c.text) in full, (c.clause_id, c.text[:80])


def test_clause_ids_are_unique(clauses):
    ids = [c.clause_id for c in clauses]
    assert len(ids) == len(set(ids))


def test_foreign_legal_references_survive_segmentation(clauses):
    """Stage 3 has to be able to see that this Saudi-placed letter cites Indian law."""
    text = " ".join(c.text for c in clauses)
    assert "Maternity Benefit Act, 1961" in text
    assert "LPA" in text  # Indian salary notation, kept as written


# --------------------------------------------------------------------------- clause -> topic -> RAG
class RecordingSource:
    """The existing fake adapter with the hook exposed, so evidence keeps its real shape.

    It stands in for the infrastructure under the RAG (Qdrant, the embedding model), never for the
    segmentation or classification logic these tests are about.
    """

    def __init__(self, knowledge_base) -> None:
        self.inner = FakeRegulatoryAdapter(knowledge_base)
        self.calls: list[tuple[str, str, str | None]] = []

    def evidence_for_clause(self, question, clause_text, clause_name=None, top_k=None):
        self.calls.append((question, clause_text, clause_name))
        return self.inner.retrieve_evidence(
            RegulatoryQuery(question=question, contract_context=clause_text, clause_name=clause_name))

    def retrieve_evidence(self, query, top_k=None):
        raise AssertionError("clause retrieval must go through evidence_for_clause()")


@pytest.fixture
def source(knowledge_base) -> RecordingSource:
    return RecordingSource(knowledge_base)


def test_clauses_with_a_topic_are_sent_through_the_existing_hook(clauses, source):
    collector = RegulatoryEvidenceCollector(source)

    records = [collector.collect_for_clause(c, topic_for_clause(c)) for c in clauses]

    sent = [r for r in records if r.regulatory_topic is not None]
    assert sent, "at least the probation, notice and leave clauses must reach the RAG"
    assert len(source.calls) == sum(len(r.queries) for r in sent)
    for _question, clause_text, clause_name in source.calls:
        assert clause_text and clause_name


def test_the_probation_clause_is_checked_against_the_existing_probation_topic(clauses, source):
    probation = _of_type(clauses, ClauseType.PROBATION)[0]
    record = RegulatoryEvidenceCollector(source).collect_for_clause(probation, topic_for_clause(probation))
    assert record.regulatory_topic == "probation"
    assert record.clause_text == probation.text
    assert record.queries


def test_unmapped_clause_types_are_preserved_without_a_topic(clauses, source):
    collector = RegulatoryEvidenceCollector(source)
    confidentiality = _of_type(clauses, ClauseType.CONFIDENTIALITY)
    assert confidentiality, "the letter has a confidentiality section"
    record = collector.collect_for_clause(confidentiality[0], topic_for_clause(confidentiality[0]))
    assert record.regulatory_topic is None
    assert record.clause_text == confidentiality[0].text
    assert source.calls == []


# --------------------------------------------------------------------------- Stage 2 draws no verdict
def test_no_clause_record_states_a_compliance_outcome(clauses, source):
    collector = RegulatoryEvidenceCollector(source)
    for c in clauses[:10]:
        record = collector.collect_for_clause(c, topic_for_clause(c))
        dumped = record.model_dump()
        assert "compliant" not in dumped and "assessment" not in dumped
        if record.check is not None:
            assert record.check.status in ("evidence_found", "insufficient_evidence", "error", "not_run")


# --------------------------------------------------------------------------- hardening regressions
def test_the_retirement_rule_is_not_part_of_the_weekly_rest_clause(clauses):
    """The letter's item 15 states a retirement age. It is not a weekly-rest rule and must not be
    carried into the clause that gets checked against the weekly-rest article."""
    weekly = _of_type(clauses, ClauseType.WEEKLY_REST)
    assert len(weekly) == 1
    assert "weekly off would be for 2 days" in weekly[0].text
    assert "automatically retire" not in weekly[0].text
    assert "age of 62" not in weekly[0].text


def test_the_retirement_rule_survives_as_its_own_clause(clauses):
    """Preserved, not deleted - Stage 3 may still want to see it."""
    retirement = [c for c in clauses if "automatically retire" in c.text]
    assert len(retirement) == 1
    assert retirement[0].clause_type in (ClauseType.OTHER, ClauseType.UNKNOWN)


def test_general_mentions_of_benefits_are_not_classified_as_benefit_rules(clauses):
    """"Your compensation and benefits are attached as Annexure A" grants nothing."""
    for c in clauses:
        if "attached as Annexure A" in c.text:
            assert c.clause_type is not ClauseType.BENEFITS
        if "including, but not limited to, the compensation and benefits" in c.text:
            assert c.clause_type not in (ClauseType.BENEFITS, ClauseType.ALLOWANCES)


# --------------------------------------------------------------------------- retrieval eligibility
def test_only_eligible_clauses_are_queried(clauses, source):
    from agents.regulatory import is_eligible_for_retrieval, retrieval_ineligibility

    collector = RegulatoryEvidenceCollector(source)
    records = []
    for c in clauses:
        reason = retrieval_ineligibility(c)
        records.append(collector.skip_clause(c, reason) if reason
                       else collector.collect_for_clause(c, topic_for_clause(c)))

    eligible = [c for c in clauses if is_eligible_for_retrieval(c)]
    assert eligible, "the probation, notice, hours, rest, leave and wage clauses must be eligible"
    assert len(records) == len(clauses), "no clause may be dropped from the record"
    assert len(source.calls) == sum(len(r.queries) for r in records if r.regulatory_topic)
    queried_ids = {r.clause_id for r in records if r.regulatory_topic}
    assert queried_ids == {c.clause_id for c in eligible}


def test_weak_and_unknown_clauses_are_preserved_but_never_queried(clauses, source):
    from agents.regulatory import retrieval_ineligibility

    collector = RegulatoryEvidenceCollector(source)
    skipped = [c for c in clauses if retrieval_ineligibility(c) is not None]
    assert skipped, "this letter contains clauses that are not safe to query"
    for c in skipped:
        record = collector.skip_clause(c, retrieval_ineligibility(c))
        assert record.clause_text == c.text          # preserved verbatim
        assert record.regulatory_topic is None
        assert record.check.status == "not_run"
        assert record.check.notes                    # the reason is recorded, not hidden
    assert source.calls == [], "a skipped clause must not reach the RAG at all"


def test_the_probation_clause_is_eligible_and_the_retirement_clause_is_not(clauses):
    from agents.regulatory import is_eligible_for_retrieval

    probation = _of_type(clauses, ClauseType.PROBATION)[0]
    retirement = [c for c in clauses if "automatically retire" in c.text][0]
    assert is_eligible_for_retrieval(probation) is True
    assert is_eligible_for_retrieval(retirement) is False
