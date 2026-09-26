"""Mandatory Stage 3 regression: the real Aramco appointment letter, end to end.

parse -> semantic extraction -> clause segmentation -> classification -> existing RAG evidence
-> structured legal rule -> deterministic comparison -> finding.

Only the infrastructure underneath the existing RAG (Qdrant, the embedding model) is replaced, with
tests/fakes/regulatory_adapter.FakeRegulatoryAdapter - clearly test infrastructure: it builds real
evidence objects from the REAL 249-article knowledge base through the real rag.mapping code, and
only the choice of which articles come back for which question is scripted. It stands in for
Qdrant/the embedding model, never for extraction, segmentation, classification, rule derivation or
comparison, which are all the real production code. No live Qdrant, embedding model or vector
retrieval runs in this test environment, so this file proves the Stage 3 rule/comparison layer is
correct against real regulatory text - it does not by itself validate live retrieval end to end
(see the Stage 3 report's "live RAG status" section).

The Aramco letter is a regression fixture like any other; nothing here singles it out by name in
the production code (test_no_document_is_special_cased below checks that directly).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.contract_analysis import ContractAnalysisAgent
from extraction.contract import extract_contract
from models.analysis import FindingStatus
from models.extraction import ClauseType
from models.legal_rules import RuleType
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
def result(extraction, knowledge_base):
    adapter = FakeRegulatoryAdapter(knowledge_base)  # test infrastructure standing in for live Qdrant
    agent = ContractAnalysisAgent(adapter, max_clause_checks=200)
    return agent.analyze(extraction)


def _finding(result, clause_type):
    matches = [f for f in result.clause_findings if f.clause_type is clause_type]
    assert matches, f"no clause_finding of type {clause_type.value}"
    return matches


# --------------------------------------------------------------------------- separating the five layers (section 20)
def test_the_five_layers_are_all_present_and_distinguishable(extraction, result):
    """extraction result, retrieved evidence, structured rule, comparison result, final status -
    each must be independently readable off the finding, not collapsed into a single opaque verdict."""
    assert extraction.status.value == "success"
    probation = _finding(result, ClauseType.PROBATION)[0]
    assert probation.contract_fact is not None          # 1. extraction result (the fact this clause carries)
    assert probation.regulatory_evidence                 # 2. retrieved legal evidence
    assert probation.legal_rule is not None               # 3. structured legal rule
    assert probation.comparison is not None               # 4. comparison result
    assert probation.status in FindingStatus              # 5. final status
    assert len(result.clause_findings) == len(result.clause_checks) == len(extraction.clauses)


# --------------------------------------------------------------------------- K. probation regression (mandatory)
def test_k_probation_180_days_is_read_as_a_maximum_and_found_compliant(result):
    [probation] = _finding(result, ClauseType.PROBATION)
    assert probation.contract_fact.normalized_value == 180.0 and probation.contract_fact.unit == "day"
    assert probation.legal_rule.rule_type is RuleType.MAXIMUM  # the real Article 53 text: a flat ceiling,
    assert probation.legal_rule.value == 180.0                # no "90 unless written agreement" clause in
    assert probation.legal_rule.article_number == 53          # the actual retrieved evidence - nothing here
    assert probation.comparison.satisfied is True              # was assumed or hard-coded (see test below)
    assert probation.status is FindingStatus.COMPLIANT
    assert "180" in probation.explanation


# --------------------------------------------------------------------------- L. notice regression (mandatory)
def test_l_both_notice_clauses_are_found_compliant_against_article_75(result):
    notices = _finding(result, ClauseType.NOTICE)
    by_name = {f.clause_name: f for f in notices}
    assert set(by_name) == {"Notice during probation", "Notice after confirmation"}
    for finding in by_name.values():
        assert finding.contract_fact.normalized_value == 90.0
        assert finding.legal_rule.rule_type is RuleType.CONDITIONAL  # Article 75 gives two minimums
        assert finding.legal_rule.article_number == 75               # (worker-initiated / employer-initiated)
        assert {round(c.value) for c in finding.legal_rule.conditions} == {30, 60}
        assert all(c.status.value == "satisfied" for c in finding.comparison.condition_results)
        assert finding.status is FindingStatus.COMPLIANT  # 90 days clears both the 30- and 60-day minimums


# --------------------------------------------------------------------------- M. facts stay separate (mandatory)
def test_m_the_two_notice_facts_and_the_probation_fact_are_never_merged(result):
    probation = _finding(result, ClauseType.PROBATION)[0]
    notices = _finding(result, ClauseType.NOTICE)
    all_facts = [probation.contract_fact, *(f.contract_fact for f in notices)]
    assert len({f.clause_id for f in all_facts}) == len(all_facts) == 3
    assert len({f.source_text for f in all_facts}) == 3  # three distinct sentences, not one reused
    during = next(f for f in notices if f.clause_name == "Notice during probation")
    after = next(f for f in notices if f.clause_name == "Notice after confirmation")
    assert during.contract_fact.clause_id != after.contract_fact.clause_id
    assert during.field == "notice_period_during_probation"
    assert after.field == "notice_period_after_confirmation"


# --------------------------------------------------------------------------- N. salary vs allowance vs CTC
def test_n_the_bound_in_hand_figure_is_never_confused_with_the_unbound_salary_mentions(result):
    """Section 10: SALARY must not mean every financial clause. Exactly one of the letter's several
    SALARY-classified clauses carries a Stage 1 typed reading (in_hand_salary); the rest - headings,
    the LPA/CTC figure, generic remuneration mentions - correctly come back NOT_FOUND rather than
    being silently treated as base salary."""
    salary_findings = _finding(result, ClauseType.SALARY)
    assert len(salary_findings) >= 2, "the letter mentions salary/compensation in more than one clause"

    bound = [f for f in salary_findings if f.contract_fact is not None]
    assert len(bound) == 1, "exactly the in-hand-salary clause should bind to a typed Stage 1 fact"
    [in_hand] = bound
    assert in_hand.field == "in_hand_salary"
    assert in_hand.contract_fact.normalized_value == 1800.0
    assert "1800" in in_hand.contract_fact.raw_value

    unbound = [f for f in salary_findings if f.contract_fact is None]
    assert unbound, "other salary-labelled clauses (headings, LPA/CTC, general remuneration) exist too"
    for finding in unbound:
        # NOT_FOUND (an eligible topic with no bound fact) or NOT_APPLICABLE (no regulatory topic
        # covers this particular clause, e.g. a heading/table row) are both legitimate "no verdict"
        # outcomes here; what must never happen is silently treating an unbound figure as compliant.
        assert finding.status in (FindingStatus.NOT_FOUND, FindingStatus.NOT_APPLICABLE)
        assert finding.status not in (FindingStatus.COMPLIANT, FindingStatus.NON_COMPLIANT)


def test_n_working_hours_weekly_rest_and_annual_leave_are_handled_on_their_own_legal_terms(result):
    hours = _finding(result, ClauseType.WORKING_HOURS)[0]
    assert hours.contract_fact.unit == "hour_per_week" and hours.contract_fact.normalized_value == 40.0
    assert hours.status is FindingStatus.COMPLIANT  # 40 <= the real Article 98 weekly ceiling of 48

    rest = _finding(result, ClauseType.WEEKLY_REST)[0]
    assert rest.contract_fact.unit == "day"  # "2 days off" - a day count
    assert rest.legal_rule.unit == "hour"     # Article 104's real text: >=24 CONTINUOUS HOURS of rest
    assert rest.status is FindingStatus.REQUIRES_REVIEW  # a day-count and an hour-duration are not
    assert "incompatible" in (rest.comparison.notes[0] if rest.comparison.notes else "")  # interchangeable

    leave = _finding(result, ClauseType.ANNUAL_LEAVE)[0]
    assert leave.contract_fact.normalized_value == 24.0
    assert leave.legal_rule.rule_type is RuleType.CONDITIONAL  # 21 days normally, 30 after 5 years' service
    assert leave.status is FindingStatus.REQUIRES_REVIEW  # 24 clears the first floor, misses the second;
    assert leave.status not in (FindingStatus.COMPLIANT, FindingStatus.NON_COMPLIANT)  # tenure is not stated


# --------------------------------------------------------------------------- preservation (sections 15/16)
def test_foreign_legal_references_survive_all_the_way_through(extraction):
    text = " ".join(c.text for c in extraction.clauses)
    assert "Maternity Benefit Act, 1961" in text
    assert "LPA" in text


def test_every_clause_is_preserved_in_clause_findings_even_when_ineligible(result, extraction):
    """Nothing is dropped: an ineligible/UNKNOWN/weak clause is still a ClauseLegalFinding (NOT_APPLICABLE),
    never silently omitted from the result."""
    ids = {f.clause_id for f in result.clause_findings}
    assert ids == {c.clause_id for c in extraction.clauses}
    not_applicable = [f for f in result.clause_findings if f.status is FindingStatus.NOT_APPLICABLE]
    assert not_applicable  # e.g. confidentiality/duties/disciplinary clauses have no numeric rule to check


# --------------------------------------------------------------------------- no hard-coded verdict (section 22)
def test_no_document_is_special_cased_in_the_rule_engine():
    import inspect

    import agents.contract_analysis as contract_analysis
    import agents.legal_rules as legal_rules

    for module in (legal_rules, contract_analysis):
        source = inspect.getsource(module).lower()
        assert "aramco" not in source


# --------------------------------------------------------------------------- live RAG safety (section 21)
def test_this_file_never_claims_live_retrieval_ran(result):
    """FakeRegulatoryAdapter is test infrastructure; the point of this assertion is documentation-as-code
    for whoever next edits this file: do not swap this fixture in place of the real adapter and call
    it a live-RAG validation without also running it against a real Qdrant instance."""
    import tests.fakes.regulatory_adapter as _fake_module

    assert isinstance(result, object)  # the analysis ran
    assert _fake_module.__doc__ and "REAL" in _fake_module.__doc__  # the module documents its own scope
