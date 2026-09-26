"""Stage 3: the deterministic legal-rule and comparison engine (agents/legal_rules.py).

Tests A-J exercise the comparison arithmetic directly (compare_fact_to_rule / convert) with
hand-built rules: they do not need a real document. K/L/M and the full pipeline are covered by
tests/extraction/test_aramco_legal_findings.py, which is the mandatory end-to-end regression;
N (salary vs allowance vs CTC) is also covered there against the real document, since that is the
scenario Sanad actually needs to get right. O-S here test the finding-builder's status semantics,
evidence traceability, and that an LLM explainer can never move the deterministic status.
"""

from __future__ import annotations

import pytest

from agents.legal_rules import (
    build_clause_legal_finding,
    compare_fact_to_rule,
    convert,
    derive_legal_rule,
)
from models.analysis import ClauseCheck, ClauseLegalFinding, EvidenceReference, FindingStatus, RegulatoryCheck
from models.extraction import ClauseType, ClauseValue, ExtractionMethodName, FieldStatus, SourceSpan
from models.legal_rules import Comparator, ConditionStatus, ContractFact, LegalRule, RuleCondition, RuleType
from models.regulatory import ArticleReference, RegulatoryEvidence, SourceDocument


# --------------------------------------------------------------------------- helpers
def clause(text: str, clause_type: ClauseType, name: str = "Clause", clause_id: str = "C01") -> ClauseValue:
    return ClauseValue(
        clause_id=clause_id, clause_name=name, clause_type=clause_type, text=text, confidence="high",
        source_span=SourceSpan(text=text, page_number=1, section_id="s01", method=ExtractionMethodName.CLAUSE_SENTENCE),
    )


def fact(value: float, unit: str, *, clause_id: str = "C01", raw: str | None = None, field: str = "x") -> ContractFact:
    raw = raw or f"{value:g} {unit}"
    return ContractFact(clause_id=clause_id, field=field, raw_value=raw, normalized_value=value, unit=unit,
                        source_text=raw, page_number=1, confidence="high")


def rule(rule_type: RuleType, comparator: Comparator, *, value=None, value_max=None, unit=None,
        conditions=(), **kw) -> LegalRule:
    return LegalRule(rule_id="R01", topic="t", article_citation="Article X", article_number=1, rule_type=rule_type,
                     comparator=comparator, value=value, value_max=value_max, unit=unit,
                     conditions=list(conditions), source_evidence_id="kb-1", **kw)


def evidence_from_text(kb_index: int, article_number: int, arabic_text: str) -> EvidenceReference:
    source = SourceDocument(title_ar="x", title_en="x", publisher="x", knowledge_base_file="x")
    item = RegulatoryEvidence(
        reference=ArticleReference(kb_index=kb_index, article_number=article_number),
        citation=f"Article {article_number}", arabic_content=arabic_text, rank=1, score=0.8,
        score_is_rounded=False, source=source, metadata_complete=True, raw_metadata={})
    return EvidenceReference(evidence_id=f"kb-{kb_index}", evidence=item)


def evidence_from_kb(kb_index: int, article_number: int, knowledge_base) -> EvidenceReference:
    return evidence_from_text(kb_index, article_number, knowledge_base[kb_index - 1]["arabic_content"])


def evidence_found_check(clause_id: str, clause_type: ClauseType, clause_text: str, topic: str,
                         evidence: EvidenceReference) -> ClauseCheck:
    return ClauseCheck(
        clause_id=clause_id, clause_type=clause_type, clause_name="x", clause_text=clause_text,
        regulatory_topic=topic, queries=["q"],
        check=RegulatoryCheck(topic=topic, triggered_by_fields=[], questions=["q"], status="evidence_found",
                              relevant_evidence_ids=[evidence.evidence_id], retrieved_evidence_ids=[evidence.evidence_id]),
        evidence=[evidence],
    )


# --------------------------------------------------------------------------- A. MINIMUM
def test_a_minimum_comparison_passes_at_and_above_the_floor():
    r = rule(RuleType.MINIMUM, Comparator.GE, value=21, unit="day")
    assert compare_fact_to_rule(fact(21, "day"), r).satisfied is True  # exactly at the minimum: allowed
    assert compare_fact_to_rule(fact(30, "day"), r).satisfied is True  # legally may be MORE than the minimum


def test_a_minimum_comparison_fails_below_the_floor():
    r = rule(RuleType.MINIMUM, Comparator.GE, value=60, unit="day")
    result = compare_fact_to_rule(fact(45, "day"), r)
    assert result.satisfied is False
    assert "falls short" in result.explanation


# --------------------------------------------------------------------------- B. MAXIMUM
def test_a_maximum_comparison_passes_at_and_below_the_ceiling():
    r = rule(RuleType.MAXIMUM, Comparator.LE, value=180, unit="day")
    assert compare_fact_to_rule(fact(180, "day"), r).satisfied is True  # AT the maximum is still allowed
    assert compare_fact_to_rule(fact(90, "day"), r).satisfied is True  # legally may be LESS than the maximum


def test_a_maximum_comparison_fails_above_the_ceiling():
    r = rule(RuleType.MAXIMUM, Comparator.LE, value=48, unit="hour_per_week")
    result = compare_fact_to_rule(fact(55, "hour_per_week"), r)
    assert result.satisfied is False
    assert "exceeds" in result.explanation


# --------------------------------------------------------------------------- C. EXACT
def test_an_exact_comparison_requires_the_precise_value():
    r = rule(RuleType.EXACT, Comparator.EQ, value=24, unit="hour")
    assert compare_fact_to_rule(fact(24, "hour"), r).satisfied is True
    assert compare_fact_to_rule(fact(23, "hour"), r).satisfied is False
    assert compare_fact_to_rule(fact(25, "hour"), r).satisfied is False


# --------------------------------------------------------------------------- D. RANGE
def test_a_range_comparison_checks_both_bounds():
    r = rule(RuleType.RANGE, Comparator.RANGE, value=10, value_max=20, unit="day")
    assert compare_fact_to_rule(fact(10, "day"), r).satisfied is True
    assert compare_fact_to_rule(fact(20, "day"), r).satisfied is True
    assert compare_fact_to_rule(fact(9, "day"), r).satisfied is False
    assert compare_fact_to_rule(fact(21, "day"), r).satisfied is False


def test_a_range_rule_cannot_be_built_with_its_bounds_reversed():
    with pytest.raises(ValueError):
        rule(RuleType.RANGE, Comparator.RANGE, value=20, value_max=10, unit="day")


# --------------------------------------------------------------------------- E. CONDITIONAL structure
def test_a_conditional_rule_evaluates_every_condition_independently():
    r = rule(RuleType.CONDITIONAL, Comparator.CONDITIONAL, conditions=[
        RuleCondition(description="worker-initiated minimum", value=30, unit="day", comparator=Comparator.GE),
        RuleCondition(description="employer-initiated minimum", value=60, unit="day", comparator=Comparator.GE),
    ])
    result = compare_fact_to_rule(fact(90, "day"), r)
    assert [c.status for c in result.condition_results] == [ConditionStatus.SATISFIED, ConditionStatus.SATISFIED]
    assert result.satisfied is True  # every applicable sub-threshold held


def test_a_conditional_rule_must_carry_at_least_one_condition():
    with pytest.raises(ValueError):
        rule(RuleType.CONDITIONAL, Comparator.CONDITIONAL, conditions=[])


# --------------------------------------------------------------------------- F/G/H. condition states
def test_f_a_textual_condition_is_satisfied_when_the_clause_says_so():
    r = rule(RuleType.CONDITIONAL, Comparator.CONDITIONAL,
            conditions=[RuleCondition(description="requires the worker's written agreement")])
    f = fact(180, "day", raw="180 days, extended with the written agreement of the employee")
    result = compare_fact_to_rule(f, r)
    assert result.condition_results[0].status is ConditionStatus.SATISFIED


def test_g_a_textual_condition_is_not_stated_when_the_clause_is_silent():
    r = rule(RuleType.CONDITIONAL, Comparator.CONDITIONAL,
            conditions=[RuleCondition(description="requires the worker's written agreement")])
    result = compare_fact_to_rule(fact(180, "day"), r)
    assert result.condition_results[0].status is ConditionStatus.NOT_STATED
    assert result.satisfied is None  # "not stated" must never present itself as a clean pass or fail


def test_h_a_numeric_sub_threshold_can_be_clearly_not_satisfied():
    r = rule(RuleType.CONDITIONAL, Comparator.CONDITIONAL,
            conditions=[RuleCondition(description="minimum after 5 years' service", value=30, unit="day",
                                      comparator=Comparator.GE)])
    result = compare_fact_to_rule(fact(24, "day"), r)
    assert result.condition_results[0].status is ConditionStatus.NOT_SATISFIED
    assert result.satisfied is False


def test_a_conditional_rule_mixing_satisfied_and_not_satisfied_reports_neither():
    """Real example: Article 109's leave minimum is 21 days normally, 30 after 5 years' service. A
    24-day contract clears the first and misses the second - compare_fact_to_rule must not pick a
    side; _status_from_comparison (exercised via build_clause_legal_finding) turns this mix into
    REQUIRES_REVIEW, never COMPLIANT or NON_COMPLIANT (tests below)."""
    r = rule(RuleType.CONDITIONAL, Comparator.CONDITIONAL, conditions=[
        RuleCondition(description="normal minimum", value=21, unit="day", comparator=Comparator.GE),
        RuleCondition(description="minimum after 5 years", value=30, unit="day", comparator=Comparator.GE),
    ])
    result = compare_fact_to_rule(fact(24, "day"), r)
    assert [c.status for c in result.condition_results] == [ConditionStatus.SATISFIED, ConditionStatus.NOT_SATISFIED]
    assert result.satisfied is None


# --------------------------------------------------------------------------- I/J. units
def test_i_unit_normalization_converts_compatible_units():
    assert convert(14, "day", "week") == 2
    assert convert(1, "year", "month") == 12
    assert convert(90, "day", "day") == 90


def test_j_incompatible_units_are_never_silently_reconciled():
    """A day-count and an hour-count are not the same kind of quantity - see the module's own
    comment on why hour_per_day/hour_per_week are not converted into each other either."""
    r = rule(RuleType.MINIMUM, Comparator.GE, value=24, unit="hour")
    result = compare_fact_to_rule(fact(2, "day"), r)
    assert result.satisfied is None
    assert "incompatible units" in result.notes
    assert convert(40, "hour_per_week", "hour_per_day") is None


# --------------------------------------------------------------------------- rule extraction never invents a value
def test_an_unrecognised_article_becomes_unknown_not_a_guess():
    evidence = evidence_from_text(1, 1, "يحدد صاحب العمل نموذج التوقيع المعتمد لدى المنشأة.")
    r = derive_legal_rule("R1", "other", evidence)
    assert r.rule_type is RuleType.UNKNOWN
    assert r.value is None and r.comparator is Comparator.UNKNOWN
    assert r.confidence == "low"


def test_a_legal_rule_cannot_carry_an_invented_value_on_an_unknown_type():
    with pytest.raises(ValueError):
        LegalRule(rule_id="R1", topic="t", rule_type=RuleType.UNKNOWN, comparator=Comparator.UNKNOWN, value=42)


# --------------------------------------------------------------------------- O. NOT_FOUND
def test_o_not_found_means_no_contract_fact_was_available_not_noncompliance(make_contract, knowledge_base):
    _, contract = make_contract(["This paragraph states nothing the Stage 1 extractor recognises."])
    item = clause("The company may adjust total compensation at its discretion from time to time.",
                 ClauseType.SALARY, "Salary")
    check = evidence_found_check("C01", ClauseType.SALARY, item.text, "wage", evidence_from_kb(92, 90, knowledge_base))

    finding = build_clause_legal_finding(1, item, check, contract)

    assert finding.status is FindingStatus.NOT_FOUND
    assert finding.contract_fact is None
    assert finding.status not in (FindingStatus.COMPLIANT, FindingStatus.NON_COMPLIANT)


# --------------------------------------------------------------------------- P. AMBIGUOUS
PROBATION_180 = "The initial period of probation is 180 days from your date of joining."
PROBATION_90 = "The probation period shall be 90 days for all administrative staff."


def test_p_ambiguous_means_the_contracts_own_wording_conflicts(make_contract, knowledge_base):
    _, contract = make_contract([f"{PROBATION_180} {PROBATION_90}"])
    assert contract.probation_period.status is FieldStatus.AMBIGUOUS  # Stage 1's own contradiction case

    # Stage 2 never actually merges two classified sentences into one clause (each starts its own
    # group - see extraction/clauses.py), so this hand-built clause is a deliberate synthetic case:
    # it proves the AMBIGUOUS mechanism itself works, independent of whether the real segmenter ever
    # produces such a clause.
    combined = f"{PROBATION_180} {PROBATION_90}"
    item = clause(combined, ClauseType.PROBATION, "Probation")
    check = evidence_found_check("C01", ClauseType.PROBATION, combined, "probation", evidence_from_kb(54, 53, knowledge_base))

    finding = build_clause_legal_finding(1, item, check, contract)

    assert finding.status is FindingStatus.AMBIGUOUS
    assert finding.contract_fact is None  # no single reading is asserted when the clause itself conflicts
    assert finding.status not in (FindingStatus.COMPLIANT, FindingStatus.NON_COMPLIANT)


# --------------------------------------------------------------------------- Q. REQUIRES_REVIEW
def test_q_requires_review_when_the_evidence_could_not_be_reduced_to_a_rule(make_contract, knowledge_base):
    """build_clause_legal_finding trusts the ClauseCheck it is given (eligibility was already decided
    upstream by agents.regulatory.retrieval_ineligibility); here the evidence itself is simply too
    unstructured to become a deterministic rule."""
    _, contract = make_contract(["The initial period of probation is 180 days from your date of joining."])
    item = [c for c in contract.clauses if c.clause_type is ClauseType.PROBATION][0]
    evidence = evidence_from_text(2, 2, "يحدد صاحب العمل نموذج التوقيع المعتمد لدى المنشأة.")
    check = evidence_found_check(item.clause_id, ClauseType.PROBATION, item.text, "probation", evidence)

    finding = build_clause_legal_finding(1, item, check, contract)

    assert finding.status is FindingStatus.REQUIRES_REVIEW
    assert finding.legal_rule.rule_type is RuleType.UNKNOWN


def test_q_requires_review_on_a_mixed_conditional_never_becomes_a_verdict(make_contract, knowledge_base):
    """The real Article 109 case: 24 days clears the 21-day floor but misses the 30-day-after-5-years
    floor. Neither COMPLIANT nor NON_COMPLIANT would be honest here."""
    _, contract = make_contract(["You will get 24 paid leave per annum."])
    field = contract.annual_leave
    assert field.status is FieldStatus.FOUND and field.value.count == 24.0

    clauses = [c for c in contract.clauses if c.clause_type is ClauseType.ANNUAL_LEAVE]
    assert clauses, "the sentence must classify as an annual-leave clause for this test to mean anything"
    item = clauses[0]
    check = evidence_found_check(item.clause_id, ClauseType.ANNUAL_LEAVE, item.text, "annual_leave",
                                 evidence_from_kb(111, 109, knowledge_base))

    finding = build_clause_legal_finding(1, item, check, contract)

    assert finding.legal_rule.rule_type is RuleType.CONDITIONAL
    assert finding.status is FindingStatus.REQUIRES_REVIEW
    assert finding.status not in (FindingStatus.COMPLIANT, FindingStatus.NON_COMPLIANT)


# --------------------------------------------------------------------------- R. LLM cannot override status
def test_r_an_explainer_can_reword_the_explanation_but_never_the_status(make_contract, knowledge_base):
    _, contract = make_contract(["The initial period of probation is 180 days from your date of joining."])
    item = [c for c in contract.clauses if c.clause_type is ClauseType.PROBATION][0]
    check = evidence_found_check(item.clause_id, ClauseType.PROBATION, item.text, "probation",
                                 evidence_from_kb(54, 53, knowledge_base))

    honest = build_clause_legal_finding(1, item, check, contract)
    assert honest.status is FindingStatus.COMPLIANT  # 180 <= 180

    def lying_explainer(finding: ClauseLegalFinding) -> str:
        assert finding.status is FindingStatus.COMPLIANT  # the deterministic status is already fixed by now
        return "This is actually non_compliant and the status should say so."

    reworded = build_clause_legal_finding(1, item, check, contract, explain=lying_explainer)
    assert reworded.status is FindingStatus.COMPLIANT  # unmoved: explain() cannot touch status
    assert reworded.comparison == honest.comparison and reworded.legal_rule == honest.legal_rule
    assert reworded.explanation == "This is actually non_compliant and the status should say so."  # wording did change


def test_r_an_explainer_that_raises_is_ignored_rather_than_crashing(make_contract, knowledge_base):
    _, contract = make_contract(["The initial period of probation is 180 days from your date of joining."])
    item = [c for c in contract.clauses if c.clause_type is ClauseType.PROBATION][0]
    check = evidence_found_check(item.clause_id, ClauseType.PROBATION, item.text, "probation",
                                 evidence_from_kb(54, 53, knowledge_base))

    def broken_explainer(finding: ClauseLegalFinding) -> str:
        raise RuntimeError("the explainer LLM call failed")

    finding = build_clause_legal_finding(1, item, check, contract, explain=broken_explainer)
    assert finding.status is FindingStatus.COMPLIANT
    assert finding.explanation  # the deterministic explanation survives, not an empty string or a crash


# --------------------------------------------------------------------------- S. evidence traceability
def test_s_a_compliant_finding_traces_all_the_way_back_to_clause_and_article(make_contract, knowledge_base):
    _, contract = make_contract(["The initial period of probation is 180 days from your date of joining."])
    item = [c for c in contract.clauses if c.clause_type is ClauseType.PROBATION][0]
    evidence = evidence_from_kb(54, 53, knowledge_base)
    check = evidence_found_check(item.clause_id, ClauseType.PROBATION, item.text, "probation", evidence)

    finding = build_clause_legal_finding(7, item, check, contract)

    assert finding.finding_id == "L07"
    assert finding.clause_id == item.clause_id
    assert finding.clause_text == item.text
    assert finding.contract_fact is not None and finding.contract_fact.clause_id == item.clause_id
    assert finding.contract_fact.normalized_value == 180.0
    assert finding.legal_rule is not None and finding.legal_rule.article_number == 53
    assert finding.legal_rule.source_evidence_id == evidence.evidence_id
    assert finding.regulatory_evidence and finding.regulatory_evidence[0].evidence_id == evidence.evidence_id
    assert finding.comparison is not None and finding.comparison.satisfied is True
    assert finding.status is FindingStatus.COMPLIANT
    assert finding.explanation
    assert finding.confidence == item.confidence
    # every one of these can be read back off the single ClauseLegalFinding object - nothing needs
    # to be re-derived or re-queried to answer "why did Sanad produce this result?"


# --------------------------------------------------------------------------- T. NON_COMPLIANT
def test_t_a_maximum_violation_is_found_non_compliant_against_the_real_article(make_contract, knowledge_base):
    """Stage 4 quality gate needs at least one clause-level NON_COMPLIANT case exercised through the
    real finding-builder (none of the five real contract fixtures happens to violate a hard maximum),
    so this synthetic clause is built the same way as test_s above but with a value the real Article
    53 text (180-day ceiling) actually rejects."""
    _, contract = make_contract(["The initial period of probation is 200 days from your date of joining."])
    item = [c for c in contract.clauses if c.clause_type is ClauseType.PROBATION][0]
    evidence = evidence_from_kb(54, 53, knowledge_base)
    check = evidence_found_check(item.clause_id, ClauseType.PROBATION, item.text, "probation", evidence)

    finding = build_clause_legal_finding(1, item, check, contract)

    assert finding.contract_fact is not None and finding.contract_fact.normalized_value == 200.0
    assert finding.legal_rule.rule_type is RuleType.MAXIMUM and finding.legal_rule.value == 180.0
    assert finding.comparison is not None and finding.comparison.satisfied is False
    assert finding.status is FindingStatus.NON_COMPLIANT
    assert "180" in finding.explanation
    # the deterministic origin is the same object graph as the COMPLIANT case above - only the
    # arithmetic result differs, never an LLM's opinion
    assert finding.legal_rule.extraction_method == "pattern"
