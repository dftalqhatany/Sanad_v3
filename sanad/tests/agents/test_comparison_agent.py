"""Comparison & Recommendation Agent: every contract goes through the Analysis Agent once, then structured comparison."""

from __future__ import annotations

import logging

import pytest

from sanad.agents import AnalysisAgent, ContractAnalysisAgent, ContractComparisonAgent, LLMEvidenceInterpreter
from sanad.agents.comparison_dimensions import duration, monthly_money
from sanad.config import AnalysisSettings, SanadSettings
from sanad.extraction import extract_contract
from sanad.models.analysis import AnalysisStatus, SalaryBenchmark, SalaryObservation, SalarySource
from sanad.models.comparison import (
    ContractComparisonResult,
    ContractValue,
    DecisionFactor,
    DimensionComparison,
    Recommendation,
)
from sanad.models.extraction import FieldStatus
from tests.fakes.llm import FakeLLMClient
from tests.fakes.regulatory_adapter import KB_PROBATION_ART_53, FakeRegulatoryAdapter

OFFER_A = ["Job Title: Data Analyst", "Contract Type: Fixed-term", "Basic Salary: 12,000 SAR per month",
           "Housing Allowance: 3,000 SAR per month", "Transportation Allowance: 800 SAR per month",
           "Working Hours: 8 hours per day, 40 hours per week", "Working Days: Sunday to Thursday",
           "Annual Leave: 30 days", "Probation Period: 90 days", "Notice Period: 60 days", "Work Location: Riyadh"]
OFFER_B = ["Job Title: Data Analyst", "Contract Type: Fixed-term", "Basic Salary: 10,000 SAR per month",
           "Housing Allowance: 2,500 SAR per month", "Transportation Allowance: 500 SAR per month",
           "Working Hours: 9 hours per day, 45 hours per week", "Working Days: Sunday to Thursday",
           "Annual Leave: 21 days", "Probation Period: 180 days", "Notice Period: 30 days", "Work Location: Jeddah"]
OFFER_C = ["Job Title: Data Analyst", "Contract Type: Fixed-term", "Basic Salary: 15,000 SAR per month",
           "Housing Allowance: 3,750 SAR per month", "Transportation Allowance: 1,000 SAR per month",
           "Working Hours: 8 hours per day, 48 hours per week", "Working Days: 6 days per week",
           "Annual Leave: 30 days", "Probation Period: 90 days", "Notice Period: 60 days", "Work Location: Dammam"]
PROBATION_QUOTE_AR = "على ألا يزيد مجموع المدة في جميع الأحوال على مائة وثمانين يوماً"


class SpyAnalysisAgent(AnalysisAgent):
    def __init__(self, contract_agent):
        super().__init__(contract_agent)
        self.calls = []

    def analyze(self, **kwargs):
        self.calls.append(kwargs)
        return super().analyze(**kwargs)


@pytest.fixture
def offers(make_contract):
    built = {}
    for name, lines in (("a", OFFER_A), ("b", OFFER_B), ("c", OFFER_C)):
        document, contract = make_contract(lines)
        built[name] = (document, contract.model_copy(update={"filename": f"offer_{name}.docx",
                                                               "document_id": f"doc_offer_{name}"}))
    return built


@pytest.fixture
def spy(fake_rag):
    return SpyAnalysisAgent(ContractAnalysisAgent(fake_rag))


def agent_for(spy_agent, **kwargs) -> ContractComparisonAgent:
    return ContractComparisonAgent(spy_agent, **kwargs)


def values_by_contract(dimension: DimensionComparison) -> dict[str, ContractValue]:
    return {v.contract_id: v for v in dimension.values}


# --------------------------------------------------------------------------- workflow
def test_two_contracts_are_each_analysed_once_by_the_analysis_agent(offers, spy, fake_rag):
    contracts = [offers["a"][1], offers["b"][1]]
    result = agent_for(spy).compare(contracts)

    assert [call["contract"] for call in spy.calls] == contracts  # once per contract, in order
    assert all(call["cv"] is None and call["use_contract_job_title"] is False for call in spy.calls)
    assert [c.contract_id for c in result.contracts] == ["contract_1", "contract_2"]
    assert [c.label for c in result.contracts] == ["offer_a.docx", "offer_b.docx"]
    assert [c.contract_analysis.document_id for c in result.contracts] == ["doc_offer_a", "doc_offer_b"]
    per_contract_questions = sum(len(check.questions) for check in result.contracts[0].contract_analysis.regulatory_checks)
    assert len(fake_rag.calls) == per_contract_questions * 2  # each contract's own regulatory analysis
    assert result.status is AnalysisStatus.SUCCESS
    assert ContractComparisonResult.model_validate_json(result.model_dump_json()).model_dump() == result.model_dump()


def test_two_contracts_dominance_recommendation_is_explained_with_evidence(offers, spy):
    documents = {"contract_1": offers["a"][0], "contract_2": offers["b"][0]}
    result = agent_for(spy).compare([offers["a"][1], offers["b"][1]])
    recommendation = result.recommendation

    assert recommendation.status == "preferred_contract" and recommendation.method == "dominance"
    assert recommendation.preferred_contract_id == "contract_1"
    assert {f.dimension for f in recommendation.decision_factors} == {
        "basic_salary", "stated_pay", "weekly_working_hours", "daily_working_hours", "annual_leave", "probation_period"}
    for factor in recommendation.decision_factors:
        assert factor.favours == ["contract_1"]
        assert {v.contract_id for v in factor.evidence} == {"contract_1", "contract_2"}
        for value in factor.evidence:
            assert value.finding_ids and value.fields_used and value.source_texts
            assert all(text in documents[value.contract_id].full_text for text in value.source_texts)
    assert "no language model" in recommendation.basis
    assert "Working days per week" in recommendation.explanation  # equal dimensions are named, not hidden
    assert any("market" in caveat for caveat in recommendation.caveats)


def test_salary_values_and_stated_pay_composition(offers, spy):
    result = agent_for(spy).compare([offers["a"][1], offers["b"][1], offers["c"][1]])
    basic = values_by_contract(result.dimension("basic_salary"))
    assert [basic[c].value for c in ("contract_1", "contract_2", "contract_3")] == [12000, 10000, 15000]
    assert basic["contract_1"].unit == "SAR/month" and basic["contract_1"].raw_values == ["12,000 SAR"]
    pay = values_by_contract(result.dimension("stated_pay"))
    assert pay["contract_1"].value == 15800
    assert pay["contract_1"].display == "15,800 SAR/month (basic 12,000 + housing allowance 3,000 + transportation allowance 800)"
    assert set(pay["contract_1"].fields_used) == {"salary", "housing_allowance", "transportation_allowance"}
    assert any("Not found in the contract, so not included: other allowances" in n for n in pay["contract_1"].notes)
    assert result.dimension("stated_pay").best_contract_ids == ["contract_3"]
    market = result.dimension("salary_vs_market")
    assert market.ranking == "not_ranked" and {v.status for v in market.values} == {"not_available"}


def test_three_contracts_with_trade_offs_have_no_clear_preference(offers, spy):
    result = agent_for(spy).compare([offers["a"][1], offers["b"][1], offers["c"][1]])
    recommendation = result.recommendation

    assert len(spy.calls) == 3 and len(result.contracts) == 3
    assert recommendation.status == "no_clear_preference" and recommendation.preferred_contract_id is None
    assert result.dimension("basic_salary").best_contract_ids == ["contract_3"]
    assert result.dimension("weekly_working_hours").best_contract_ids == ["contract_1"]
    assert result.dimension("working_days_per_week").best_contract_ids == ["contract_1", "contract_2"]
    assert result.dimension("annual_leave").best_contract_ids == ["contract_1", "contract_3"]
    assert any(t.startswith("Basic salary: contract_3 leads") for t in recommendation.trade_offs)
    assert any(t.startswith("Weekly working hours: contract_1 leads") for t in recommendation.trade_offs)
    separating = [d for d in result.comparison.all() if d.ranking != "not_ranked" and d.comparable and not d.all_equal]
    assert all(d.best_contract_ids != ["contract_2"] for d in separating)  # the weakest offer never leads alone
    assert "trade-offs" in recommendation.explanation


def test_priorities_are_applied_in_order_and_explained(offers, spy):
    contracts = [offers["a"][1], offers["b"][1], offers["c"][1]]
    by_pay = agent_for(spy).compare(contracts, priorities=["stated_pay"]).recommendation
    assert by_pay.status == "preferred_contract" and by_pay.method == "priorities"
    assert by_pay.preferred_contract_id == "contract_3" and by_pay.priorities_used == ["stated_pay"]
    assert [f.dimension for f in by_pay.decision_factors] == ["stated_pay"]

    by_leave_then_hours = agent_for(spy).compare(contracts, priorities=["annual_leave", "weekly_working_hours"])
    recommendation = by_leave_then_hours.recommendation
    assert recommendation.preferred_contract_id == "contract_1"
    assert [(f.dimension, f.favours) for f in recommendation.decision_factors] == [
        ("annual_leave", ["contract_1", "contract_3"]), ("weekly_working_hours", ["contract_1"])]
    assert {v.contract_id for v in recommendation.decision_factors[1].evidence} == {"contract_1", "contract_3"}


def test_priority_that_cannot_be_compared_is_skipped_explicitly(offers, make_contract, spy):
    _, no_leave = make_contract([line for line in OFFER_B if not line.startswith("Annual Leave")])
    result = agent_for(spy).compare([offers["a"][1], no_leave], priorities=["annual_leave"])
    recommendation = result.recommendation
    assert recommendation.status == "no_clear_preference"
    assert "None of the given priorities could be compared" in recommendation.explanation
    assert any("'Annual leave' was skipped" in c for c in recommendation.caveats)


# --------------------------------------------------------------------------- missing / incomparable data
def test_missing_terms_are_reported_not_inferred(offers, make_contract, spy):
    lines = [line for line in OFFER_B if not line.startswith(("Basic Salary", "Annual Leave"))]
    _, incomplete = make_contract(lines)
    result = agent_for(spy).compare([offers["a"][1], incomplete])

    basic = result.dimension("basic_salary")
    assert not basic.comparable and basic.best_contract_ids == []
    missing = values_by_contract(basic)["contract_2"]
    assert missing.status == "not_found" and missing.value is None and missing.finding_ids
    assert "was not found in the contract" in basic.explanation
    assert values_by_contract(result.dimension("stated_pay"))["contract_2"].status == "not_found"
    assert values_by_contract(result.dimension("annual_leave"))["contract_2"].status == "not_found"
    risks = {(r.contract_id, r.code, r.field) for r in result.risks}
    assert ("contract_2", "key_term_not_found", "salary") in risks
    assert ("contract_2", "key_term_not_found", "annual_leave") in risks
    assert all("not a finding of non-compliance" in r.message for r in result.risks if r.code == "key_term_not_found")
    assert any(c.startswith("Basic salary could not be compared") for c in result.recommendation.caveats)
    assert result.recommendation.status == "preferred_contract"  # still decided on the comparable dimensions
    assert {f.dimension for f in result.recommendation.decision_factors}.isdisjoint({"basic_salary", "annual_leave"})


def test_ambiguous_terms_are_not_compared_and_flagged(offers, make_contract, spy):
    _, ambiguous = make_contract([*OFFER_B, "The basic salary is 11,000 SAR per month."])
    result = agent_for(spy).compare([offers["a"][1], ambiguous])
    value = values_by_contract(result.dimension("basic_salary"))["contract_2"]
    assert value.status == "ambiguous" and {"10,000 SAR", "11,000 SAR"} <= set(value.raw_values)
    assert ("contract_2", "ambiguous_term", "salary", "medium") in {
        (r.contract_id, r.code, r.field, r.severity) for r in result.risks}


def test_different_units_and_qualifiers_are_not_converted_by_assumption(offers, make_contract, spy):
    _, months = make_contract([*[l for l in OFFER_B if not l.startswith(("Probation", "Annual Leave"))],
                               "Probation Period: 3 months", "Annual Leave: 30 working days"])
    result = agent_for(spy).compare([offers["a"][1], months])
    probation = result.dimension("probation_period")
    assert not probation.comparable and "different units (days, months)" in probation.explanation
    leave = result.dimension("annual_leave")
    assert not leave.comparable and "different units (days, working days)" in leave.explanation


def test_percentage_allowance_is_never_turned_into_an_amount(offers, make_contract, spy):
    _, percentage = make_contract([*[l for l in OFFER_B if not l.startswith("Housing")],
                                   "Housing Allowance: 25% of basic salary"])
    result = agent_for(spy).compare([offers["a"][1], percentage])
    value = values_by_contract(result.dimension("stated_pay"))["contract_2"]
    assert value.status == "not_comparable" and "percentage (25% of basic salary)" in value.display
    assert "housing_allowance" in value.fields_used
    allowances = values_by_contract(result.dimension("allowances"))["contract_2"]
    assert "housing allowance: 25% of basic salary" in allowances.display


def test_written_total_salary_is_preferred_over_adding_components(offers, make_contract, spy):
    _, with_total = make_contract([*OFFER_B, "Total Salary: 13,500 SAR per month"])
    value = values_by_contract(agent_for(spy).compare([offers["a"][1], with_total]).dimension("stated_pay"))["contract_2"]
    assert value.value == 13500 and value.fields_used == ["total_salary"] and "written in the contract" in value.display


def test_normalisation_is_exact_arithmetic_only():
    assert monthly_money({"amount": 120000, "currency": "SAR", "period": "annual"})[:3] == (10000, "SAR", None)
    assert monthly_money({"amount": 5000, "currency": "SAR", "period": None})[0] is None
    assert monthly_money({"amount": 5000, "currency": None, "period": "monthly"})[2] == "has no written currency"
    assert duration({"count": 2, "unit": "week", "qualifier": None})[:2] == (14, "days")
    assert duration({"count": 1, "unit": "year", "qualifier": None})[:2] == (12, "months")
    assert duration({"count": 21, "unit": "day", "qualifier": "working"})[:2] == (21, "working days")


# --------------------------------------------------------------------------- compliance, CV, salary benchmark
def test_grounded_non_compliance_makes_a_difference_and_is_a_high_risk(offers, make_contract, fake_rag):
    def respond(payload):
        fact = payload["contract_fact"]
        evidence_ids = [e["evidence_id"] for e in payload["evidence"]]
        if fact["field"] != "probation_period" or f"kb-{KB_PROBATION_ART_53}" not in evidence_ids:
            return {"assessment": "requires_review", "explanation": "Needs review.", "cited_evidence_ids": evidence_ids}
        verdict = "non_compliant" if fact["raw_value"] == "200 days" else "compliant"
        return {"assessment": verdict, "explanation": "Article 53 limits probation to 180 days.",
                "cited_evidence_ids": [f"kb-{KB_PROBATION_ART_53}"],
                "evidence_quotes": [{"evidence_id": f"kb-{KB_PROBATION_ART_53}", "quote": PROBATION_QUOTE_AR}],
                "contract_quote": fact["raw_value"]}

    _, long_probation = make_contract([*[l for l in OFFER_A if not l.startswith("Probation")], "Probation Period: 200 days"])
    analysis = AnalysisAgent(ContractAnalysisAgent(fake_rag, LLMEvidenceInterpreter(FakeLLMClient(respond))))
    result = ContractComparisonAgent(analysis).compare([offers["a"][1], long_probation])

    compliance = result.dimension("compliance")
    assert compliance.comparable and compliance.best_contract_ids == ["contract_1"]
    violation = values_by_contract(compliance)["contract_2"]
    assert violation.value == 1 and violation.finding_ids and "Probation Period: 200 days" in violation.source_texts
    [risk] = [r for r in result.risks if r.code == "non_compliant_finding"]
    assert (risk.contract_id, risk.severity, risk.field) == ("contract_2", "high", "probation_period")
    assert "المادة الثالثة والخمسون" in risk.message
    assert result.recommendation.preferred_contract_id == "contract_1"
    assert "compliance" in {f.dimension for f in result.recommendation.decision_factors}


def test_without_interpreter_compliance_ties_and_says_why(offers, spy):
    result = agent_for(spy).compare([offers["a"][1], offers["b"][1]])
    compliance = result.dimension("compliance")
    assert compliance.comparable and compliance.all_equal
    assert all("0 does not mean the contract is compliant" in v.notes[0] for v in compliance.values)
    assert any("requires human review" in c for c in result.recommendation.caveats)
    assert {r.code for r in result.risks if r.severity == "info"} >= {"requires_review"}


def test_cv_is_compared_with_each_contract_job_title(offers, make_contract, cv_en, spy):
    _, accountant = make_contract([*[l for l in OFFER_A if not l.startswith("Job Title")], "Job Title: Senior Accountant"])
    _, untitled = make_contract([l for l in OFFER_A if not l.startswith("Job Title")])
    result = agent_for(spy).compare([offers["a"][1], accountant, untitled], cv_en)

    assert all(call["cv"] is cv_en and call["use_contract_job_title"] for call in spy.calls)
    jobs = [c.cv_analysis.job_compatibility for c in result.contracts]
    assert jobs[0].target_job.title == "Data Analyst" and jobs[0].target_job.source == "contract_job_title"
    assert jobs[1].target_job.title == "Senior Accountant" and jobs[2] is None
    values = values_by_contract(result.dimension("cv_compatibility"))
    assert values["contract_1"].status == "available" and values["contract_1"].value == 3
    assert "cv.work_experience" in values["contract_1"].fields_used and "Job Title: Data Analyst" in values["contract_1"].source_texts
    assert values["contract_2"].value == 1
    assert values["contract_3"].status == "not_found" and "job title was not found" in values["contract_3"].display
    assert ("contract_2", "cv_requirement_gap") in {(r.contract_id, r.code) for r in result.risks}
    assert any("job title was not found" in w for w in result.warnings)
    assert result.cv_document_id == cv_en.document_id

    two = agent_for(spy).compare([offers["a"][1], accountant], cv_en)
    assert two.dimension("cv_compatibility").best_contract_ids == ["contract_1"]


def test_sourced_salary_benchmark_is_shown_but_never_estimated(offers, fake_rag):
    class SourcedProvider:
        name = "test-provider"

        def benchmark(self, contract, context=None):
            observation = SalaryObservation(
                source_name="Example survey", url="https://example.org/survey", tier="market",
                retrieved_at="2026-09-18", quote="Data Analyst salaries range from 9,000 to 14,000 SAR per month.",
                minimum=9000, maximum=14000, currency="SAR", period="monthly", monthly_min=9000, monthly_max=14000,
                basis="base_salary")
            return SalaryBenchmark(status="success", provider=self.name, message="ok", market_min=9000, market_max=14000,
                                   currency="SAR", location="Riyadh", job_title="Data Analyst", period="monthly",
                                   basis="base_salary", data_quality="market", observations=[observation],
                                   sources=[SalarySource(name="Example survey", url="https://example.org/survey",
                                                         source_type="market_survey", tier="market",
                                                         retrieved_at="2026-09-18")])

    analysis = AnalysisAgent(ContractAnalysisAgent(fake_rag, salary_provider=SourcedProvider()))
    result = ContractComparisonAgent(analysis).compare([offers["a"][1], offers["b"][1]])
    market = result.dimension("salary_vs_market")
    assert market.ranking == "not_ranked" and market.best_contract_ids == []
    assert all("https://example.org/survey" in v.display and "9,000 - 14,000" in v.display for v in market.values)
    assert not any("market" in c for c in result.recommendation.caveats)


# --------------------------------------------------------------------------- failures
def test_rag_unavailable_for_all_contracts_is_rag_error_but_terms_are_still_compared(offers, knowledge_base):
    rag = FakeRegulatoryAdapter(knowledge_base, error_code="vector_db_unreachable")
    result = ContractComparisonAgent(AnalysisAgent(ContractAnalysisAgent(rag))).compare([offers["a"][1], offers["b"][1]])

    assert result.status is AnalysisStatus.RAG_ERROR
    assert [e.code for e in result.errors] == ["contract_analysis_failed", "contract_analysis_failed"]
    assert not result.dimension("compliance").comparable
    assert {(r.contract_id, r.code) for r in result.risks if r.severity == "medium"} == {
        ("contract_1", "regulatory_check_error"), ("contract_2", "regulatory_check_error")}
    assert result.recommendation.preferred_contract_id == "contract_1"
    assert "compliance" not in {f.dimension for f in result.recommendation.decision_factors}
    assert any("regulatory RAG was unavailable" in c for c in result.recommendation.caveats)


def test_rag_failing_for_one_contract_is_partial(offers, knowledge_base):
    class FailsAfter(FakeRegulatoryAdapter):
        limit: int = 0

        def retrieve_evidence(self, query, top_k=None):
            if len(self.calls) >= self.limit:
                self.error_code = "vector_db_unreachable"
            return super().retrieve_evidence(query, top_k)

    probe = FakeRegulatoryAdapter(knowledge_base)
    ContractAnalysisAgent(probe).analyze(offers["a"][1])
    rag = FailsAfter(knowledge_base)
    rag.limit = len(probe.calls)
    result = ContractComparisonAgent(AnalysisAgent(ContractAnalysisAgent(rag))).compare([offers["a"][1], offers["b"][1]])
    assert [c.analysis_status for c in result.contracts] == [AnalysisStatus.SUCCESS, AnalysisStatus.RAG_ERROR]
    assert result.status is AnalysisStatus.PARTIAL
    assert values_by_contract(result.dimension("compliance"))["contract_2"].status == "not_comparable"


def test_unreadable_contract_prevents_a_recommendation(offers, parsed, spy):
    unreadable = extract_contract(parsed["sample_empty.pdf"])
    result = agent_for(spy).compare([offers["a"][1], unreadable])
    assert result.contracts[1].analysis_status is AnalysisStatus.INVALID_INPUT
    assert result.status is AnalysisStatus.PARTIAL
    assert result.recommendation.status == "not_possible"
    assert ("contract_2", "analysis_failed", "high") in {(r.contract_id, r.code, r.severity) for r in result.risks}
    assert values_by_contract(result.dimension("basic_salary"))["contract_2"].status == "analysis_unavailable"


@pytest.mark.parametrize("contracts, kwargs, code", [
    ("one", {}, "too_few_contracts"),
    ("six", {}, "too_many_contracts"),
    ("with_cv", {}, "invalid_input"),
    ("two", {"priorities": ["salary_size"]}, "invalid_priority"),
    ("two", {"priorities": ["annual_leave", "annual_leave"]}, "invalid_priority"),
    ("two", {"priorities": ["cv_compatibility"]}, "invalid_priority"),
    ("two", {"priorities": "annual_leave"}, "invalid_priority"),
    ("two", {"labels": ["only one"]}, "invalid_input"),
    ("two", {"cv": "a cv"}, "invalid_input"),
])
def test_invalid_requests_are_explicit_and_analyse_nothing(offers, cv_en, spy, contracts, kwargs, code):
    a, b = offers["a"][1], offers["b"][1]
    lists = {"one": [a], "six": [a, b] * 3, "with_cv": [a, cv_en], "two": [a, b]}
    result = agent_for(spy).compare(lists[contracts], **kwargs)
    assert result.status is AnalysisStatus.INVALID_INPUT
    assert code in [e.code for e in result.errors]
    assert spy.calls == [] and result.contracts == [] and result.recommendation is None


def test_contracts_argument_must_be_a_list(spy):
    result = agent_for(spy).compare("contract.pdf")
    assert result.status is AnalysisStatus.INVALID_INPUT and spy.calls == []


def test_unexpected_failure_is_analysis_error(offers, spy, monkeypatch):
    monkeypatch.setattr(ContractComparisonAgent, "_compare", lambda *a: 1 / 0)
    result = agent_for(spy).compare([offers["a"][1], offers["b"][1]])
    assert result.status is AnalysisStatus.ANALYSIS_ERROR and result.errors[0].code == "comparison_failed"
    assert len(result.contracts) == 2 and result.recommendation is None


def test_identical_contracts_are_equal_and_flagged(offers, spy):
    result = agent_for(spy).compare([offers["a"][1], offers["a"][1]])
    assert result.recommendation.status == "no_clear_preference"
    assert "equal on every dimension" in result.recommendation.explanation
    assert any("same document" in w for w in result.warnings)


def test_two_equal_leaders_are_not_split_arbitrarily(offers, spy):
    result = agent_for(spy).compare([offers["a"][1], offers["b"][1], offers["a"][1]])
    assert result.recommendation.status == "no_clear_preference"
    assert "contract_1, contract_3 are equal to each other" in result.recommendation.explanation


def test_custom_labels_and_max_contracts(offers, spy):
    result = agent_for(spy, max_contracts=3).compare([offers["a"][1], offers["b"][1]], labels=["Offer A", "Offer B"])
    assert [c.label for c in result.contracts] == ["Offer A", "Offer B"]
    assert agent_for(spy, max_contracts=3).compare([offers["a"][1]] * 4).errors[0].code == "too_many_contracts"
    with pytest.raises(ValueError):
        ContractComparisonAgent(AnalysisAgent())


# --------------------------------------------------------------------------- traceability and models
def test_pdf_contract_values_keep_page_numbers(parsed, contract_en, spy):
    pdf = extract_contract(parsed["sample_contract_text.pdf"])
    result = agent_for(spy).compare([pdf, contract_en])
    salary = values_by_contract(result.dimension("basic_salary"))
    assert salary["contract_1"].page_numbers and salary["contract_2"].page_numbers == []
    assert result.dimension("basic_salary").all_equal


def test_arabic_contract_can_be_part_of_a_three_way_comparison(offers, contract_ar, spy):
    result = agent_for(spy).compare([offers["a"][1], offers["b"][1], contract_ar])
    assert result.status is AnalysisStatus.SUCCESS and len(result.contracts) == 3
    value = values_by_contract(result.dimension("basic_salary"))["contract_3"]
    assert value.value == 9500 and value.unit == "SAR/month"
    hours = values_by_contract(result.dimension("weekly_working_hours"))["contract_3"]
    assert hours.status == "not_comparable" and contract_ar.working_hours.status is FieldStatus.FOUND


def test_models_refuse_unexplained_or_inconsistent_recommendations():
    value = ContractValue(contract_id="contract_1", status="available", value=1, display="1")
    with pytest.raises(ValueError, match="decision factor"):
        Recommendation(status="preferred_contract", method="dominance", preferred_contract_id="contract_1", explanation="x")
    factor = DecisionFactor(dimension="d", label="d", favours=["contract_2"], explanation="x", evidence=[value])
    with pytest.raises(ValueError, match="lead on at least one"):
        Recommendation(status="preferred_contract", method="dominance", preferred_contract_id="contract_1",
                       explanation="x", decision_factors=[factor])
    with pytest.raises(ValueError):
        Recommendation(status="no_clear_preference", method="dominance", preferred_contract_id="contract_1", explanation="x")
    with pytest.raises(ValueError, match="only an 'available' value"):
        ContractValue(contract_id="contract_1", status="not_found", value=3, display="x")
    with pytest.raises(ValueError, match="comparable ranked dimension"):
        DimensionComparison(dimension="notice_period", label="n", category="contract_terms", ranking="not_ranked",
                            rationale="r", values=[value], comparable=True, best_contract_ids=["contract_1"], explanation="e")
    with pytest.raises(ValueError, match="requires at least one ErrorInfo"):
        ContractComparisonResult(status=AnalysisStatus.INVALID_INPUT, summary="x")


def test_logs_contain_no_contract_content(offers, spy, caplog):
    with caplog.at_level(logging.DEBUG, logger="sanad"):
        agent_for(spy).compare([offers["a"][1], offers["b"][1]])
    assert "12,000" not in caplog.text and "Data Analyst" not in caplog.text and "Riyadh" not in caplog.text


def test_from_settings_reuses_the_analysis_agent(fake_rag):
    agent = ContractComparisonAgent.from_settings(SanadSettings(openai_api_key=None), AnalysisSettings(),
                                                  evidence_source=fake_rag)
    assert isinstance(agent.analysis_agent, AnalysisAgent)
    assert agent.analysis_agent.contract_agent.evidence_source is fake_rag
