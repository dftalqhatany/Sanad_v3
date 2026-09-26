"""Stage 4 regression: the four real contract fixtures other than the Aramco letter, end to end.

tests/extraction/test_aramco_legal_findings.py is the mandatory Stage 3/4 regression for the Aramco
appointment letter. This file gives the same "parse -> extract -> segment -> classify -> existing RAG
evidence -> structured legal rule -> deterministic comparison -> finding" pipeline the same treatment
for the remaining four fixtures (two real bilingual PDFs, two synthetic-but-realistic DOCX contracts),
so that all five fixtures required by the Stage 4 hardening task have Contract Analysis Agent coverage,
not only extraction-level coverage.

Only the infrastructure underneath the existing RAG (Qdrant, the embedding model) is replaced, exactly
as in the Aramco file: tests/fakes/regulatory_adapter.FakeRegulatoryAdapter builds real evidence objects
from the real 249-article knowledge base; only which articles come back for which question is scripted.

These four documents are structurally different from the Aramco letter (two are dense, tabular,
bilingual PDFs; two are shorter Arabic DOCX contracts), and clause segmentation does not always manage
to link a segmented clause back to a Stage 1 field for them (most often on the two PDFs, whose fields
are read from table cells rather than narrative sentences) - that is a genuine, pre-existing gap in the
clause <-> field linkage documented here and in the Stage 4 hardening report, not something this test
file hides. Where a clause_findings-level (deterministic, Stage 3) result exists it is asserted
directly; where it does not, the field-level (Stage 4) finding is still checked, always exactly as
NOT_FOUND / NOT_APPLICABLE / AMBIGUOUS-turned-REQUIRES_REVIEW - never a compliance guess.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.contract_analysis import ContractAnalysisAgent
from extraction.contract import extract_contract
from models.analysis import AnalysisStatus, FindingStatus
from models.extraction import ClauseType, FieldStatus
from models.legal_rules import RuleType
from tests.fakes.regulatory_adapter import FakeRegulatoryAdapter

DOCS_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "documents"
FIXTURES = [
    "contract_shifa_albahr_pharmacist.pdf",
    "contract_taibah_grand_front_office_manager.pdf",
    "data_engineer_contract_1.docx",
    "data_engineer_contract_3.docx",
]


@pytest.fixture(scope="module", params=FIXTURES)
def fixture_name(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def extraction(fixture_name, processor):
    path = DOCS_DIR / fixture_name
    if not path.exists():
        pytest.skip(f"{fixture_name} is not checked in at {path}")
    document = processor.parse_bytes(path.read_bytes(), fixture_name)
    return extract_contract(document)

@pytest.fixture(scope="module")
def result(extraction, knowledge_base):
    adapter = FakeRegulatoryAdapter(knowledge_base)  # test infrastructure standing in for live Qdrant
    agent = ContractAnalysisAgent(adapter, max_clause_checks=200)
    return agent.analyze(extraction)


# --------------------------------------------------------------------------- pipeline never crashes
def test_all_four_fixtures_are_analysed_without_error(extraction, result, fixture_name):
    """Regression for a real bug found while adding this coverage: a field whose extraction status is
    NOT_APPLICABLE but which still has a regulatory topic (e.g. an explicit probation waiver) used to
    raise a KeyError deep in ContractAnalysisAgent._finding and silently turn into ANALYSIS_ERROR with
    zero findings for the whole document (agents/contract_analysis.py + the NOT_APPLICABLE validator in
    models/analysis.py were both hardened for this)."""
    assert extraction.status.value in ("success", "partial")
    assert result.status not in (AnalysisStatus.ANALYSIS_ERROR, AnalysisStatus.INVALID_INPUT,
                                 AnalysisStatus.RAG_ERROR), fixture_name
    assert len(result.clause_findings) == len(result.clause_checks) == len(extraction.clauses)
    assert len(result.findings) == len(extraction.fields())


def test_every_status_produced_is_one_of_the_declared_statuses(result):
    for finding in result.findings:
        assert finding.status in FindingStatus
    for finding in result.clause_findings:
        assert finding.status in FindingStatus


# --------------------------------------------------------------------------- Priority 1 regression, on real documents
def test_no_field_level_finding_ever_carries_a_compliance_verdict(result, fixture_name):
    """Stage 4 hardening: compliant/non_compliant may only ever come from the deterministic
    clause-level comparison, never from the field-level (LLM-eligible) path - checked here against
    real documents, not only the synthetic cases in tests/agents/test_contract_analysis.py."""
    assert not {f.status for f in result.findings} & {FindingStatus.COMPLIANT, FindingStatus.NON_COMPLIANT}, fixture_name


# --------------------------------------------------------------------------- explicitly not applicable
def test_shifa_and_taibah_state_a_probation_waiver_that_is_not_applicable_not_not_found(extraction, result, fixture_name):
    if fixture_name not in ("contract_shifa_albahr_pharmacist.pdf", "contract_taibah_grand_front_office_manager.pdf"):
        pytest.skip("probation waiver wording is specific to the shifa/taibah fixtures")
    assert extraction.probation_period.status is FieldStatus.NOT_APPLICABLE
    assert extraction.probation_status.status is FieldStatus.NOT_APPLICABLE
    finding = result.finding("probation_period")
    assert finding.status is FindingStatus.NOT_APPLICABLE
    assert finding.status is not FindingStatus.NOT_FOUND  # the contract *addressed* it and ruled it out
    assert "does not apply" in finding.explanation


# --------------------------------------------------------------------------- ambiguous extraction (real fixture)
def test_taibah_housing_allowance_is_ambiguous_and_requires_review_not_a_guess(extraction, result, fixture_name):
    if fixture_name != "contract_taibah_grand_front_office_manager.pdf":
        pytest.skip("this fixture's housing_allowance is the one that reads ambiguously")
    assert extraction.housing_allowance.status is FieldStatus.AMBIGUOUS
    finding = result.finding("housing_allowance")
    assert finding.status is FindingStatus.REQUIRES_REVIEW
    assert finding.status not in (FindingStatus.COMPLIANT, FindingStatus.NON_COMPLIANT)
    assert "ambiguous" in finding.explanation.lower()


# --------------------------------------------------------------------------- conditional / qualitative legal rule
def test_data_engineer_contracts_salary_rule_is_qualitative_and_never_a_guessed_verdict(extraction, result, fixture_name):
    if fixture_name not in ("data_engineer_contract_1.docx", "data_engineer_contract_3.docx"):
        pytest.skip("this assertion is about the two data-engineer contracts' salary clause")
    [salary_finding] = [f for f in result.clause_findings if f.field == "salary"]
    assert salary_finding.contract_fact is not None and salary_finding.contract_fact.normalized_value > 0
    assert salary_finding.legal_rule is not None
    # Article 90 (payment of wages) is a REQUIREMENT/qualitative rule, not a numeric floor or ceiling:
    # there is nothing for a monthly salary figure to be compared against, so the deterministic engine
    # must say NOT_APPLICABLE - never COMPLIANT/NON_COMPLIANT and never REQUIRES_REVIEW as a shrug.
    assert salary_finding.legal_rule.rule_type in (RuleType.REQUIREMENT, RuleType.INFORMATIONAL, RuleType.PROHIBITION)
    assert salary_finding.status is FindingStatus.NOT_APPLICABLE
    assert salary_finding.status not in (FindingStatus.COMPLIANT, FindingStatus.NON_COMPLIANT)


# --------------------------------------------------------------------------- missing information
def test_net_salary_and_split_notice_periods_are_not_found_never_guessed(extraction, result, fixture_name):
    """None of these four fixtures states a separate notice period for during-probation vs
    after-confirmation, so that field must always be NOT_FOUND - not silently defaulted, not merged
    with a different notice field. The same holds for 'net salary' on three of the four fixtures; Shifa
    is the one exception (see test_extraction_hardening.py::
    test_net_salary_field_exists_and_stays_well_formed_on_every_fixture for why: its PDF states the
    figure in a genuine, distinct table row, a pre-existing and unrelated extraction path, not something
    Stage 4.1's clause-to-field linkage work touches). Since Shifa's net_salary has no regulatory topic
    (net_salary is a document-fact-only field), its field-level finding is NOT_APPLICABLE, not NOT_FOUND,
    for that one field on that one fixture - never silently defaulted, not merged with a different
    salary field."""
    for name in ("notice_period_during_probation", "notice_period_after_confirmation"):
        field = getattr(extraction, name)
        assert field.status is FieldStatus.NOT_FOUND, (fixture_name, name)
        finding = result.finding(name)
        assert finding.status is FindingStatus.NOT_FOUND, (fixture_name, name)

    if fixture_name == "contract_shifa_albahr_pharmacist.pdf":
        assert extraction.net_salary.status is FieldStatus.FOUND, (fixture_name, "net_salary")
        assert result.finding("net_salary").status is FindingStatus.NOT_APPLICABLE, (fixture_name, "net_salary")
    else:
        assert extraction.net_salary.status is FieldStatus.NOT_FOUND, (fixture_name, "net_salary")
        assert result.finding("net_salary").status is FindingStatus.NOT_FOUND, (fixture_name, "net_salary")


# --------------------------------------------------------------------------- salary components stay separate
def test_salary_components_are_read_as_separate_facts_never_cross_compared(extraction, fixture_name):
    if fixture_name not in ("data_engineer_contract_1.docx", "data_engineer_contract_3.docx"):
        pytest.skip("only these two fixtures state both a basic salary and a written total")
    basic = extraction.salary
    total = extraction.total_salary
    assert basic.status is FieldStatus.FOUND and total.status is FieldStatus.FOUND
    assert basic.value.amount != total.value.amount  # basic pay and the written total are different figures
    assert total.value.amount > basic.value.amount    # the total includes the allowances basic excludes
    housing = extraction.housing_allowance
    transport = extraction.transportation_allowance
    assert housing.status is FieldStatus.FOUND and transport.status is FieldStatus.FOUND
    assert {basic.value.amount, housing.value.amount, transport.value.amount, total.value.amount} == {
        basic.value.amount, housing.value.amount, transport.value.amount, total.value.amount
    }  # four genuinely distinct readings, not one figure relabelled four times
    assert len({basic.value.amount, housing.value.amount, transport.value.amount, total.value.amount}) == 4
