"""Data-mapping correctness for Legal Compliance.

The deterministic clause_findings status (agents/legal_rules.py, Stage 3) must be the ONLY source of
a COMPLIANT/NON_COMPLIANT/REQUIRES_REVIEW/... verdict shown anywhere in the interface, and it must
never be overridden or contradicted by the informational, always-demoted field-level `findings`
(Stage 4 - see the module docstring of agents/contract_analysis.py and the validator on
models.analysis.AnalysisFinding.status, which makes a field-level compliant/non_compliant reading
impossible by construction).

Regression for a real bug found by manually testing the running application against the Aramco
appointment letter: Probation period, Working hours, Annual leave and Notice period were shown as
"Needs attention" (the field-level, always-demoted reading) in the Legal Compliance card/tab and in
Findings' "Requires review" list, while the deterministic clause-level result for several of them was
actually Compliant, and while the SAME contract's Findings "Key findings" list correctly showed
Compliant for the same field, sourced from clause_findings. Root cause: `frontend/ui.py`'s Legal
Compliance tab rendered `view.finding_rows` (field-level) instead of the deterministic clause_findings,
and `view.findings_summary`'s "requires_review" list read the same field-level findings. Both were
switched to read only `view.legal_compliance_findings` (see frontend/view.py), which is now the single
source every Legal Compliance display (Overview card, Legal Compliance tab, Findings tab) reads from.

Also covers a related, separately-observed mapping gap: the Aramco letter states its pay as a single
"in-hand" figure (`in_hand_salary`), not a `salary`/`total_salary` split; the Compensation view read
only the latter two, so a genuinely extracted figure was shown as "not found".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.contract_analysis import ContractAnalysisAgent
from extraction.contract import extract_contract
from frontend import i18n, ui, view
from tests.fakes.regulatory_adapter import FakeRegulatoryAdapter

pytest.importorskip("fastapi")

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "documents" / "aramco_appointment_letter.pdf"


@pytest.fixture(scope="module")
def aramco_result(processor, knowledge_base) -> dict:
    """The real Aramco letter, run through the real extraction + Stage 3 rule engine (only Qdrant/the
    embedding model are faked - see tests/fakes/regulatory_adapter.py), reshaped exactly as the API
    would return it: {"analysis": {"contract_analysis": <ContractAnalysisResult>}}.
    """
    if not FIXTURE.exists():
        pytest.skip(f"{FIXTURE.name} is not checked in")
    document = processor.parse_bytes(FIXTURE.read_bytes(), FIXTURE.name)
    extraction = extract_contract(document)
    adapter = FakeRegulatoryAdapter(knowledge_base)
    agent = ContractAnalysisAgent(adapter, max_clause_checks=200)
    analysis = agent.analyze(extraction)
    return {"analysis": {"contract_analysis": json.loads(analysis.model_dump_json())}}


# --------------------------------------------------------------------------- 1-6: every allowed status renders
@pytest.mark.parametrize("status, label_en", [
    ("compliant", "Compliant"),
    ("non_compliant", "Non-compliant"),
    ("requires_review", "Needs attention"),
    ("not_found", "Not stated in the contract"),
    ("not_applicable", "Contract fact"),
    ("ambiguous", "Contract wording is unclear"),
])
def test_each_allowed_status_renders_its_own_label(status, label_en):
    _tone, _mark, label_key = ui.FINDING_STATES[status]
    assert i18n.t(label_key, "en") == label_en


def test_no_status_outside_the_six_allowed_ones_is_ever_invented():
    from models.analysis import FindingStatus
    allowed = {"compliant", "non_compliant", "requires_review", "not_found", "not_applicable", "ambiguous"}
    # error/insufficient_evidence are real FindingStatus values too (a failed or unusable check), kept
    # distinct from the six deterministic compliance outcomes the user's spec names - both are still
    # rendered honestly (never silently reclassified into one of the six) via the same FINDING_STATES map.
    assert allowed <= {s.value for s in FindingStatus}
    assert set(ui.FINDING_STATES) == {s.value for s in FindingStatus}


# --------------------------------------------------------------------------- 7/8: Aramco end-to-end consistency
def test_the_aramco_probation_is_compliant_everywhere_and_never_requires_review(aramco_result):
    rows = {row["field"]: row["status"] for row in view.legal_compliance_findings(aramco_result)}
    assert rows["probation_period"] == "compliant"

    counts = view.legal_compliance_counts(aramco_result)
    assert counts.get("compliant", 0) >= 1

    findings = view.findings_summary(aramco_result)
    assert any(f["field"] == "probation_period" for f in findings["key_findings"])
    assert all(f["field"] != "probation_period" for f in findings["requires_review"])


def test_the_aramco_notice_and_working_hours_are_also_compliant_and_consistent(aramco_result):
    rows = {row["field"]: row["status"] for row in view.legal_compliance_findings(aramco_result)}
    for field in ("notice_period_during_probation", "notice_period_after_confirmation", "working_hours"):
        assert rows[field] == "compliant", f"{field} should be compliant (see Article 75 / Article 98)"

    findings = view.findings_summary(aramco_result)
    key_fields = {f["field"] for f in findings["key_findings"]}
    review_fields = {f["field"] for f in findings["requires_review"]}
    for field in ("notice_period_during_probation", "notice_period_after_confirmation", "working_hours"):
        assert field in key_fields and field not in review_fields


def test_no_field_is_ever_in_both_key_findings_and_requires_review(aramco_result):
    findings = view.findings_summary(aramco_result)
    key_fields = {f["field"] for f in findings["key_findings"]}
    review_fields = {f["field"] for f in findings["requires_review"]}
    assert not (key_fields & review_fields)


def test_the_legal_compliance_tab_and_the_overview_card_share_the_same_counts(aramco_result):
    """The compact Overview card (counts) and the full Legal Compliance tab (rows) must never
    disagree: they are required to read the same function (view.legal_compliance_findings)."""
    rows = view.legal_compliance_findings(aramco_result)
    tallied: dict[str, int] = {}
    for row in rows:
        tallied[row["status"]] = tallied.get(row["status"], 0) + 1
    assert view.legal_compliance_counts(aramco_result) == tallied


# --------------------------------------------------------------------------- 9: salary benchmark stays separate
def test_salary_clauses_never_appear_in_legal_compliance(aramco_result):
    rows = view.legal_compliance_findings(aramco_result)
    assert all(row["field"] != "in_hand_salary" for row in rows)


def test_salary_benchmark_never_changes_the_legal_compliance_counts(aramco_result):
    baseline = view.legal_compliance_counts(aramco_result)
    mutated = json.loads(json.dumps(aramco_result))
    # A deliberately unfavourable benchmark (well above the contract's own pay) - if compliance and
    # salary benchmarking were ever mixed, this would be the shape of bug that turns a "below market
    # range" salary into a non-compliant legal finding. It must not move the needle at all.
    mutated["analysis"]["contract_analysis"]["salary_benchmark"] = {
        "status": "success", "message": "", "job_title": "Driver", "location": "Riyadh",
        "market_min": 5000.0, "market_max": 8000.0, "currency": "SAR", "period": "monthly",
        "basis": "base_salary", "data_quality": "market",
        "contract_salary": {"amount": 1800.0, "currency": "SAR", "period": "monthly"},
        "sources": [], "observations": [], "limitations": [],
    }
    assert view.legal_compliance_counts(mutated) == baseline


# --------------------------------------------------------------------------- compensation mapping fix
def test_the_aramco_in_hand_salary_is_shown_instead_of_a_false_not_found(aramco_result):
    rows = {row["field"]: row for row in view.compensation_summary(aramco_result)}
    salary_row = rows["salary"]
    assert salary_row["display"] != "__not_found__"
    assert salary_row["label"] == "Salary (in-hand)"
    assert "1,800" in salary_row["display"] or "1800" in salary_row["display"]
    # every OTHER compensation component this contract genuinely does not state is still "not found" -
    # nothing is invented for housing/transportation/other allowances/total compensation
    for field in ("housing_allowance", "transportation_allowance", "other_allowances", "total_salary"):
        assert rows[field]["display"] == "__not_found__"


# --------------------------------------------------------------------------- 11/14: comparison grouping intact
def test_comparison_sections_groups_the_salary_benchmark_dimension_on_its_own():
    comparison = {
        "contracts": [{"contract_id": "c1", "label": "Contract A"}, {"contract_id": "c2", "label": "Contract B"}],
        "comparison": {
            "salary": [
                {"dimension": "basic_salary", "label": "Basic salary", "ranking": "higher_is_better",
                 "values": [{"contract_id": "c1", "display": "5,000 SAR"}, {"contract_id": "c2", "display": "6,000 SAR"}],
                 "best_contract_ids": ["c2"], "explanation": "Contract B pays more."},
                {"dimension": "salary_vs_market", "label": "Salary vs market", "ranking": "not_ranked",
                 "values": [{"contract_id": "c1", "display": "inside range"}], "best_contract_ids": [],
                 "explanation": "Informational only."},
            ],
        },
    }
    sections = view.comparison_sections(comparison)
    assert [row["Dimension"] for row in sections["salary"]] == ["Basic salary"]
    assert [row["Dimension"] for row in sections["salary_benchmark"]] == ["Salary vs market"]
    assert "salary_benchmark" in view.COMPARISON_GROUP_ORDER


# --------------------------------------------------------------------------- 12/13: navigation shape
def test_home_has_exactly_three_primary_workflows():
    from frontend import pages
    assert len(pages.SERVICES) == 3
    assert {key for key, *_ in pages.SERVICES} == {"ask", "analyze", "compare"}


def test_salary_benchmark_has_no_standalone_nav_entry_or_page():
    from frontend import app, pages
    assert {key for key, *_ in app.NAV} == {"home", "ask", "analyze", "compare"}
    assert "salary" not in pages.PAGES
    # the backend task and service call are kept - only the standalone entry point is gone
    assert app.SERVICE_TASKS["salary"] == "salary_benchmark"
