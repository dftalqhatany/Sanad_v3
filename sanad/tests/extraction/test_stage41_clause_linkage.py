"""Stage 4.1 regression: clause-to-field linkage hardening, specifically for Shifa and Taibah.

The Stage 4 hardening report found that `derive_contract_fact` relied almost entirely on verbatim
text containment between a clause's own segmented text and a field's `source.text`, which happens to
work for narrative-sentence contracts (Aramco, the two data-engineer DOCX contracts) but leaves Shifa
Al-Bahr (a dense, table-heavy, bilingual PDF) with zero clause-to-field links and Taibah (also
table-heavy and bilingual) with only one - even though extraction itself succeeds for both.

Stage 4.1 extends `agents.legal_rules._readings_within_clause` with a structural fallback keyed on
`ExtractedField.sources[].table_id` / `section_id` (see `_source_within_clause`, `_structural_candidates`
in agents/legal_rules.py) once text matching fails, adds table-row clause synthesis to
`extraction.clauses.ClauseSegmenter` (`_table_row_clauses`) so a table-sourced field has a clause to
link to in the first place, and adds a topic-assignment fallback in
`ContractAnalysisAgent._clause_checks` (via `linked_field_for_clause`) so a clause that structurally
links to exactly one field is not skipped before `derive_contract_fact` is ever called, purely because
its own sentence-classification came back OTHER/UNKNOWN.

None of this touches extraction, parsing, the RAG, or the deterministic legal-rule engine itself -
only how an already-extracted field is connected to the clause the legal-rule engine evaluates it
against. This file locks in the concrete, real-fixture linkage counts that resulted, and documents the
one remaining, known, out-of-scope gap (working_hours on these two fixtures - see the last test).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.contract_analysis import ContractAnalysisAgent
from extraction.contract import extract_contract
from models.analysis import FindingStatus
from models.extraction import FieldStatus
from tests.fakes.regulatory_adapter import FakeRegulatoryAdapter

DOCS_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "documents"


@pytest.fixture(scope="module")
def shifa_extraction(processor):
    path = DOCS_DIR / "contract_shifa_albahr_pharmacist.pdf"
    if not path.exists():
        pytest.skip(f"fixture not checked in at {path}")
    return extract_contract(processor.parse_bytes(path.read_bytes(), path.name))


@pytest.fixture(scope="module")
def taibah_extraction(processor):
    path = DOCS_DIR / "contract_taibah_grand_front_office_manager.pdf"
    if not path.exists():
        pytest.skip(f"fixture not checked in at {path}")
    return extract_contract(processor.parse_bytes(path.read_bytes(), path.name))


@pytest.fixture(scope="module")
def shifa_result(shifa_extraction, knowledge_base):
    adapter = FakeRegulatoryAdapter(knowledge_base)
    return ContractAnalysisAgent(adapter, max_clause_checks=250).analyze(shifa_extraction)


@pytest.fixture(scope="module")
def taibah_result(taibah_extraction, knowledge_base):
    adapter = FakeRegulatoryAdapter(knowledge_base)
    return ContractAnalysisAgent(adapter, max_clause_checks=250).analyze(taibah_extraction)


def _linked(result) -> dict[str, list]:
    """clause_findings that actually carry a linked field, grouped by field name."""
    by_field: dict[str, list] = {}
    for finding in result.clause_findings:
        if finding.field:
            by_field.setdefault(finding.field, []).append(finding)
    return by_field


# --------------------------------------------------------------------------- headline regression
def test_shifa_linkage_count_no_longer_zero(shifa_result):
    """Before Stage 4.1, Shifa had exactly zero clause_findings with a linked field, despite
    extraction succeeding for annual_leave, total_salary and other fields - the entire point of this
    stage. This is the single most important regression to guard."""
    linked = _linked(shifa_result)
    assert len(linked) >= 4, sorted(linked)


def test_taibah_linkage_count_improves_past_one(taibah_result):
    """Before Stage 4.1, Taibah had exactly one clause_finding with a linked field. This guards that
    the fix generalizes rather than being a one-field special case."""
    linked = _linked(taibah_result)
    assert len(linked) >= 3, sorted(linked)


# --------------------------------------------------------------------------- specific important fields, Shifa
def test_shifa_annual_leave_links_via_a_synthesized_table_row_clause(shifa_result):
    """Shifa states annual leave inside a table row ('30 calendar days'); this can only link at all
    once ClauseSegmenter synthesizes a clause for that row (extraction.clauses._table_row_clauses) and
    the structural table_id fallback (_structural_candidates) connects it back to the annual_leave
    field, since the row's text does not verbatim-contain the field's own source text."""
    findings = [f for f in shifa_result.clause_findings if f.field == "annual_leave"]
    assert findings, "annual_leave never linked to any clause for Shifa"
    finding = findings[0]
    assert finding.contract_fact is not None
    assert finding.contract_fact.normalized_value == 30.0
    assert finding.contract_fact.unit == "day"
    assert finding.legal_rule is not None
    assert finding.status in (FindingStatus.COMPLIANT, FindingStatus.NON_COMPLIANT,
                               FindingStatus.REQUIRES_REVIEW)


def test_shifa_total_salary_links_and_is_never_silently_reclassified_as_salary(shifa_result):
    """total_salary (SAR 14,000/month, gross) must link as itself, not be conflated with the separate
    `salary` (basic) field - the two stay distinct facts even once linkage is more permissive."""
    total = [f for f in shifa_result.clause_findings if f.field == "total_salary"]
    assert total, "total_salary never linked to any clause for Shifa"
    assert total[0].contract_fact.normalized_value == 14000.0
    assert total[0].contract_fact.unit == "sar_monthly"
    assert not any(f.field == "salary" for f in shifa_result.clause_findings
                   if f.clause_id == total[0].clause_id), "salary and total_salary must not share a clause"


def test_shifa_allowance_field_links_where_the_source_supports_it(shifa_result):
    """other_allowances is FOUND (named allowances, no numeric amount) and should still be able to
    link to its own clause even without a comparable value - linkage is about connecting field to
    clause, not about the field having a number the legal rule can compare."""
    allowances = [f for f in shifa_result.clause_findings if f.field == "other_allowances"]
    assert allowances, "other_allowances never linked to any clause for Shifa"


# --------------------------------------------------------------------------- specific important fields, Taibah
def test_taibah_annual_leave_links(taibah_result):
    findings = [f for f in taibah_result.clause_findings if f.field == "annual_leave"]
    assert findings, "annual_leave never linked to any clause for Taibah"
    assert findings[0].contract_fact.normalized_value == 21.0
    assert findings[0].contract_fact.unit == "day"


def test_taibah_salary_links(taibah_result):
    findings = [f for f in taibah_result.clause_findings if f.field == "salary"]
    assert findings, "salary never linked to any clause for Taibah"
    assert findings[0].contract_fact.normalized_value == 8000.0
    assert findings[0].contract_fact.unit == "sar_monthly"


def test_taibah_transportation_allowance_links(taibah_result):
    """Taibah's transportation allowance (SAR 500/month) is a table-sourced field; this is the
    fixture's clearest table_id-based linkage case."""
    findings = [f for f in taibah_result.clause_findings if f.field == "transportation_allowance"]
    assert findings, "transportation_allowance never linked to any clause for Taibah"
    assert findings[0].contract_fact.normalized_value == 500.0
    assert findings[0].contract_fact.unit == "sar_monthly"


def test_taibah_housing_allowance_correctly_stays_unlinked_with_no_value_to_compare(taibah_extraction):
    """Not a bug: Taibah's only housing_allowance candidate reads 'Not applicable (housing is...)'
    with no numeric value, so it is correctly excluded from linkage candidates - Stage 4.1 makes
    linkage more capable, it does not invent values extraction never found."""
    assert taibah_extraction.housing_allowance.status is FieldStatus.AMBIGUOUS
    assert taibah_extraction.housing_allowance.value is None


# --------------------------------------------------------------------------- known, documented, out-of-scope gap
def test_working_hours_is_found_but_remains_unlinked_on_both_fixtures_known_gap(shifa_result, taibah_result):
    """Both fixtures state working hours in ordinary prose sentences ('...working hours shall be daily
    eight (8) hours...', '...forty-eight (48) hours weekly...') that extraction reads correctly - but
    the sentence-joining pass in extraction.text.iter_sentences merges those specific sentences into a
    neighbouring section's sentence stream, so extraction.clauses.ClauseSegmenter never emits a
    ClauseValue for their own section_id at all. Structural linkage can only connect a field to a
    clause that exists; it cannot manufacture one for a section that produced zero clauses, and closing
    that gap would mean changing sentence segmentation itself, which is out of Stage 4.1's scope ("do
    not rebuild extraction, parsing, RAG, or the legal-rule engine"). This test exists so that gap stays
    documented and visible rather than silently reappearing as a mystery if segmentation changes again;
    it is not a regression this stage introduced, and it should be revisited in a future segmentation
    ticket, not papered over here."""
    for result, name in ((shifa_result, "shifa"), (taibah_result, "taibah")):
        linked_fields = {f.field for f in result.clause_findings if f.field}
        assert "working_hours" not in linked_fields, (
            f"working_hours now links for {name} - if extraction.clauses segmentation was extended to "
            "cover this sentence, please strengthen this test into a positive linkage assertion instead "
            "of deleting it."
        )
