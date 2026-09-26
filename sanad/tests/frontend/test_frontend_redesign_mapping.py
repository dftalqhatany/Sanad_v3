"""Correctness fixes and additions from the "MAJOR UI/UX REDESIGN + SALARY BENCHMARK INTEGRATION FIX"
request, tested against synthetic (but schema-accurate) payloads rather than a real pipeline run, so
they run without the extraction/RAG stack.

Part 23 regression: a real fixture (the Shifa pharmacist appointment letter) showed "Other Allowances"
as unreadable Arabic - `إ ج مال ي بدلا ت ن قد ي ة أ خ رى` - even though the contract's own extracted
`value` correctly held two named allowances (just with no numeric figure for either). Root cause,
found by tracing `extraction.fields()` directly against that fixture: `_component_display` skipped a
named-but-figureless allowance entirely, so `_field_row` fell through to `raw_value`, which for this
Arabic table cell had come back as a known PDF-extraction artifact (each letter its own
whitespace-separated token). That is a frontend mapping bug, not a data-availability gap: the contract
did state something usable (the allowance names); it was simply never reached. Fixed by (1) having
`_component_display` show a named component's name when it has no figure, and (2) never using
`raw_value` as a primary-value fallback when it reads as this kind of corrupted text (`_looks_corrupted`)
- in that case the field is shown the same way a genuinely ambiguous one is, with the raw text still
available in evidence (`source_text`), never fabricated and never silently dropped.

Also covers the Compare Contracts additions this request asked for: per-contract Legal Compliance and
Salary Benchmark blocks built from each contract's own embedded `contract_analysis` (never the
comparison agent's field-level "compliance" dimension, which structurally cannot see a non-compliant
clause - see agents/comparison_dimensions.py and legal_compliance_findings's docstring), and the
"Contract A - <job title>" naming convention.
"""

from __future__ import annotations

from frontend import view


def _fact(status: str, value=None, raw_value=None, source_text=None, page=None) -> dict:
    return {"extraction_status": status, "value": value, "raw_value": raw_value,
            "source_text": source_text, "page_number": page}


def _finding(field: str, fact: dict) -> dict:
    return {"field": field, "contract_fact": fact}


def _result(findings: list[dict], clause_findings: list[dict] | None = None, salary_benchmark=None) -> dict:
    return {"analysis": {"contract_analysis": {
        "findings": findings, "clause_findings": clause_findings or [], "salary_benchmark": salary_benchmark,
    }}}


CORRUPTED_ARABIC = "إ ج مال ي بدلا ت ن قد ي ة أ خ رى"  # the exact Shifa fixture raw_value, letter-spaced
ENCODING_GARBLED = "��دوس� رده"  # replacement-char corruption,
# matching what the user's own screenshot of the running app showed in the Compensation evidence panel:
# genuine undisplayable glyphs, not just letter-spacing


# --------------------------------------------------------------------------- Part 23: corrupted text
def test_looks_corrupted_flags_the_real_shifa_fixture_text_but_not_ordinary_arabic():
    assert view._looks_corrupted(CORRUPTED_ARABIC) is True
    assert view._looks_corrupted("جدة، المملكة العربية السعودية") is False  # an ordinary Arabic phrase
    assert view._looks_corrupted("Jeddah") is False
    assert view._looks_corrupted(None) is False
    assert view._looks_corrupted("") is False


def test_a_named_allowance_with_no_figure_shows_its_name_not_a_corrupted_fallback():
    """Exact shape of the Shifa other_allowances field: two named items, neither with an amount, and a
    raw_value that is the corrupted PDF-extraction artifact. The primary value must be the name(s),
    never the corrupted raw text."""
    value = [{"name": "9.1.1.4 Total Other Cash Allowances", "amount": None, "currency": None,
             "period": None, "percentage": None, "percentage_basis": None},
             {"name": "Professional allowance", "amount": None, "currency": None, "period": None,
              "percentage": None, "percentage_basis": None}]
    result = _result([_finding("other_allowances", _fact("found", value=value, raw_value=CORRUPTED_ARABIC))])
    rows = {r["field"]: r for r in view.compensation_summary(result, detailed=True)}
    display = rows["other_allowances"]["display"]
    assert CORRUPTED_ARABIC not in display
    assert "Professional allowance" in display
    assert "9.1.1.4 Total Other Cash Allowances" in display


def test_a_figureless_unnamed_component_with_a_corrupted_raw_value_is_shown_as_ambiguous_not_garbled():
    """If there were no usable name at all and only a corrupted raw_value, Sanad must show 'ambiguous'
    (with the raw text still reachable via source_text/evidence) rather than the garbled text itself."""
    result = _result([_finding("other_allowances",
                               _fact("found", value=[{"name": None, "amount": None}], raw_value=CORRUPTED_ARABIC,
                                     source_text=CORRUPTED_ARABIC))])
    rows = {r["field"]: r for r in view.compensation_summary(result, detailed=True)}
    assert rows["other_allowances"]["display"] == "__ambiguous__"
    assert rows["other_allowances"]["source_text"] == CORRUPTED_ARABIC  # never dropped - just not primary


def test_an_ordinary_uncorrupted_raw_value_is_still_shown_as_before():
    """The fix must not change the ordinary, correct case: a plain textual raw_value with no structured
    `value` still displays exactly as it always has."""
    result = _result([_finding("housing_allowance", _fact("found", value=None, raw_value="SAR 2,500.00/month"))])
    rows = {r["field"]: r for r in view.compensation_summary(result, detailed=True)}
    assert rows["housing_allowance"]["display"] == "SAR 2,500.00/month"


def test_overview_display_never_shows_corrupted_text_either():
    result = _result([_finding("work_location", _fact("found", value=CORRUPTED_ARABIC, raw_value=CORRUPTED_ARABIC))])
    rows = {r["field"]: r["display"] for r in view.contract_overview_rows(result)}
    assert rows["work_location"] == "__ambiguous__"


def test_encoding_garbled_text_is_flagged_even_though_its_not_arabic_letter_spacing():
    """Regression for the screenshot the user sent of the running app: the Compensation evidence panel
    showed replacement-character glyphs, not merely letter-spaced Arabic - a stronger, unambiguous
    corruption signal that must be caught on its own, independent of the Arabic-token heuristic."""
    assert view._has_encoding_corruption(ENCODING_GARBLED) is True
    assert view._looks_corrupted(ENCODING_GARBLED) is True
    assert view._has_encoding_corruption("Jeddah") is False
    assert view._has_encoding_corruption(CORRUPTED_ARABIC) is False  # letter-spacing only, not garbled bytes


def test_safe_evidence_text_drops_encoding_garbage_but_keeps_letter_spaced_arabic():
    """Evidence is allowed to show oddly-spaced-but-real Arabic (the user's own instruction: raw source
    text can stay in evidence) but never genuine encoding garbage, which is not "raw source text" -
    it is unreadable bytes with a page number attached."""
    assert view.safe_evidence_text(ENCODING_GARBLED) is None
    assert view.safe_evidence_text(CORRUPTED_ARABIC) == CORRUPTED_ARABIC
    assert view.safe_evidence_text("SAR 14,000.00: Fourteen thousand Saudi Riyals") == \
        "SAR 14,000.00: Fourteen thousand Saudi Riyals"
    assert view.safe_evidence_text(None) is None


def test_a_component_with_encoding_garbled_raw_value_is_ambiguous_not_shown_verbatim():
    """Same fix as the letter-spacing case, but for the stronger corruption signal: a compensation
    component whose only text is genuinely garbled must still fall back to 'ambiguous', never render
    replacement glyphs as if they were the contract's own words."""
    result = _result([_finding("other_allowances",
                               _fact("found", value=[{"name": None, "amount": None}], raw_value=ENCODING_GARBLED))])
    rows = {r["field"]: r for r in view.compensation_summary(result, detailed=True)}
    assert rows["other_allowances"]["display"] == "__ambiguous__"


# --------------------------------------------------------------------------- Compare Contracts: compliance blocks
def _clause(field: str, status: str, clause_name: str) -> dict:
    return {"field": field, "clause_type": "hours", "clause_name": clause_name, "status": status,
            "explanation": f"{clause_name} explanation.", "contract_fact": {"raw_value": "8 hours", "page_number": 1},
            "legal_rule": {"article_number": "98"}, "regulatory_evidence": []}


def _compared_contract(cid: str, label: str, clause_findings: list[dict], findings: list[dict] | None = None,
                       salary_benchmark=None) -> dict:
    return {"contract_id": cid, "label": label, "filename": label,
            "contract_analysis": {"clause_findings": clause_findings, "findings": findings or [],
                                  "salary_benchmark": salary_benchmark}}


def test_comparison_compliance_reads_each_contracts_own_clause_findings_not_the_dimension():
    comparison = {"contracts": [
        _compared_contract("contract_1", "Offer A", [_clause("working_hours", "compliant", "Working hours"),
                                                     _clause("notice_period", "non_compliant", "Notice period")]),
        _compared_contract("contract_2", "Offer B", [_clause("working_hours", "compliant", "Working hours"),
                                                     _clause("notice_period", "compliant", "Notice period")]),
    ]}
    blocks = view.comparison_compliance_by_contract(comparison)
    by_id = {b["contract_id"]: b for b in blocks}
    assert by_id["contract_1"]["counts"] == {"compliant": 1, "non_compliant": 1}
    assert by_id["contract_2"]["counts"] == {"compliant": 2}

    differences = view.comparison_compliance_differences(blocks)
    assert [d["field"] for d in differences] == ["notice_period"]
    statuses = {d["label"]: d["status"] for d in differences[0]["per_contract"]}
    assert statuses == {"Offer A": "non_compliant", "Offer B": "compliant"}


def test_comparison_compliance_differences_is_empty_when_every_contract_agrees():
    comparison = {"contracts": [
        _compared_contract("contract_1", "Offer A", [_clause("working_hours", "compliant", "Working hours")]),
        _compared_contract("contract_2", "Offer B", [_clause("working_hours", "compliant", "Working hours")]),
    ]}
    blocks = view.comparison_compliance_by_contract(comparison)
    assert view.comparison_compliance_differences(blocks) == []


# --------------------------------------------------------------------------- Compare Contracts: salary blocks
def _benchmark(status: str, amount: float | None = 1800.0) -> dict:
    if status == "not_configured":
        return {"status": "not_configured", "message": "Salary benchmarking is not configured.",
                "job_title": "Driver", "location": "Riyadh",
                "contract_salary": {"amount": amount, "currency": "SAR", "period": "monthly"} if amount else None,
                "sources": [], "observations": [], "limitations": []}
    return {"status": "success", "message": "", "job_title": "Driver", "location": "Riyadh",
            "market_min": 4000.0, "market_max": 6000.0, "currency": "SAR", "period": "monthly",
            "basis": "base_salary", "data_quality": "market",
            "contract_salary": {"amount": amount, "currency": "SAR", "period": "monthly"},
            "sources": [], "observations": [], "limitations": []}


def test_comparison_salary_by_contract_reuses_salary_view_per_contract():
    comparison = {"contracts": [
        _compared_contract("contract_1", "Offer A", [], salary_benchmark=_benchmark("success", 5000.0)),
        _compared_contract("contract_2", "Offer B", [], salary_benchmark=_benchmark("not_configured")),
    ]}
    blocks = {b["contract_id"]: b["salary"] for b in view.comparison_salary_by_contract(comparison)}
    assert blocks["contract_1"]["status"] == "success"
    assert blocks["contract_1"]["market_range"] == "4,000 - 6,000 SAR"
    assert blocks["contract_2"]["status"] == "not_configured"


# --------------------------------------------------------------------------- naming convention (Part 19)
def test_contract_naming_uses_job_title_and_keeps_the_full_name_as_secondary():
    comparison = {"contracts": [
        _compared_contract("contract_1", "data_engineer_contract_1.docx",
                           [], findings=[_finding("job_title", _fact("found", raw_value="Data Engineer"))]),
        _compared_contract("contract_2", "data_engineer_contract_3.docx",
                           [], findings=[_finding("job_title", _fact("not_found"))]),
    ]}
    names = view.comparison_contract_names(comparison)
    assert names["contract_1"]["short"] == "Contract A — Data Engineer"
    assert names["contract_1"]["full"] == "data_engineer_contract_1.docx"
    assert names["contract_2"]["short"] == "Contract B"  # no job title found - no fabrication
    assert names["contract_2"]["full"] == "data_engineer_contract_3.docx"


def test_comparison_contract_cards_carries_job_title_location_and_headline_pay():
    findings = [
        _finding("job_title", _fact("found", raw_value="Pharmacist")),
        _finding("work_location", _fact("found", raw_value="Jeddah")),
    ]
    comparison = {"contracts": [_compared_contract("contract_1", "shifa.pdf", [], findings=findings)]}
    [card] = view.comparison_contract_cards(comparison)
    assert card["short_name"] == "Contract A — Pharmacist"
    assert card["job_title"] == "Pharmacist"
    assert card["location"] == "Jeddah"


# --------------------------------------------------------------------------- no raw internal terms leak through
def test_comparison_compliance_blocks_never_expose_a_raw_finding_id():
    """Regression guard for the reported 'F13'-style exposure: nothing this module hands to the
    Compare Contracts compliance blocks carries a raw finding_id - only field/clause_name/status."""
    comparison = {"contracts": [
        _compared_contract("contract_1", "Offer A", [_clause("working_hours", "compliant", "Working hours")]),
    ]}
    blocks = view.comparison_compliance_by_contract(comparison)
    for block in blocks:
        for row in block["rows"]:
            assert "finding_id" not in row


# --------------------------------------------------------------------------- Salary Benchmark: no backend/env leak
def test_not_configured_salary_limitations_never_reach_the_general_notes_list():
    """Regression guard: 'not_configured' has its own clean explanation (salary.unavailable) on the
    Salary Benchmark card/tab. Its `limitations` text names raw backend environment variables
    (SANAD_SEARCH_PROVIDER, SANAD_SALARY_ENABLED) - that must never also surface in the general
    Notes/caveats list a user reads elsewhere on the page."""
    result = _result([], salary_benchmark={
        "status": "not_configured", "provider": "none",
        "message": "Salary benchmarking is not configured; no market salary data was used or estimated.",
        "limitations": ["Salary benchmarking is not configured; set SANAD_SEARCH_PROVIDER with an API key, or "
                        "SANAD_SALARY_ENABLED=1 to read the documented sources directly."],
    })
    notes = view.caveats(result)
    assert not any("SANAD_" in note for note in notes)


def test_other_salary_limitations_still_reach_the_notes_list():
    """A limitation that isn't about missing configuration (e.g. why no range could be built) is still
    a useful, human-readable caveat and should still show up."""
    result = _result([], salary_benchmark={
        "status": "insufficient_data", "provider": "web_search",
        "message": "No usable salary figure could be read from the sources that answered, so no market "
                   "range is reported.",
        "limitations": ["No source stated a salary with a currency and period."],
    })
    notes = view.caveats(result)
    assert "No source stated a salary with a currency and period." in notes


# --------------------------------------------------------------------------- Compare Contracts: compensation table
def _comp_finding(field: str, amount: float, currency: str = "SAR", period: str = "monthly") -> dict:
    return _finding(field, _fact("found", value={"amount": amount, "currency": currency, "period": period}))


def test_comparison_compensation_table_reads_each_contracts_own_compensation_summary():
    """Part 2 (Compare Contracts redesign): Compensation is one row per pay component, contracts as
    columns, read from each contract's own compensation_summary - never the backend's 'salary'/
    'benefits' comparison dimensions, which split/concatenate these awkwardly."""
    findings_a = [_comp_finding("salary", 10000), _comp_finding("housing_allowance", 2000),
                 _comp_finding("total_salary", 12000)]
    findings_b = [_comp_finding("salary", 15000), _comp_finding("total_salary", 15000)]
    comparison = {"contracts": [
        _compared_contract("contract_1", "Offer A", [], findings=findings_a),
        _compared_contract("contract_2", "Offer B", [], findings=findings_b),
    ]}
    rows = {r["field"]: r for r in view.comparison_compensation_table(comparison)}
    assert rows["salary"]["contract_1"] == "SAR 10,000/month"
    assert rows["total_salary"]["contract_1"] == "SAR 12,000/month"
    assert rows["housing_allowance"]["contract_1"] == "SAR 2,000/month"
    assert rows["housing_allowance"]["contract_2"] == "__not_found__"
    # basic salary is never conflated with total compensation
    assert rows["salary"]["contract_2"] == "SAR 15,000/month" and rows["total_salary"]["contract_2"] == "SAR 15,000/month"
    assert set(COMPENSATION_DETAIL_FIELDS_TESTED := {"salary", "housing_allowance", "transportation_allowance",
                                                     "other_allowances", "total_salary", "net_salary"}) == \
          set(rows.keys())


# --------------------------------------------------------------------------- Compare Contracts: quick-compare strip
def _dimension(label: str, values: dict[str, str], dimension_key: str | None = None) -> dict:
    return {"dimension": dimension_key or label.lower().replace(" ", "_"), "label": label, "ranking": "not_ranked",
            "values": [{"contract_id": cid, "display": v} for cid, v in values.items()],
            "best_contract_ids": [], "explanation": ""}


def test_comparison_quick_compare_pulls_compensation_leave_probation_and_benchmark():
    """Part 2: the top-of-page strip reuses the same functions the sections below call - no new
    computation - so its numbers can never disagree with what a reader finds further down the page."""
    comparison = {
        "contracts": [
            _compared_contract("contract_1", "Offer A", [], findings=[_comp_finding("total_salary", 12000)],
                               salary_benchmark=_benchmark("success", 12000)),
            _compared_contract("contract_2", "Offer B", [], findings=[_comp_finding("salary", 15000)],
                               salary_benchmark=_benchmark("not_configured")),
        ],
        "comparison": {
            "benefits": [_dimension("Annual leave", {"contract_1": "21 days", "contract_2": "30 days"})],
            "contract_terms": [_dimension("Probation period", {"contract_1": "90 days", "contract_2": "60 days"})],
        },
    }
    rows = {r["key"]: r for r in view.comparison_quick_compare(comparison)}
    assert rows["compensation"]["values"]["contract_1"] == "SAR 12,000/month"
    assert rows["compensation"]["values"]["contract_2"] == "SAR 15,000/month"  # falls back to basic salary
    assert rows["annual_leave"]["values"] == {"contract_1": "21 days", "contract_2": "30 days"}
    assert rows["probation"]["values"] == {"contract_1": "90 days", "contract_2": "60 days"}
    assert rows["salary_benchmark"]["values"]["contract_1"] == "4,000 - 6,000 SAR"
    assert rows["salary_benchmark"]["values"]["contract_2"] == "__not_configured__"


# --------------------------------------------------------------------------- Compare Contracts: risk cards
def test_risk_groups_groups_by_severity_highest_first_and_never_exposes_a_finding_id():
    comparison = {
        "contracts": [_compared_contract("contract_1", "Offer A", [])],
        "risks": [
            {"contract_id": "contract_1", "severity": "high", "code": "missing_clause", "field": "notice_period",
             "finding_id": "F13", "message": "Notice period is missing."},
            {"contract_id": "contract_1", "severity": "medium", "code": "ambiguous_value", "field": "salary",
             "finding_id": "F14", "message": "Salary is ambiguous."},
            {"contract_id": "contract_1", "severity": "medium", "code": "ambiguous_value", "field": "housing",
             "finding_id": "F15", "message": "Housing allowance is ambiguous."},
        ],
    }
    groups = view.risk_groups(comparison)
    assert [g["severity"] for g in groups] == ["high", "medium"]  # highest severity first, empty groups left out
    by_severity = {g["severity"]: g["items"] for g in groups}
    assert len(by_severity["high"]) == 1 and len(by_severity["medium"]) == 2
    for group in groups:
        for item in group["items"]:
            assert "finding_id" not in item
            assert "F13" not in str(item.values()) and "F14" not in str(item.values())
    assert by_severity["high"][0]["contract"] == "Contract A"  # naming convention, no job title extracted
