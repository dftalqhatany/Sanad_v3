"""Turns an OrchestratorResult JSON payload into rows the user interface renders.

Pure functions over the API payload: no decisions, no thresholds, no re-ranking, no analysis. Text
that the backend already wrote (explanations, caveats, recommendations) is passed through unchanged.
"""

from __future__ import annotations

from typing import Any

STATUS_LABELS = {
    "success": "Completed",
    "partial": "Completed with gaps",
    "insufficient_evidence": "Not enough evidence",
    "invalid_input": "Request rejected",
    "rag_error": "Regulatory search unavailable",
    "analysis_error": "Failed",
}
ROUTE_LABELS = {
    "contract_analysis": "Contract analysis",
    "cv_analysis": "CV analysis",
    "contract_comparison": "Contract comparison",
    "regulatory_question": "Regulatory question",
    "salary_benchmark": "Salary benchmarking",
    "none": "Not routed",
}
FINDING_LABELS = {
    "compliant": "Compliant", "non_compliant": "Non-compliant", "requires_review": "Requires review",
    "insufficient_evidence": "Insufficient evidence", "not_applicable": "Document fact",
    "not_found": "Not found in contract", "error": "Check failed",
}
SEVERITY_ORDER = {"high": 0, "medium": 1, "info": 2}

# ------------------------------------------------------------------ contract overview (Analyze Contract dashboard)
# Field name -> display label. Presentation only, same convention as FINDING_LABELS above: the
# service's own field name, made readable, never a translation of backend-written content.
OVERVIEW_FIELDS = (
    ("employee_name", "Employee Name"), ("employer_name", "Employer"), ("job_title", "Job Title"),
    ("work_location", "Work Location"), ("contract_type", "Contract Type"), ("start_date", "Start Date"),
    ("end_date", "End Date"), ("contract_duration", "Contract Duration"), ("working_days", "Working Days"),
    ("working_hours", "Daily Working Hours"), ("probation_status", "Probation Status"),
)
COMPENSATION_FIELDS = (
    ("salary", "Basic Salary"), ("housing_allowance", "Housing Allowance"),
    ("transportation_allowance", "Transportation"), ("other_allowances", "Other Allowances"),
    ("total_salary", "Total Compensation"),
)
COMPENSATION_DETAIL_FIELDS = COMPENSATION_FIELDS + (("net_salary", "Net Salary"),)
# Clause types that are about pay, not about a labour-law timing/hours/leave rule. Excluded from Legal
# Compliance so compensation content stays entirely inside Compensation/Salary Benchmark - the same
# separation the product requires between Salary Benchmark and Legal Compliance (see salary_view)
# extends to ordinary salary/allowance clauses: a below-range or qualitative pay clause must never
# read as a legal-compliance verdict.
_COMPENSATION_CLAUSE_TYPES = {"salary", "allowances", "benefits", "housing", "transportation", "health_insurance",
                             "overtime"}


def _number(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def _findings_by_field(analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {f.get("field"): f for f in analysis.get("findings", []) if f.get("field")}


def _has_encoding_corruption(text: str | None) -> bool:
    """The unmistakable signature of genuine encoding corruption: the Unicode replacement character,
    or a Private Use Area codepoint (U+E000-U+F8FF) - what a PDF's embedded font emits when it has no
    usable character map, so the extractor read raw glyph ids instead of text. Unlike letter-spacing
    (see `_looks_corrupted`), there is no readable text underneath this to fall back to, so a single
    occurrence disqualifies the text even from evidence - see `safe_evidence_text`."""
    return bool(text) and any(ch == "�" or "" <= ch <= "" for ch in text)


def _looks_corrupted(text: str | None) -> bool:
    """Two known PDF-extraction artifacts, either one enough to disqualify text from being shown as a
    primary value:

    1. Genuine encoding corruption - see `_has_encoding_corruption`.
    2. Arabic pulled out glyph-by-glyph (each letter or ligature its own whitespace-separated token,
       e.g. "إ ج مال ي") instead of as joined words - a heuristic, not a parser: at least four tokens,
       most of them Arabic, and most of THOSE one or two characters long (an ordinary Arabic word
       extracted correctly is rarely that short).

    Either way, a reader would see nonsense, so text like this must never be shown as a primary value
    anywhere in the interface. Case 2 can still appear in evidence/source-text, where the page and
    section provenance make it useful even though it reads oddly; case 1 cannot (see
    `safe_evidence_text`), since it is not "oddly formatted text" but literally undisplayable bytes.
    """
    if not text:
        return False
    if _has_encoding_corruption(text):
        return True
    tokens = text.split()
    if len(tokens) < 4:
        return False
    arabic_tokens = [tok for tok in tokens if any("؀" <= ch <= "ۿ" for ch in tok)]
    if len(arabic_tokens) < 4:
        return False
    short = [tok for tok in arabic_tokens if len(tok) <= 2]
    return len(short) / len(arabic_tokens) > 0.6


def safe_evidence_text(text: str | None) -> str | None:
    """A piece of source text (a contract quote, typically), safe to render in an evidence panel -
    or None if it has genuine encoding corruption (`_has_encoding_corruption`), in which case there is
    nothing readable to show, page reference or not. Letter-spaced-but-otherwise-valid Arabic (the
    other half of `_looks_corrupted`) is returned unchanged here: it reads oddly, but it is real text
    the contract actually contains - exactly what an evidence panel is for, unlike a primary value."""
    if text and _has_encoding_corruption(text):
        return None
    return text


def _overview_display(finding: dict[str, Any] | None) -> str:
    """A single overview value: the contract's own words when found, a neutral value otherwise.

    Never invents a value: a field the service did not find or could not extract is shown as
    "not found" (in the caller's own words, via i18n); a field the contract explicitly ruled out
    (not_applicable) is shown as such, since that is itself a fact the contract states, not a gap.

    Never shows a corrupted extraction artifact either (see _looks_corrupted): a raw or structured
    value that reads as garbled Arabic falls back to "ambiguous" rather than being shown as-is - the
    contract genuinely states something here, but not in a form Sanad can show as a primary value.
    """
    if finding is None:
        return "__not_found__"
    fact = finding.get("contract_fact", {})
    status = fact.get("extraction_status")
    if status == "found":
        raw = fact.get("raw_value")
        if raw and not _looks_corrupted(raw):
            return raw
        value = fact.get("value")
        if value is not None and not _looks_corrupted(str(value)):
            return str(value)
        return "__ambiguous__" if (raw or value is not None) else "__not_found__"
    if status == "ambiguous":
        return "__ambiguous__"
    if status == "not_applicable":
        return "__not_applicable__"
    return "__not_found__"


def contract_overview_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per Contract Information field, in the fixed order Sanad always shows them in."""
    analysis = contract_analysis(result)
    if not analysis:
        return []
    findings = _findings_by_field(analysis)
    return [{"field": field, "label": label, "display": _overview_display(findings.get(field))}
            for field, label in OVERVIEW_FIELDS]


def _legal_compliance_findings_from_analysis(analysis: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The shared implementation behind `legal_compliance_findings` (below) and Compare Contracts'
    per-contract compliance blocks (`comparison_compliance_by_contract`) - both read a single contract
    analysis's own `clause_findings` and nothing else, so a contract's compliance picture is identical
    whether it is opened on its own or as part of a comparison. See `legal_compliance_findings`'s
    docstring for why `clause_findings` is the only source ever allowed to supply a compliance verdict.
    """
    if not analysis:
        return []
    rows = []
    for finding in analysis.get("clause_findings", []):
        field = finding.get("field")
        if not field or finding.get("clause_type") in _COMPENSATION_CLAUSE_TYPES:
            continue
        fact = finding.get("contract_fact") or {}
        rule = finding.get("legal_rule") or {}
        contract_value = fact.get("raw_value")
        if contract_value and _looks_corrupted(contract_value):
            contract_value = None  # never shown as the primary value here either - see _looks_corrupted
        rows.append({
            "field": field,
            "clause_name": finding.get("clause_name") or field.replace("_", " ").title(),
            "status": finding.get("status", "not_found"),
            "explanation": finding.get("explanation", ""),
            "contract_value": contract_value or "-",
            "page": fact.get("page_number"),
            "article_number": rule.get("article_number"),
            "evidence": [
                {"citation": e.get("citation"), "arabic_text": e.get("arabic_text"), "english_text": e.get("english_text")}
                for e in finding.get("regulatory_evidence", [])
            ],
        })
    return rows


def legal_compliance_findings(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The ONE authoritative list behind every Legal Compliance display: Overview, the Legal
    Compliance tab and Findings & Recommendations all read this same list, so they can never disagree.

    Source: `analysis.clause_findings` only (Stage 3, agents.legal_rules.build_clause_legal_finding) -
    the sole place a COMPLIANT/NON_COMPLIANT verdict is allowed to come from (see the module docstring
    of agents/contract_analysis.py and the validator on models.analysis.AnalysisFinding.status). The
    field-level `analysis.findings` list is never read here: by design it may carry an interpreter's
    own compliant/non_compliant reading in its explanation text, but that reading is unconditionally
    demoted to "requires_review" before it reaches `status` - showing it as a compliance verdict would
    be exactly the contradiction ("explanation says compliant, status says needs attention") this
    function exists to prevent.

    One row per clause the deterministic engine actually evaluated against a bound contract fact
    (`field` is set); a clause the segmenter could not tie to any extracted value carries no
    compliance information and is left out, not guessed at. Pay-related clause types are left out too
    - see _COMPENSATION_CLAUSE_TYPES - so this list is legal compliance only, never compensation.
    """
    return _legal_compliance_findings_from_analysis(contract_analysis(result))


def legal_compliance_counts(result: dict[str, Any]) -> dict[str, int]:
    """Status counts for the compact Overview card - the same rows as the detail tab, just tallied."""
    counts: dict[str, int] = {}
    for row in legal_compliance_findings(result):
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts


def _component_display(value: Any) -> str | None:
    """One salary/allowance component, formatted as the contract states it. None if there is nothing to show.

    A component the contract names but states no figure for (no amount, no percentage) still shows
    that name rather than being skipped - the alternative would fall through to a raw-text fallback in
    `_field_row`, which for an Arabic table cell can come back corrupted (see `_looks_corrupted`); a
    name is always plain, human-readable text, so it is the safe thing to show.
    """
    if value is None:
        return None
    items = value if isinstance(value, list) else [value]
    parts = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if item.get("amount") is not None:
            currency = item.get("currency") or ""
            period = {"monthly": "/month", "annual": "/year"}.get(item.get("period"), "")
            figure = f"{currency} {_number(item['amount'])}{period}".strip()
            text = f"{name}: {figure}" if name and len(items) > 1 else figure
        elif item.get("percentage") is not None:
            basis = f" of {item['percentage_basis']}" if item.get("percentage_basis") else ""
            figure = f"{_number(item['percentage'])}%{basis}"
            text = f"{name}: {figure}" if name and len(items) > 1 else figure
        elif name:
            text = name
        else:
            continue
        parts.append(text)
    return "; ".join(parts) if parts else None


def _field_row(field: str, label: str, finding: dict[str, Any] | None) -> dict[str, Any]:
    fact = (finding or {}).get("contract_fact", {})
    status = fact.get("extraction_status")
    if status == "found":
        display = _component_display(fact.get("value"))
        if display is None:
            raw = fact.get("raw_value")
            # A raw-text fallback is only trustworthy when it reads as ordinary text. A known
            # PDF-extraction artifact (Arabic pulled out letter-by-letter - see _looks_corrupted) must
            # never be shown as the primary value: the contract does state something here, it's just
            # not shown as this component's headline figure - it stays available in evidence via
            # source_text/page below, exactly as an ordinary "ambiguous" component would be.
            if raw and not _looks_corrupted(raw):
                display = raw
            elif raw:
                display = "__ambiguous__"
            else:
                display = "__not_found__"
    elif status == "ambiguous":
        display = "__ambiguous__"
    elif status == "not_applicable":
        display = "__not_applicable__"
    else:
        display = "__not_found__"
    return {"field": field, "label": label, "display": display,
            "page": fact.get("page_number"), "source_text": fact.get("source_text")}


def compensation_summary(result: dict[str, Any], *, detailed: bool = False) -> list[dict[str, Any]]:
    """The Compensation card/tab: each salary component kept separate, never added up here.

    A component the contract does not state is left out rather than shown as zero or estimated -
    `stated_pay`-style addition belongs to Compare Contracts, which already does it explicitly with
    its own caveats; this view only reports what each individual field says.

    Some short-form contracts (e.g. an appointment letter) state a single take-home figure rather than
    a separate basic salary - extracted into its own schema field, `in_hand_salary`, not `salary`.
    When the contract has no basic salary but does have that figure, the "Basic Salary" row is
    relabelled to say so and shows it, instead of a plain "not found" for a number the contract
    actually states; the two are never both shown as if they were the same thing, and nothing is
    invented when neither is present.
    """
    analysis = contract_analysis(result)
    if not analysis:
        return []
    findings = _findings_by_field(analysis)
    fields = COMPENSATION_DETAIL_FIELDS if detailed else COMPENSATION_FIELDS
    rows = [_field_row(field, label, findings.get(field)) for field, label in fields]
    for row in rows:
        if row["field"] == "salary" and row["display"] == "__not_found__":
            in_hand = _field_row("in_hand_salary", "Salary (in-hand)", findings.get("in_hand_salary"))
            if in_hand["display"] != "__not_found__":
                # Keep this row's identity as the "salary" slot (so callers keyed on `field` still find
                # it there); only its label/value/provenance change to show the in-hand figure instead.
                row["label"], row["display"] = in_hand["label"], in_hand["display"]
                row["page"], row["source_text"] = in_hand["page"], in_hand["source_text"]
    return rows


def findings_summary(result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Findings tab: Key findings / Requires review are a strict split of `legal_compliance_findings`
    (never the field-level `findings`), so a field can never land in both - see that function's
    docstring for why field-level findings are never a compliance source. "Missing information" reads
    field-level NOT_FOUND, which is safe: a field-level finding is NOT_FOUND only when the field itself
    is genuinely absent from the extraction, never as a stand-in for a compliance verdict.
    """
    analysis = contract_analysis(result)
    if not analysis:
        return {"key_findings": [], "requires_review": [], "missing_information": []}
    compliance_rows = legal_compliance_findings(result)
    key_findings = [row for row in compliance_rows if row["status"] in ("compliant", "non_compliant")]
    requires_review = [row for row in compliance_rows if row["status"] in ("requires_review", "ambiguous")]
    missing_information = [
        {"field": f.get("field"), "explanation": f.get("explanation", "")}
        for f in analysis.get("findings", []) if f.get("status") == "not_found"
    ]
    return {"key_findings": key_findings, "requires_review": requires_review,
            "missing_information": missing_information}


# ------------------------------------------------------------------ comparison, grouped for display
# Display order for the grouped Compare Contracts view. "salary_benchmark" is split out of the
# service's own "salary" category (which otherwise also holds basic/stated pay) into its own group,
# since Sanad shows salary benchmarking as its own result, separate from - and never folded into -
# basic/stated pay or legal compliance.
COMPARISON_GROUP_ORDER = ("compliance", "salary", "salary_benchmark", "benefits", "working_conditions",
                          "contract_terms", "cv_compatibility")


def comparison_sections(comparison: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """The same rows `comparison_rows` builds, grouped by category for a sectioned display.

    Pure regrouping of what the service already computed - no new values, no re-ranking, nothing a
    single flat `comparison_rows()` table does not already contain.
    """
    names = {c["contract_id"]: c.get("label") or c["contract_id"] for c in comparison.get("contracts", [])}
    sections: dict[str, list[dict[str, Any]]] = {key: [] for key in COMPARISON_GROUP_ORDER}
    for section, dimensions in (comparison.get("comparison") or {}).items():
        for dimension in dimensions:
            row = _dimension_row(names, section, dimension)
            group = "salary_benchmark" if dimension.get("dimension") == "salary_vs_market" else section
            sections.setdefault(group, []).append(row)
    return sections


def comparison_compliance_by_contract(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare Contracts' Legal Compliance section, one block per contract: that contract's own status
    counts and rows, from its own embedded `contract_analysis.clause_findings` - the exact same
    authoritative source and shared helper (`_legal_compliance_findings_from_analysis`) as the
    single-contract Legal Compliance tab, so a contract's compliance never reads differently in
    Compare Contracts than it does opened on its own.

    Deliberately NOT derived from the comparison agent's own "compliance" dimension: that dimension is
    scored from each contract's field-level `findings`, which can never carry a compliant/non_compliant
    verdict (see `legal_compliance_findings`), so it always undercounts non-compliant clauses. Reading
    clause_findings directly here, per contract, avoids that gap without changing the comparison agent.
    """
    blocks = []
    for entry in comparison.get("contracts", []):
        rows = _legal_compliance_findings_from_analysis(entry.get("contract_analysis"))
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        blocks.append({
            "contract_id": entry.get("contract_id"),
            "label": entry.get("label") or entry.get("filename") or entry.get("contract_id"),
            "counts": counts,
            "rows": rows,
        })
    return blocks


def comparison_compliance_differences(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The plain-language "key differences" list: fields where the compared contracts do not all land
    on the same compliance status, built only from `comparison_compliance_by_contract`'s own rows (so
    still clause_findings only, never a new computation). A field every contract shares the same status
    for is not a difference and is left out."""
    by_field: dict[str, dict[str, dict[str, Any]]] = {}
    for block in blocks:
        for row in block["rows"]:
            by_field.setdefault(row["field"], {})[block["contract_id"]] = {
                "contract_id": block["contract_id"], "label": block["label"], "status": row["status"],
                "clause_name": row["clause_name"],
            }
    differences = []
    for field, by_contract in by_field.items():
        statuses = {v["status"] for v in by_contract.values()}
        if len(statuses) > 1:
            differences.append({
                "field": field,
                "clause_name": next(iter(by_contract.values()))["clause_name"],
                "per_contract": [{"contract_id": v["contract_id"], "label": v["label"], "status": v["status"]}
                                for v in by_contract.values()],
            })
    return differences


def comparison_salary_by_contract(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare Contracts' Salary Benchmark section, one block per contract: reuses `salary_view` on
    that contract's own embedded analysis (wrapped in the same shape a single-contract result already
    has) rather than re-deriving anything - the same figures the Salary Benchmark tab would show for
    that contract analysed on its own."""
    blocks = []
    for entry in comparison.get("contracts", []):
        analysis = entry.get("contract_analysis") or {}
        salary = salary_view({"analysis": {"contract_analysis": analysis}})
        blocks.append({
            "contract_id": entry.get("contract_id"),
            "label": entry.get("label") or entry.get("filename") or entry.get("contract_id"),
            "salary": salary,
        })
    return blocks


def comparison_compensation_table(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare Contracts' Compensation section: one row per pay component (Basic Salary, Housing
    Allowance, Transportation, Other Allowances, Total Compensation, Net Salary), contracts as columns
    - built by calling `compensation_summary` (detailed) on each contract's own embedded analysis, the
    exact same figures the single-contract Compensation tab shows for that contract. Basic salary and
    total compensation are always separate rows here, never merged into one figure or compared against
    each other - the same guarantee `compensation_summary` already gives a single contract.

    Each row is keyed by contract_id (not by display name), so a caller maps to short/full names itself
    - see `comparison_contract_names`.
    """
    per_contract: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in comparison.get("contracts", []):
        cid = entry.get("contract_id")
        pseudo_result = {"analysis": {"contract_analysis": entry.get("contract_analysis") or {}}}
        per_contract[cid] = {row["field"]: row for row in compensation_summary(pseudo_result, detailed=True)}
    contract_ids = [entry.get("contract_id") for entry in comparison.get("contracts", [])]
    rows = []
    for field, label in COMPENSATION_DETAIL_FIELDS:
        row: dict[str, Any] = {"field": field, "label": label}
        for cid in contract_ids:
            found = per_contract.get(cid, {}).get(field)
            row[cid] = found["display"] if found else "__not_found__"
        rows.append(row)
    return rows


def comparison_quick_compare(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """The top-of-page quick-compare strip: Compensation / Annual Leave / Probation / Salary Benchmark,
    each contract's value side by side - so a reader sees the headline differences before reaching any
    section below. Every value here is read from a function a section below also calls (never a new
    computation): `comparison_compensation_table` for pay, `comparison_sections` for the existing
    annual-leave/probation comparison dimensions, and `comparison_salary_by_contract` for the benchmark.
    """
    contract_ids = [entry.get("contract_id") for entry in comparison.get("contracts", [])]
    if not contract_ids:
        return []

    comp_rows = {row["field"]: row for row in comparison_compensation_table(comparison)}
    total_row, basic_row = comp_rows.get("total_salary"), comp_rows.get("salary")

    def pay_value(cid: str) -> str:
        if total_row and total_row.get(cid) != "__not_found__":
            return total_row[cid]
        if basic_row:
            return basic_row.get(cid, "__not_found__")
        return "__not_found__"

    sections = comparison_sections(comparison)
    full_name_to_id = {(entry.get("label") or entry.get("filename") or entry.get("contract_id")): entry.get("contract_id")
                       for entry in comparison.get("contracts", [])}

    def dimension_value(group: str, dimension_label: str, cid: str) -> str:
        for row in sections.get(group, []):
            if row.get("Dimension") == dimension_label:
                for full_name, value in row.items():
                    if full_name_to_id.get(full_name) == cid and value not in (None, ""):
                        return value
        return "__not_found__"

    salary_blocks = {block["contract_id"]: block["salary"] for block in comparison_salary_by_contract(comparison)}

    def benchmark_value(cid: str) -> str:
        salary = salary_blocks.get(cid)
        if not salary:
            return "__not_found__"
        if salary["status"] == "not_configured":
            return "__not_configured__"
        if salary["market_range"] != "no market data":
            return salary["market_range"]
        return "__no_range__"

    return [
        {"key": "compensation", "label": "Compensation", "values": {cid: pay_value(cid) for cid in contract_ids}},
        {"key": "annual_leave", "label": "Annual Leave",
         "values": {cid: dimension_value("benefits", "Annual leave", cid) for cid in contract_ids}},
        {"key": "probation", "label": "Probation",
         "values": {cid: dimension_value("contract_terms", "Probation period", cid) for cid in contract_ids}},
        {"key": "salary_benchmark", "label": "Salary Benchmark",
         "values": {cid: benchmark_value(cid) for cid in contract_ids}},
    ]


def risk_groups(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Risks and review items, grouped by severity for a card layout instead of one flat table -
    highest severity first. Each item keeps contract/field/message in plain words; the internal
    finding_id is never included (see `risk_rows`, which already leaves it out of its own default
    columns - this reads the same underlying `comparison['risks']`, independently, for grouping)."""
    names = {c["contract_id"]: c.get("label") or c["contract_id"] for c in comparison.get("contracts", [])}
    contract_names = comparison_contract_names(comparison)
    order = ("high", "medium", "info")
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in order}
    for risk in comparison.get("risks", []):
        severity = risk.get("severity") if risk.get("severity") in grouped else "info"
        cid = risk.get("contract_id")
        grouped[severity].append({
            "contract": contract_names.get(cid, {}).get("short") or names.get(cid, cid),
            "type": (risk.get("code") or "").replace("_", " "),
            "field": (risk.get("field") or "-").replace("_", " "),
            "message": risk.get("message"),
        })
    return [{"severity": key, "items": grouped[key]} for key in order if grouped[key]]


_NAME_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def comparison_contract_names(comparison: dict[str, Any]) -> dict[str, dict[str, str]]:
    """The 'Contract A - <job title>' naming convention: short, safe for a table header or card title,
    built from that contract's own extracted job title (never invented). The contract's own label or
    filename is kept alongside as the full name - never lost, just no longer the only name shown."""
    names: dict[str, dict[str, str]] = {}
    for index, entry in enumerate(comparison.get("contracts", [])):
        findings = _findings_by_field(entry.get("contract_analysis") or {})
        job_title = _overview_display(findings.get("job_title"))
        letter = _NAME_LETTERS[index] if index < len(_NAME_LETTERS) else str(index + 1)
        short = (f"Contract {letter}" if job_title in ("__not_found__", "__ambiguous__", "__not_applicable__")
                else f"Contract {letter} — {job_title}")
        names[entry.get("contract_id")] = {
            "short": short,
            "full": entry.get("label") or entry.get("filename") or entry.get("contract_id"),
        }
    return names


def comparison_contract_cards(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare Contracts' summary cards: name, job title, location and headline pay for each compared
    contract, all read from the same functions (`contract_overview_rows`, `compensation_summary`) the
    single-contract Overview tab already uses - no new fields, no new computation, just applied once
    per contract instead of once per whole result."""
    names = comparison_contract_names(comparison)
    cards = []
    for entry in comparison.get("contracts", []):
        cid = entry.get("contract_id")
        pseudo_result = {"analysis": {"contract_analysis": entry.get("contract_analysis") or {}}}
        overview_by_field = {row["field"]: row["display"] for row in contract_overview_rows(pseudo_result)}
        comp_by_field = {row["field"]: row for row in compensation_summary(pseudo_result)}
        headline = comp_by_field.get("total_salary")
        if not headline or headline["display"] == "__not_found__":
            headline = comp_by_field.get("salary")
        info = names.get(cid, {})
        cards.append({
            "contract_id": cid,
            "short_name": info.get("short", cid),
            "full_name": info.get("full", cid),
            "job_title": overview_by_field.get("job_title", "__not_found__"),
            "location": overview_by_field.get("work_location", "__not_found__"),
            "salary_label": headline["label"] if headline else None,
            "salary_display": headline["display"] if headline else "__not_found__",
        })
    return cards


def overview(result: dict[str, Any]) -> dict[str, Any]:
    routing = result.get("routing", {})
    return {
        "status": result.get("status"),
        "status_label": STATUS_LABELS.get(result.get("status", ""), result.get("status", "")),
        "route": routing.get("route"),
        "route_label": ROUTE_LABELS.get(routing.get("route", ""), routing.get("route", "")),
        "agent": routing.get("agent"),
        "rule": routing.get("rule"),
        "reason": routing.get("reason"),
        "summary": result.get("summary", ""),
        "warnings": result.get("warnings", []),
        "errors": [f"{e.get('code')}: {e.get('message')}" for e in result.get("errors", [])],
        "disclaimer": result.get("disclaimer", ""),
    }


def document_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "File": document.get("filename"),
            "Role": (document.get("role") or "not identified"),
            "Role from": document.get("role_source"),
            "Document": document.get("document_status"),
            "Extraction": document.get("extraction_status") or "-",
            "Pages": document.get("page_count") or "-",
            "Usable": "yes" if document.get("usable") else "no",
            "Notes": " ".join(document.get("notes", []) + [e.get("message", "") for e in document.get("errors", [])]),
        }
        for document in result.get("documents", [])
    ]


# --------------------------------------------------------------------------- single analysis
def contract_analysis(result: dict[str, Any]) -> dict[str, Any] | None:
    analysis = (result.get("analysis") or {}).get("contract_analysis")
    return analysis or None


def cv_analysis(result: dict[str, Any]) -> dict[str, Any] | None:
    return (result.get("analysis") or {}).get("cv_analysis") or None


def finding_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for finding in analysis.get("findings", []):
        fact = finding.get("contract_fact", {})
        rows.append({
            "Finding": finding.get("finding_id"),
            "Field": finding.get("field", "").replace("_", " "),
            "Status": FINDING_LABELS.get(finding.get("status", ""), finding.get("status")),
            "Contract says": fact.get("raw_value") or "-",
            "Page": fact.get("page_number") or "-",
            "Articles": ", ".join(str(e.get("article_number")) for e in finding.get("regulatory_evidence", [])
                                  if e.get("article_number")) or "-",
            "Explanation": finding.get("explanation", ""),
            "Source text": fact.get("source_text") or "",
        })
    return rows


def evidence_items(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Every retrieved article, with its citation and both texts (Arabic is the legal reference)."""
    return [
        {
            "evidence_id": item.get("evidence_id"),
            "citation": item.get("citation"),
            "article_number": item.get("article_number"),
            "article_name": item.get("article_name"),
            "part": item.get("part"),
            "chapter": item.get("chapter"),
            "arabic_text": item.get("arabic_text"),
            "english_text": item.get("english_text"),
            "rank": item.get("rank"),
            "score": item.get("score"),
            "source": item.get("source"),
        }
        for item in analysis.get("evidence", [])
    ]


def compatibility_rows(cv: dict[str, Any] | None) -> list[dict[str, Any]]:
    compatibility = (cv or {}).get("job_compatibility")
    if not compatibility:
        return []
    return [
        {
            "Requirement": requirement.get("requirement"),
            "Type": requirement.get("requirement_type", "").replace("_", " "),
            "Importance": requirement.get("importance"),
            "Status": requirement.get("status", "").replace("_", " "),
            "Explanation": requirement.get("explanation"),
            "CV evidence": "; ".join(e.get("text", "") for e in requirement.get("cv_evidence", [])),
        }
        for requirement in compatibility.get("requirements", [])
    ]


# --------------------------------------------------------------------------- comparison
def _dimension_row(names: dict[str, str], section: str, dimension: dict[str, Any]) -> dict[str, Any]:
    row = {"Category": section.replace("_", " "), "Dimension": dimension.get("label"),
           "Ranked": {"higher_is_better": "higher is better", "lower_is_better": "lower is better",
                      "not_ranked": "not ranked"}.get(dimension.get("ranking"), "")}
    for value in dimension.get("values", []):
        row[names.get(value["contract_id"], value["contract_id"])] = value.get("display", "")
    row["Leads"] = ", ".join(names.get(i, i) for i in dimension.get("best_contract_ids", [])) or "-"
    row["Explanation"] = dimension.get("explanation", "")
    return row


def comparison_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per compared dimension, one column per contract, plus who leads and why."""
    names = {c["contract_id"]: c.get("label") or c["contract_id"] for c in comparison.get("contracts", [])}
    return [_dimension_row(names, section, dimension)
            for section, dimensions in (comparison.get("comparison") or {}).items()
            for dimension in dimensions]


def recommendation_view(comparison: dict[str, Any]) -> dict[str, Any]:
    recommendation = comparison.get("recommendation") or {}
    names = {c["contract_id"]: c.get("label") or c["contract_id"] for c in comparison.get("contracts", [])}
    preferred = recommendation.get("preferred_contract_id")
    return {
        "status": recommendation.get("status"),
        "preferred": names.get(preferred, preferred),
        "method": recommendation.get("method"),
        "explanation": recommendation.get("explanation", ""),
        "factors": [
            {
                "Dimension": factor.get("label"),
                "Favours": ", ".join(names.get(i, i) for i in factor.get("favours", [])),
                "Why": factor.get("explanation"),
                "Evidence": "; ".join(f"{names.get(v['contract_id'], v['contract_id'])}: {v.get('display', '')}"
                                      for v in factor.get("evidence", [])),
            }
            for factor in recommendation.get("decision_factors", [])
        ],
        "trade_offs": recommendation.get("trade_offs", []),
        "caveats": recommendation.get("caveats", []),
        "basis": recommendation.get("basis", ""),
    }


def risk_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    names = {c["contract_id"]: c.get("label") or c["contract_id"] for c in comparison.get("contracts", [])}
    risks = sorted(comparison.get("risks", []), key=lambda r: (SEVERITY_ORDER.get(r.get("severity", "info"), 3),
                                                               r.get("contract_id", "")))
    return [
        {
            "Contract": names.get(risk.get("contract_id"), risk.get("contract_id")),
            "Severity": risk.get("severity"),
            "Type": risk.get("code", "").replace("_", " "),
            "Field": (risk.get("field") or "-").replace("_", " "),
            "Finding": risk.get("finding_id") or "-",
            "Message": risk.get("message"),
        }
        for risk in risks
    ]


# --------------------------------------------------------------------------- salary and questions
def salary_view(result: dict[str, Any]) -> dict[str, Any] | None:
    benchmark = result.get("salary_benchmark")
    if benchmark is None:
        analysis = contract_analysis(result)
        benchmark = (analysis or {}).get("salary_benchmark")
    if not benchmark:
        return None
    salary = benchmark.get("contract_salary") or {}
    has_range = benchmark.get("market_min") is not None and benchmark.get("market_max") is not None
    basis = {"base_salary": "base salary", "total_compensation": "total compensation",
             "unspecified": "pay of an unstated kind", None: ""}.get(benchmark.get("basis"), "")
    return {
        "status": benchmark.get("status"),
        "message": benchmark.get("message"),
        "contract_salary": (f"{salary.get('amount'):,.0f} {salary.get('currency') or ''}".strip()
                            if salary.get("amount") is not None else "not found in the contract"),
        "job_title": benchmark.get("job_title") or "-",
        "location": benchmark.get("location") or "-",
        "market_range": (f"{benchmark['market_min']:,.0f} - {benchmark['market_max']:,.0f} "
                         f"{benchmark.get('currency') or ''}".strip() if has_range else "no market data"),
        "period": benchmark.get("period") or "-",
        "basis": basis,
        "data_quality": benchmark.get("data_quality") or "-",
        "query": benchmark.get("query"),
        "limitations": list(benchmark.get("limitations", [])),
        "position": _salary_position(salary.get("amount"), benchmark) if has_range else None,
        "sources": [{"name": s.get("name"), "url": s.get("url"), "type": s.get("source_type"),
                     "tier": s.get("tier") or "-", "retrieved_at": s.get("retrieved_at") or "-"}
                    for s in benchmark.get("sources", [])],
        "evidence": [
            {
                "Source": observation.get("source_name"),
                "Tier": observation.get("tier"),
                "Figure": (f"{observation['minimum']:,.0f}" if observation["minimum"] == observation["maximum"]
                           else f"{observation['minimum']:,.0f} - {observation['maximum']:,.0f}")
                          + f" {observation.get('currency', '')} / {observation.get('period', '')}",
                "Monthly SAR": (f"{observation['monthly_min']:,.0f}"
                                if observation["monthly_min"] == observation["monthly_max"]
                                else f"{observation['monthly_min']:,.0f} - {observation['monthly_max']:,.0f}"),
                "Basis": observation.get("basis", "").replace("_", " "),
                "Used": "yes" if observation.get("usable") else "no",
                "Retrieved": observation.get("retrieved_at"),
                "Quote": observation.get("quote", "")[:300],
                "URL": observation.get("url"),
            }
            for observation in benchmark.get("observations", [])
        ],
    }


def _salary_position(amount, benchmark: dict[str, Any]) -> str | None:
    if amount is None:
        return None
    if amount < benchmark["market_min"]:
        return "below the observed market range"
    if amount > benchmark["market_max"]:
        return "above the observed market range"
    return "inside the observed market range"


def regulatory_answer(result: dict[str, Any]) -> dict[str, Any] | None:
    answer = result.get("regulatory_answer")
    if not answer:
        return None
    return {
        "answer": answer.get("answer"),
        "message": answer.get("message"),
        "status": answer.get("status"),
        "evidence": [
            {
                "citation": item.get("citation"),
                "article_number": (item.get("reference") or {}).get("article_number"),
                "arabic_text": item.get("arabic_content"),
                "english_text": item.get("english_content"),
                "score": item.get("score"),
                # a SIMILARITY score (how closely the article matches the question), never a claim
                # about how correct the answer is
                "similarity_percentage": item.get("similarity_percentage"),
            }
            for item in answer.get("evidence", [])
        ],
        "warnings": answer.get("warnings", []),
        "errors": [f"{e.get('code')}: {e.get('message')}" for e in answer.get("errors", [])],
    }


def caveats(result: dict[str, Any]) -> list[str]:
    """Everything the backend flagged as a limitation, in one list for the user interface."""
    items = list(result.get("warnings", []))
    analysis = contract_analysis(result)
    if analysis:
        items += analysis.get("warnings", [])
    cv = cv_analysis(result)
    if cv:
        items += cv.get("warnings", [])
    comparison = result.get("comparison")
    if comparison:
        items += comparison.get("warnings", [])
        items += (comparison.get("recommendation") or {}).get("caveats", [])
    benchmark = result.get("salary_benchmark") or (contract_analysis(result) or {}).get("salary_benchmark")
    if benchmark and benchmark.get("status") != "not_configured":
        # "not_configured" already has its own clean explanation on the Salary Benchmark card/tab
        # (see salary.unavailable); its `limitations` text names backend environment variables
        # (SANAD_SEARCH_PROVIDER, SANAD_SALARY_ENABLED) that must never surface in the general
        # Notes list a user reads on Demo Day.
        items += benchmark.get("limitations", [])
    return list(dict.fromkeys(item for item in items if item))
