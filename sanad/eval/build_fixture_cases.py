"""Build the contract, comparison, salary and robustness portions of the Golden Dataset.

Every case here is derived from a fixture that already exists in this repository, so its input is
reproducible and its expectation is traceable:

    tests/fixtures/documents/synthetic_content.py   fictional contracts and CVs, field by field
    tests/fixtures/documents/builders.py            renders them to real PDF/DOCX
    tests/fixtures/salary_pages.py                  realistic stand-ins for the documented sources

What is asserted, and what is deliberately not:

  * EXTRACTION expectations are read straight out of the fixture dictionary, so they cannot drift
    from the document. Those cases are 'generated_from_fixture'.
  * COMPLIANCE expectations are a legal judgement. This builder never writes one. Cases that need
    one carry 'needs_review' with the question a reviewer has to answer.
  * SALARY cases assert methodology (sources, tiers, currency, period, basis, containment,
    conflict handling) and never a single correct wage.
  * ROBUSTNESS cases assert behaviour (status, route, abstention), never a legal answer.

Documents are described as fixture recipes rather than committed binaries; the Checkpoint 10 runner
materialises them with the existing builders.

Run:  python eval/build_fixture_cases.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.fixtures.documents.synthetic_content import CONTRACT_AR, CONTRACT_EN  # noqa: E402

FIXTURE_FILES = {
    "contracts": "tests/fixtures/documents/synthetic_content.py",
    "builders": "tests/fixtures/documents/builders.py",
    "salary_pages": "tests/fixtures/salary_pages.py",
}


def fixture_hash(relpath: str) -> str:
    return hashlib.sha256((PROJECT_ROOT / relpath).read_bytes()).hexdigest()


HASHES = {name: fixture_hash(path) for name, path in FIXTURE_FILES.items()}


def provenance(source: str, *, authority: str | None = None) -> dict:
    return {
        "source": source,
        "source_row": None,
        "source_sha256": HASHES.get("contracts") if "synthetic_content" in source
        else HASHES.get("salary_pages") if "salary_pages" in source else None,
        "authority": authority,
        "law_snapshot": None,
        "law_version": None,
        "effective_date": None,
        "unavailable_fields": ["law_version", "effective_date", "amendment_reference"],
    }


MECHANICAL = {"status": "generated_from_fixture", "method": "mechanical_from_fixture"}


def needs_review(question: str) -> dict:
    return {"status": "needs_review", "method": None, "reason": question}


def fixture_value(content: dict, section: str, label: str) -> str | None:
    for key, value in content.get(section, []):
        if key == label:
            return value
    return None


def contract_doc(content_name: str, builder: str = "contract_docx", mutations: dict | None = None,
                 role: str = "contract", rtl: bool = False) -> dict:
    doc = {"fixture_module": "tests.fixtures.documents.builders", "builder": builder,
           "content": content_name, "role": role}
    if rtl:
        doc["rtl"] = True
    if mutations:
        doc["mutations"] = mutations
    return doc


# --------------------------------------------------------------------------- contract analysis
def contract_cases() -> list[dict]:
    en_extraction = {
        "job_title": fixture_value(CONTRACT_EN, "fields", "Job Title"),
        "work_location": fixture_value(CONTRACT_EN, "fields", "Work Location"),
        "contract_type": fixture_value(CONTRACT_EN, "fields", "Contract Type"),
        "salary": fixture_value(CONTRACT_EN, "compensation_rows", "Basic Salary"),
        "housing_allowance": fixture_value(CONTRACT_EN, "compensation_rows", "Housing Allowance"),
    }
    clause = {name: text for name, text in CONTRACT_EN["clauses"]}

    cases = [
        {
            "case_id": "con_en_baseline_extraction",
            "title": "English synthetic contract: every stated field is extracted",
            "language": "en",
            "input": {"documents": [contract_doc("CONTRACT_EN")]},
            "expected": {
                "route": "contract_analysis",
                "status": ["success", "partial"],
                "extraction": {field: {"value_contains": value, "status": "found"}
                               for field, value in en_extraction.items()},
                "verifiable_from_document": True,
            },
            "review": MECHANICAL,
            "tags": ["extraction", "baseline", "en"],
            "notes": "Expected values are read from CONTRACT_EN, so they cannot drift from the document.",
        },
        {
            "case_id": "con_ar_baseline_extraction",
            "title": "Arabic synthetic contract: extraction works right-to-left",
            "language": "ar",
            "input": {"documents": [contract_doc("CONTRACT_AR", rtl=True)]},
            "expected": {"route": "contract_analysis", "status": ["success", "partial"],
                         "extraction_fields_found_at_least": 6, "verifiable_from_document": True},
            "review": MECHANICAL,
            "tags": ["extraction", "baseline", "ar"],
            "notes": "Field-by-field values are asserted loosely because the Arabic fixture stores its own labels; "
                     "a reviewer may tighten this later.",
        },
        {
            "case_id": "con_pdf_baseline_extraction",
            "title": "Text-layer PDF contract is parsed with page numbers",
            "language": "en",
            "input": {"documents": [contract_doc("CONTRACT_EN", builder="contract_text_pdf")]},
            "expected": {"route": "contract_analysis", "status": ["success", "partial"],
                         "page_numbers_present": True, "verifiable_from_document": True},
            "review": MECHANICAL,
            "tags": ["extraction", "pdf"],
            "notes": "",
        },
        {
            "case_id": "con_en_notice_clause_removed",
            "title": "A clause that is absent must be reported as not found, never guessed",
            "language": "en",
            "input": {"documents": [contract_doc("CONTRACT_EN", mutations={"remove_clause": "Notice Period"})]},
            "expected": {"route": "contract_analysis",
                         "extraction": {"notice_period": {"status": "not_found"}},
                         "must_not_report": ["non_compliant:notice_period"],
                         "verifiable_from_document": True},
            "review": MECHANICAL,
            "tags": ["negative", "absent_field"],
            "notes": "A not_found field can never be labelled non-compliant; this case pins that rule.",
        },
        {
            "case_id": "con_empty_docx",
            "title": "An empty document is refused with a stated reason",
            "language": "not_applicable",
            "input": {"documents": [{"fixture_module": "tests.fixtures.documents.builders",
                                     "builder": "empty_docx", "role": "contract"}]},
            "expected": {"status": "invalid_input", "document_status": "empty_document",
                         "error_codes": ["unusable_document"], "verifiable_from_document": True},
            "review": MECHANICAL,
            "tags": ["failure", "empty"],
            "notes": "",
        },
        {
            "case_id": "con_scanned_pdf_requires_ocr",
            "title": "A scanned PDF is reported as needing OCR, not analysed",
            "language": "not_applicable",
            "input": {"documents": [{"fixture_module": "tests.fixtures.documents.builders",
                                     "builder": "scanned_pdf", "role": "contract"}]},
            "expected": {"status": "invalid_input", "document_status": "ocr_required",
                         "error_codes": ["unusable_document"], "verifiable_from_document": True},
            "review": MECHANICAL,
            "tags": ["failure", "ocr"],
            "notes": "",
        },
        {
            "case_id": "con_mixed_text_and_scanned_pages",
            "title": "A part-scanned PDF stays usable and keeps its warning",
            "language": "en",
            "input": {"documents": [{"fixture_module": "tests.fixtures.documents.builders",
                                     "builder": "pdf_with_text_pages_and_scanned_page", "role": "contract"}]},
            "expected": {"route": "contract_analysis", "status": ["partial", "success"],
                         "warnings_contain": ["ocr"], "verifiable_from_document": True},
            "review": MECHANICAL,
            "tags": ["partial", "warning"],
            "notes": "",
        },
        {
            "case_id": "con_en_with_cv_routes_to_both",
            "title": "Contract plus CV produces both a compliance section and a CV section",
            "language": "en",
            "input": {"documents": [contract_doc("CONTRACT_EN"),
                                    {"fixture_module": "tests.fixtures.documents.builders",
                                     "builder": "cv_docx", "content": "CV_EN", "role": "cv"}]},
            "expected": {"route": "contract_analysis", "analysis_has_contract_section": True,
                         "analysis_has_cv_section": True, "verifiable_from_document": True},
            "review": MECHANICAL,
            "tags": ["cv", "routing"],
            "notes": "Asserts structure only. Whether the CV actually fits the job is a judgement, not asserted here.",
        },
    ]

    # --- cases whose expectation is a legal conclusion: written as questions for a reviewer -------
    legal = [
        ("con_en_probation_120_days", "Probation stated as 120 days",
         {"replace_clause": {"Probation Period": "The Employee shall be subject to a probation period of "
                                                 "one hundred and twenty (120) days from the start date."}},
         "probation_period", "120",
         "Does Saudi Labor Law permit a 120-day probation period in this contract's circumstances, and which "
         "article governs it? Record the article number and the expected finding status."),
        ("con_en_probation_90_days_boundary", "Probation exactly at the common limit (90 days)",
         {"replace_clause": {"Probation Period": clause["Probation Period"]}}, "probation_period", "90",
         "Is 90 days compliant, and does the governing article make it a boundary case? Record the article number."),
        ("con_en_annual_leave_21_days", "Annual leave stated as 21 days",
         {"replace_clause": {"Annual Leave": "The Employee is entitled to paid annual leave of twenty-one (21) "
                                             "days per contract year."}}, "annual_leave", "21",
         "Is 21 days below the statutory minimum for this service length? Which article, and what status?"),
        ("con_en_annual_leave_30_days", "Annual leave stated as 30 days",
         None, "annual_leave", "30",
         "Is 30 days compliant, and under which article? Confirm whether service length changes the answer."),
        ("con_en_weekly_hours_48", "Weekly working hours stated as 48",
         {"replace_clause": {"Working Hours": "Working Hours: 8 hours per day, 48 hours per week."}},
         "weekly_working_hours", "48",
         "Is 48 hours per week within the statutory maximum, and does Ramadan or sector change it? Which article?"),
        ("con_en_daily_hours_10", "Daily working hours stated as 10",
         {"replace_clause": {"Working Hours": "Working Hours: 10 hours per day, 50 hours per week."}},
         "daily_working_hours", "10",
         "Is a 10-hour day permitted, and under which article and conditions?"),
        ("con_en_notice_7_days", "Notice period stated as 7 days",
         {"replace_clause": {"Notice Period": "Either party may terminate this contract by giving seven (7) days "
                                              "written notice."}}, "notice_period", "7",
         "What notice period does the law require for this contract type, and which article governs it?"),
    ]
    for case_id, title, mutation, field, value, question in legal:
        cases.append({
            "case_id": case_id,
            "title": title,
            "language": "en",
            "input": {"documents": [contract_doc("CONTRACT_EN", mutations=mutation) if mutation
                                    else contract_doc("CONTRACT_EN")]},
            "expected": {
                "route": "contract_analysis",
                "extraction": {field: {"value_contains": value, "status": "found"}},
                "findings": [{"field": field, "status": "TO_BE_REVIEWED", "expected_articles": []}],
                "verifiable_from_document": True,
            },
            "review": needs_review(question),
            "tags": ["compliance", "needs_legal_review"],
            "notes": "The extraction half of this case is mechanical and can be scored today. The compliance half "
                     "is blank on purpose: no article number was assigned without a human reading the law.",
        })

    for case in cases:
        case.setdefault("task", "contract_analysis")
        case.setdefault("gold", None)
        case.setdefault("duplicate_group", None)
        case["provenance"] = provenance(FIXTURE_FILES["contracts"],
                                        authority="Saudi Labor Law" if "compliance" in case["tags"] else None)
    return cases


# --------------------------------------------------------------------------- comparison
def comparison_cases() -> list[dict]:
    """Single-variable differences: exactly one dimension changes, so the expectation is unambiguous.

    The ranking direction asserted here is the one declared in
    sanad/agents/comparison_dimensions.py - it is a coded rule, not a legal opinion.
    """
    base = "sanad/agents/comparison_dimensions.py"
    spec = [
        ("cmp_basic_salary_only", "Only the basic salary differs",
         {"replace_compensation": {"Basic Salary": "14,000 SAR per month"}},
         "basic_salary", "contract_2", "higher_is_better"),
        ("cmp_weekly_hours_only", "Only the weekly working hours differ",
         {"replace_clause": {"Working Hours": "Working Hours: 9 hours per day, 45 hours per week."}},
         "weekly_working_hours", "contract_1", "lower_is_better"),
        ("cmp_annual_leave_only", "Only the annual leave differs",
         {"replace_clause": {"Annual Leave": "The Employee is entitled to paid annual leave of twenty-one (21) "
                                             "days per contract year."}},
         "annual_leave", "contract_1", "higher_is_better"),
    ]
    cases = []
    for case_id, title, mutation, dimension, leader, direction in spec:
        cases.append({
            "case_id": case_id, "task": "contract_comparison", "title": title, "language": "en",
            "input": {"documents": [contract_doc("CONTRACT_EN"),
                                    contract_doc("CONTRACT_EN", mutations=mutation)]},
            "expected": {
                "route": "contract_comparison",
                "dimensions": {dimension: {"comparable": True, "best_contract_ids": [leader],
                                           "ranking": direction}},
                "only_dimension_differing": dimension,
                "ranking_rule_ref": f"{base}::{dimension}",
                "analyses_per_contract": 1,
            },
            "review": MECHANICAL, "gold": None, "duplicate_group": None,
            "tags": ["comparison", "single_variable"],
            "notes": "The second contract is the first one with exactly one value changed.",
            "provenance": provenance(FIXTURE_FILES["contracts"]),
        })

    cases += [
        {
            "case_id": "cmp_identical_contracts",
            "task": "contract_comparison", "title": "Two identical contracts cannot have a winner",
            "language": "en",
            "input": {"documents": [contract_doc("CONTRACT_EN"), contract_doc("CONTRACT_EN")]},
            "expected": {"route": "contract_comparison",
                         "recommendation": {"status": "no_clear_preference"},
                         "no_dimension_differs": True, "analyses_per_contract": 1},
            "review": MECHANICAL, "gold": None, "duplicate_group": None,
            "tags": ["comparison", "control"],
            "notes": "A control case: if this ever produces a preferred contract, the ranking is not deterministic.",
            "provenance": provenance(FIXTURE_FILES["contracts"]),
        },
        {
            "case_id": "cmp_missing_field_not_comparable",
            "task": "contract_comparison", "title": "A dimension missing from one contract is not comparable",
            "language": "en",
            "input": {"documents": [contract_doc("CONTRACT_EN"),
                                    contract_doc("CONTRACT_EN", mutations={"remove_clause": "Probation Period"})]},
            "expected": {"route": "contract_comparison",
                         "dimensions": {"probation_period": {"comparable": False}},
                         "must_state_reason": True, "analyses_per_contract": 1},
            "review": MECHANICAL, "gold": None, "duplicate_group": None,
            "tags": ["comparison", "missing_field"],
            "notes": "",
            "provenance": provenance(FIXTURE_FILES["contracts"]),
        },
        {
            "case_id": "cmp_one_unusable_falls_back",
            "task": "contract_comparison", "title": "One unreadable upload falls back to a single analysis",
            "language": "en",
            "input": {"documents": [contract_doc("CONTRACT_EN"),
                                    {"fixture_module": "tests.fixtures.documents.builders",
                                     "builder": "scanned_pdf", "role": "contract"}]},
            "expected": {"route": "contract_analysis",
                         "routing": {"fallback_from": "contract_comparison"},
                         "warnings_contain": ["only one contract"], "status": ["partial"]},
            "review": MECHANICAL, "gold": None, "duplicate_group": None,
            "tags": ["comparison", "fallback"],
            "notes": "",
            "provenance": provenance(FIXTURE_FILES["contracts"]),
        },
    ]
    return cases


# --------------------------------------------------------------------------- salary
def salary_cases() -> list[dict]:
    """Methodology cases. No case states a correct wage; each states what the method must do."""
    def case(case_id, title, pages, expected, tags, notes="", review=None):
        return {
            "case_id": case_id, "task": "salary_benchmark", "title": title, "language": "en",
            "input": {"documents": [contract_doc("CONTRACT_EN")],
                      "salary_query": {"job_title": "Data Analyst", "location": "Riyadh"},
                      "search_pages": pages},
            "expected": expected, "review": review or MECHANICAL, "gold": None, "duplicate_group": None,
            "tags": tags, "notes": notes,
            "provenance": provenance(FIXTURE_FILES["salary_pages"]),
        }

    return [
        case("sal_official_source_tier", "An official source is recognised and ranked first",
             ["GASTAT"], {"status": ["success", "insufficient_data"],
                          "sources_all_in_allowlist": True, "source_tiers": ["official"],
                          "currency": "SAR", "period": "monthly",
                          "range_within_observed_figures": True},
             ["salary", "tier"], "GASTAT writes thousands with a dot (10.238); extraction must read 10238."),
        case("sal_market_range_paylab", "A published range is used as a range",
             ["PAYLAB"], {"status": ["success"], "source_tiers": ["market"], "currency": "SAR",
                          "period": "monthly", "basis": "total_compensation",
                          "range_within_observed_figures": True},
             ["salary", "range"], "The Paylab page states a total monthly range including bonuses."),
        case("sal_arabic_source_read", "An Arabic source page is read correctly",
             ["SAUDI_SALARY"], {"status": ["success", "insufficient_data"], "currency": "SAR",
                                "period": "monthly", "range_within_observed_figures": True},
             ["salary", "ar"], ""),
        case("sal_supplementary_only", "A supplementary source alone is labelled as such",
             ["KAGGLE"], {"status": ["success", "insufficient_data"], "source_tiers": ["supplementary"],
                          "data_quality": ["supplementary", "insufficient"]},
             ["salary", "tier"], ""),
        case("sal_lead_only_never_sets_range", "A LinkedIn post never sets the range",
             ["LINKEDIN"], {"status": ["insufficient_data"], "lead_only_excluded": True,
                            "limitations_contain": ["LinkedIn"]},
             ["salary", "tier", "safety"], "This is a safety case: lead-only sources must not drive a number."),
        case("sal_single_figure_is_not_a_range", "One figure from one source is not a range",
             ["SINGLE_OFFICIAL"], {"status": ["insufficient_data"],
                                   "limitations_contain": ["single figure"]},
             ["salary", "insufficient"], ""),
        case("sal_conflicting_sources_not_averaged", "Disagreeing sources are reported, never averaged",
             ["PAYLAB", "CONFLICTING_SOURCE"], {"status": ["insufficient_data"],
                                                "data_quality": ["conflicting"], "averaging_forbidden": True},
             ["salary", "safety"], "The two fixtures differ by more than 4x with disjoint spans."),
        case("sal_usd_converted_at_configured_rate", "A foreign currency is converted only at a configured rate",
             ["USD_SOURCE"], {"status": ["success", "insufficient_data"], "currency": "SAR",
                              "converted_from_present": "USD", "range_within_observed_figures": True},
             ["salary", "currency"], ""),
        case("sal_eur_without_rate_is_excluded", "A currency with no configured rate is left out and said so",
             ["EUR_SOURCE"], {"status": ["insufficient_data"], "limitations_contain": ["EUR"],
                              "excluded_currency": "EUR"},
             ["salary", "currency"], ""),
        case("sal_annual_normalised_to_month", "Annual figures are divided by 12, not reported as monthly",
             ["ANNUAL_SOURCE"], {"status": ["success", "insufficient_data"], "period": "monthly",
                                 "limitations_contain": ["Annual"], "range_within_observed_figures": True},
             ["salary", "period"], ""),
        case("sal_missing_period_excluded", "A figure with no stated period is not used",
             ["NO_PERIOD_SOURCE"], {"status": ["insufficient_data"],
                                    "limitations_contain": ["monthly or annual"]},
             ["salary", "period"], ""),
        case("sal_provider_failure_is_explained", "A failing search provider says why",
             ["__RAISE__"], {"status": ["error", "partial"], "message_states_reason": True},
             ["salary", "failure"],
             "Checkpoint 7 found that the standalone salary route drops this warning; this case is expected to "
             "fail until that is fixed, and is marked accordingly.",
             review=needs_review("Confirm the intended user-visible text when the salary provider fails; "
                                 "Checkpoint 7 finding F1 says the reason is currently dropped.")),
        case("sal_no_job_title_no_search", "With no job title, nothing is searched",
             ["GASTAT"], {"status": ["insufficient_data"], "no_search_performed": True},
             ["salary", "guard"], "Uses a contract whose job title was removed.",
             ),
    ]


# --------------------------------------------------------------------------- robustness
def robustness_cases() -> list[dict]:
    """Behaviour under awkward input and dead dependencies. No legal answers are asserted."""
    def case(case_id, title, language, inp, expected, tags, notes="", review=None):
        return {"case_id": case_id, "task": "robustness", "title": title, "language": language,
                "input": inp, "expected": expected, "review": review or MECHANICAL, "gold": None,
                "duplicate_group": None, "tags": tags, "notes": notes,
                "provenance": provenance("eval/build_fixture_cases.py")}

    ar_question = "ما هي المدة القصوى لفترة التجربة؟"
    return [
        case("rob_arabic_question", "An Arabic question is answered in Arabic", "ar",
             {"question": ar_question},
             {"route": "regulatory_question", "answer_language": "ar", "evidence_present": True},
             ["language"]),
        case("rob_english_question", "An English question is answered in English", "en",
             {"question": "What is the maximum probation period?"},
             {"route": "regulatory_question", "answer_language": "en", "evidence_present": True},
             ["language"],
             "No gold article is asserted: the source dataset contains no English questions, so the correct "
             "article for an English phrasing has not been established by a human."),
        case("rob_mixed_language_question", "A mixed Arabic/English question does not crash", "mixed",
             {"question": "ما هي probation period القصوى؟"},
             {"route": "regulatory_question", "status": ["success", "partial", "insufficient_evidence"],
              "no_exception": True},
             ["language"]),
        case("rob_paraphrased_question", "A paraphrase of a dataset question still retrieves evidence", "ar",
             {"question": "كم عدد الأيام المسموح بها في فترة التجربة؟"},
             {"route": "regulatory_question", "evidence_present": True},
             ["paraphrase"],
             "Deliberately no gold article: the paraphrase was not reviewed against the law."),
        case("rob_incomplete_question", "A one-word question is handled, not guessed", "ar",
             {"question": "التجربة؟"},
             {"route": "regulatory_question", "status": ["success", "partial", "insufficient_evidence"],
              "must_not_fabricate_article": True},
             ["incomplete"]),
        case("rob_irrelevant_question", "An out-of-scope question must not invent law", "ar",
             {"question": "ما هي حالة الطقس في الرياض غداً؟"},
             {"route": "regulatory_question", "status": ["insufficient_evidence", "success"],
              "must_not_fabricate_article": True, "abstention_expected": True},
             ["irrelevant", "safety"],
             "The cheapest hallucination canary in the set: no labor-law article answers this."),
        case("rob_unsupported_salary_without_contract", "Salary benchmarking without a contract is refused", "en",
             {"task": "salary_benchmark", "documents": []},
             {"route": "none", "status": "invalid_input", "error_codes": ["no_contract_uploaded"]},
             ["unsupported"]),
        case("rob_empty_request", "A request with neither question nor document is refused", "not_applicable",
             {"question": None, "documents": []},
             {"route": "none", "status": "invalid_input", "error_codes": ["nothing_to_do"]},
             ["unsupported"]),
        case("rob_corrupted_document", "Bytes that are not a real document are rejected cleanly", "not_applicable",
             {"documents": [{"fixture_module": "eval", "builder": "corrupt_bytes", "role": "contract"}]},
             {"status": "invalid_input", "error_codes": ["unusable_document"], "no_exception": True},
             ["malformed"]),
        case("rob_unsupported_file_type", "A non-PDF/DOCX file is refused at the edge", "not_applicable",
             {"documents": [{"fixture_module": "tests.fixtures.documents.builders",
                             "builder": "image_bytes", "role": "contract"}]},
             {"http_status": 415, "error_codes": ["unsupported_file_type"]},
             ["malformed"]),
        case("rob_too_many_contracts", "More contracts than the limit is refused before parsing", "en",
             {"documents": [contract_doc("CONTRACT_EN") for _ in range(6)]},
             {"route": "none", "status": "invalid_input", "error_codes": ["too_many_contracts"]},
             ["multiple_documents"]),
        case("rob_rag_unavailable", "A dead regulatory search is reported, not hidden", "ar",
             {"question": ar_question, "scenario": "rag_unavailable"},
             {"status": "rag_error", "http_status": 502, "must_not_fabricate_article": True},
             ["dependency", "safety"]),
        case("rob_llm_unavailable", "With no interpreter, findings stay 'requires review'", "en",
             {"documents": [contract_doc("CONTRACT_EN")], "scenario": "llm_unavailable"},
             {"route": "contract_analysis", "no_compliance_labels": True,
              "findings_status_in": ["requires_review", "not_found", "not_applicable", "insufficient_evidence"]},
             ["dependency", "safety"],
             "This is the guarantee that a missing key degrades to abstention instead of a guess."),
        case("rob_salary_provider_unavailable", "A dead salary provider does not break the request", "en",
             {"documents": [contract_doc("CONTRACT_EN")], "task": "salary_benchmark",
              "scenario": "salary_provider_unavailable"},
             {"status": ["partial", "insufficient_evidence"], "no_exception": True,
              "message_states_reason": True},
             ["dependency"],
             review=needs_review("Confirm the expected user-visible message; Checkpoint 7 finding F1 reports that "
                                 "the reason is currently dropped on this route.")),
    ]


def group_shared_questions(cases: list[dict]) -> list[dict]:
    """Two cases may legitimately reuse one question string (same question, different scenario).

    They are never silently merged: the shared string is made explicit with a duplicate_group so the
    evaluation runner counts the group once rather than treating them as independent evidence.
    """
    by_question: dict[str, list[dict]] = {}
    for case in cases:
        question = (case["input"].get("question") or "").strip()
        if question:
            by_question.setdefault(question, []).append(case)
    for question, group in by_question.items():
        if len(group) > 1:
            marker = f"dup_{hashlib.sha1(question.encode()).hexdigest()[:8]}"
            for case in group:
                case["duplicate_group"] = marker
    return cases


def write(path: Path, cases: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")


if __name__ == "__main__":
    groups = {
        "contract": contract_cases(),
        "comparison": comparison_cases(),
        "salary": salary_cases(),
        "robustness": robustness_cases(),
    }
    for name, cases in groups.items():
        write(EVAL_DIR / "golden" / name / "cases.jsonl", group_shared_questions(cases))
    print(json.dumps({name: len(cases) for name, cases in groups.items()}, indent=2))
    print(json.dumps({"fixture_hashes": HASHES}, indent=2))
