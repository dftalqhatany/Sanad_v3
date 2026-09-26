"""Contract fields: explicit values with provenance; NOT_FOUND / AMBIGUOUS / NOT_EXTRACTED instead of guesses."""

from __future__ import annotations

import pytest

from extraction import extract_contract
from extraction.text import flat
from models.common import ResultStatus
from models.extraction import ContractExtraction, ExtractionMethodName, FieldStatus
from tests.fixtures.documents.builders import docx_from_paragraphs, pdf_with_text_pages_and_scanned_page

FOUND, NOT_FOUND, AMBIGUOUS = FieldStatus.FOUND, FieldStatus.NOT_FOUND, FieldStatus.AMBIGUOUS


def _dump(field):
    value = field.value
    if isinstance(value, list):
        return [item.model_dump(exclude_none=True) for item in value]
    return value.model_dump(exclude_none=True) if hasattr(value, "model_dump") else value


def _contract(processor, paragraphs, table_rows=None):
    return extract_contract(processor.parse_bytes(docx_from_paragraphs(paragraphs, table_rows=table_rows), "c.docx"))


def assert_provenance(result: ContractExtraction, document) -> None:
    """Every found value is literally present in the document text it cites."""
    for name, field in result.fields().items():
        if field.status is FieldStatus.FOUND:
            assert field.source_text and field.source_text in document.full_text, name
            assert field.sources and all(s.text in document.full_text for s in field.sources), name
            if not isinstance(field.value, list):
                assert flat(field.raw_value) in flat(field.source_text), name
            assert all(s.section_id or s.table_id for s in field.sources), name
        else:
            assert field.value is None and field.raw_value is None, name


EXPECTED_EN = {
    "employee_name": "Jordan Sample",
    "employer_name": "Example Tech Solutions LLC",
    "job_title": "Data Analyst",
    "contract_type": {"raw": "Fixed-term", "normalized": "fixed_term"},
    "start_date": {"raw": "15/10/2026", "iso_date": "2026-10-15", "calendar": "gregorian"},
    "end_date": {"raw": "14/10/2027", "iso_date": "2027-10-14", "calendar": "gregorian"},
    "probation_period": {"count": 90.0, "unit": "day"},
    "salary": {"amount": 12000.0, "currency": "SAR", "period": "monthly"},
    "housing_allowance": {"amount": 3000.0, "currency": "SAR", "period": "monthly"},
    "transportation_allowance": {"amount": 800.0, "currency": "SAR", "period": "monthly"},
    "other_allowances": [{"name": "Mobile Allowance", "amount": 200.0, "currency": "SAR", "period": "monthly"}],
    "working_hours": {"hours_per_day": 8.0, "hours_per_week": 40.0},
    "working_days": {"days": ["sunday", "monday", "tuesday", "wednesday", "thursday"]},
    "annual_leave": {"count": 30.0, "unit": "day"},
    "notice_period": {"count": 60.0, "unit": "day"},
    "work_location": "Riyadh",
    "nationality": "Saudi",
    "employee_id": "EMP-00123",
}


@pytest.mark.parametrize("fixture", ["sample_contract_en.docx", "sample_contract_text.pdf"])
def test_english_contract_fields(parsed, fixture):
    document = parsed[fixture]
    result = extract_contract(document)
    assert result.status is ResultStatus.SUCCESS and result.document_type == "employment_contract"
    for name, expected in EXPECTED_EN.items():
        field = getattr(result, name)
        assert field.status is FOUND, (name, field.notes, field.candidates)
        assert _dump(field) == expected, name
    assert result.probation_period.raw_value == "ninety (90) days"
    assert result.termination_terms.status is FOUND
    assert result.termination_terms.value[0].heading == "Termination"
    assert result.termination_terms.value[0].text.startswith("This contract may be terminated")
    # Exhaustive on purpose: any field that stops being read shows up here. The narrative-only
    # fields are listed because this fixture is a labeled contract that does not state them - it
    # states one notice period for the whole contract, not one per phase.
    assert result.fields_with_status(NOT_FOUND) == [
        "contract_duration", "total_salary", "net_salary", "notice_period_during_probation",
        "notice_period_after_confirmation", "weekly_rest", "in_hand_salary", "benefits",
    ]
    assert_provenance(result, document)


def test_arabic_contract_fields(parsed):
    document = parsed["sample_contract_ar.docx"]
    result = extract_contract(document)
    expected = {
        "employee_name": "ريم الاختبار",
        "employer_name": "شركة سند التجريبية للتقنية",  # from "الطرف الأول (صاحب العمل)"
        "job_title": "محاسبة",
        "contract_type": {"raw": "محدد المدة", "normalized": "fixed_term"},
        "contract_duration": {"count": 1.0, "unit": "year"},
        "probation_period": {"count": 90.0, "unit": "day"},
        "salary": {"amount": 9500.0, "currency": "SAR", "period": "monthly"},
        "housing_allowance": {"amount": 2500.0, "currency": "SAR"},
        "transportation_allowance": {"percentage": 25.0, "percentage_basis": "من الراتب الأساسي"},
        "other_allowances": [{"name": "بدل طعام", "amount": 300.0, "currency": "SAR"}],
        "working_hours": {"hours_per_day": 8.0},
        "working_days": {"days": ["sunday", "monday", "tuesday", "wednesday", "thursday"]},
        "annual_leave": {"count": 21.0, "unit": "day"},
        "notice_period": {"count": 60.0, "unit": "day"},
        "work_location": "جدة",
        "nationality": "سعودية",
        "employee_id": "EMP-00456",
    }
    for name, value in expected.items():
        field = getattr(result, name)
        assert field.status is FOUND, (name, field.notes)
        assert _dump(field) == value, name
    assert result.probation_period.raw_value == "تسعون يوماً"
    assert result.annual_leave.raw_value == "واحد وعشرون يوماً"
    assert result.housing_allowance.raw_value == "٢٥٠٠ ريال"
    assert result.start_date.status is FOUND and result.start_date.value.iso_date is None
    assert "day/month order cannot be determined" in result.start_date.notes
    assert result.end_date.status is NOT_FOUND and result.end_date.value is None
    assert result.termination_terms.value[0].heading == "البند السابع: إنهاء العقد"
    assert_provenance(result, document)


def test_provenance_points_to_page_table_and_method(parsed):
    result = extract_contract(parsed["sample_contract_text.pdf"])
    assert result.salary.page_number == 1 and result.salary.sources[0].table_id == "t001"
    assert result.salary.sources[0].method is ExtractionMethodName.TABLE_ROW and result.salary.confidence == "high"
    assert result.salary.source_text == "Basic Salary | 12,000 SAR per month"
    assert result.notice_period.page_number == 2
    assert result.notice_period.sources[0].method is ExtractionMethodName.CLAUSE_SENTENCE
    assert result.notice_period.confidence == "medium"
    assert result.employee_name.source_text == "Employee Name: Jordan Sample"


def test_missing_fields_are_not_found_and_nothing_is_invented(processor):
    result = _contract(processor, ["Employee Name: Sam Placeholder", "This agreement is governed by the applicable law."])
    assert result.employee_name.status is FOUND and result.employee_name.value == "Sam Placeholder"
    others = [name for name in result.fields() if name != "employee_name"]
    for name in others:
        field = getattr(result, name)
        assert field.status is NOT_FOUND and field.value is None and field.sources == [], name


def test_conflicting_values_are_ambiguous_with_all_candidates(processor):
    result = _contract(processor, [
        "Basic Salary: 10,000 SAR",
        "The basic salary is 12,000 SAR per month.",
        "Job Title: Data Analyst",
        "Position: Senior Data Analyst",
    ])
    assert result.salary.status is AMBIGUOUS and result.salary.value is None
    assert {c.raw_value for c in result.salary.candidates} == {"10,000 SAR", "12,000 SAR"}
    assert any("different amounts" in note for note in result.salary.notes)
    assert result.job_title.status is AMBIGUOUS
    assert {c.value for c in result.job_title.candidates} == {"Data Analyst", "Senior Data Analyst"}


@pytest.mark.parametrize("line, note", [
    ("Probation Period: 90 or 180 days", "range or alternative"),
    ("Annual Leave: thirty (21) days", "conflicting numbers"),
    ("Probation Period: as per company policy", "could not be read"),
    ("Basic Salary: twelve thousand riyals", "could not be read"),
    ("Start Date: upon visa issuance", "no date found"),
])
def test_unreadable_or_contradictory_single_values_are_ambiguous(processor, line, note):
    result = _contract(processor, [line])
    field = next(f for f in result.fields().values() if f.status is not NOT_FOUND)
    assert field.status is AMBIGUOUS and field.value is None and field.candidates
    assert any(note in n for n in field.notes), field.notes


def test_same_value_written_twice_is_found_once_with_both_sources(processor):
    result = _contract(processor, ["Probation Period: 90 days",
                                   "The employee is subject to a probation period of ninety (90) days."])
    assert result.probation_period.status is FOUND and result.probation_period.value.count == 90
    assert len(result.probation_period.sources) == 2 and result.probation_period.confidence == "high"


def test_label_variants_and_distinct_salary_fields(processor):
    result = _contract(processor, ["إسم العامل: سامي المثال", "الراتب: ٥٠٠٠ ريال", "إجمالي الراتب: 7,000 ريال",
                                   "رقم الهوية: 1000000001"])
    assert result.employee_name.value == "سامي المثال"  # hamza variant of the label still matches
    assert result.salary.value.amount == 5000 and result.total_salary.value.amount == 7000
    assert result.employee_id.status is NOT_FOUND  # a national ID is not an employee ID


def test_labels_in_two_column_tables(processor):
    result = _contract(processor, ["Employment Contract"], table_rows=[("Employee Name", "Sam Placeholder"),
                                                                        ("Notice Period", "30 days")])
    assert result.employee_name.value == "Sam Placeholder"
    assert result.notice_period.value.count == 30 and result.notice_period.sources[0].table_id is not None


def test_unreadable_document_is_not_extracted_rather_than_not_found(parsed):
    result = extract_contract(parsed["sample_scanned.pdf"])
    assert result.status is ResultStatus.ERROR and result.errors[0].code == "document_ocr_required"
    assert all(field.status is FieldStatus.NOT_EXTRACTED for field in result.fields().values())


def test_partially_read_document_is_partial(processor):
    result = extract_contract(processor.parse_bytes(pdf_with_text_pages_and_scanned_page(), "mixed.pdf"))
    assert result.status is ResultStatus.PARTIAL and result.warnings
    assert result.employee_name.value == "Jordan Sample"


def test_extraction_result_serialises(parsed):
    result = extract_contract(parsed["sample_contract_ar.docx"])
    restored = ContractExtraction.model_validate_json(result.model_dump_json())
    assert restored.model_dump(mode="json") == result.model_dump(mode="json")
    assert restored.salary.value.amount == 9500
