"""Narrative ("semantic") extraction: prose that states a fact without a label.

These tests use short synthetic paragraphs. The mandatory end-to-end regression against the real
Aramco appointment letter lives in test_aramco_regression.py.
"""

from __future__ import annotations

import pytest

from extraction.contract import extract_contract
from extraction.semantic import extract_semantic_fields, sentence_allows
from models.extraction import FieldStatus
from tests.fixtures.documents.builders import docx_from_paragraphs

FOUND = FieldStatus.FOUND
NOT_FOUND = FieldStatus.NOT_FOUND
AMBIGUOUS = FieldStatus.AMBIGUOUS


@pytest.fixture
def prose(processor):
    """Parse narrative paragraphs the way an uploaded document would be parsed."""
    def build(paragraphs):
        return processor.parse_bytes(docx_from_paragraphs(paragraphs), "letter.docx")
    return build


# --------------------------------------------------------------------------- one rule at a time
@pytest.mark.parametrize("sentence, field, normalized", [
    ("You shall be appointed to the position of Oil truck driver's helper at the Aramco Business Unit.",
     "job_title", "Oil truck driver's helper"),
    ("You will be employed as a Senior Site Engineer with the company.",
     "job_title", "a Senior Site Engineer"),
    ("Your initial place of work shall be in Riyadh.", "work_location", "Riyadh"),
    ("Your appointment will be effective from July 24, 2025, unless agreed otherwise.",
     "start_date", "2025-07-24"),
    ("The initial period of probation is 180 days from your date of joining.",
     "probation_period", "180 day"),
    ("The probationary period shall be 90 days.", "probation_period", "90 day"),
    ("You will be required to work up to 40 hours a week.", "working_hours", "40 h/week"),
    ("Your weekly off would be for 2 days.", "weekly_rest", "2 day"),
    ("You will get 24 paid leave per annum.", "annual_leave", "24 day"),
    ("You will be entitled to 21 days of leave per year.", "annual_leave", "21 day"),
])
def test_a_fact_written_as_prose_is_read_with_its_normalized_form(prose, sentence, field, normalized):
    fields = extract_semantic_fields(prose([sentence]))
    assert field in fields, f"{field} was not read from: {sentence}"
    assert fields[field].status is FOUND
    assert fields[field].normalized_value == normalized


def test_the_greeting_line_names_the_employee(prose):
    fields = extract_semantic_fields(prose(["Dear Mr./Ms.Akram Hosen Ovi ,Pas: A05365009"]))
    assert fields["employee_name"].status is FOUND
    assert fields["employee_name"].value == "Akram Hosen Ovi"


def test_a_take_home_figure_keeps_its_currency_and_period(prose):
    fields = extract_semantic_fields(prose(["In hand monthly- 1800 Riyal"]))
    salary = fields["in_hand_salary"]
    assert salary.status is FOUND
    assert (salary.value.amount, salary.value.currency, salary.value.period) == (1800.0, "SAR", "monthly")
    assert salary.normalized_value == "SAR 1800 monthly"


# --------------------------------------------------------------------------- the field-mixing rule
PROBATION = "The initial period of probation is 180 days from your date of joining."
NOTICE_DURING = ("During the period of probation, either side can terminate this employment contract "
                 "by giving the other 90 days' notice or their gross salary in lieu thereof.")
NOTICE_AFTER = ("After confirmation, either side may terminate your services by giving 90 days' notice "
                "in writing or by paying your gross salary in lieu thereof.")


def test_probation_and_the_two_notice_periods_stay_three_separate_fields(prose):
    fields = extract_semantic_fields(prose([PROBATION, NOTICE_DURING, NOTICE_AFTER]))
    assert fields["probation_period"].normalized_value == "180 day"
    assert fields["notice_period_during_probation"].normalized_value == "90 day"
    assert fields["notice_period_after_confirmation"].normalized_value == "90 day"


def test_a_notice_sentence_never_contributes_to_the_probation_period(prose):
    """The regression: "During the period of probation ... 90 days' notice" mentions probation but
    states a notice period. Reading its 90 days as the probation length made the field ambiguous
    and hid the real 180 days."""
    fields = extract_semantic_fields(prose([NOTICE_DURING]))
    assert "probation_period" not in fields
    assert fields["notice_period_during_probation"].normalized_value == "90 day"


def test_a_leave_sentence_that_mentions_probation_is_not_a_probation_period(prose):
    sentence = "After the completion of your probation, you will be eligible for 12 paid days."
    assert "probation_period" not in extract_semantic_fields(prose([sentence]))


@pytest.mark.parametrize("text, allowed", [
    (PROBATION, True),
    (NOTICE_DURING, False),
    (NOTICE_AFTER, False),
    ("After the completion of your probation, you will be eligible for 12 paid days.", False),
    ("فترة التجربة تسعون يوماً من تاريخ المباشرة.", True),
    ("The probation period is three months.", True),
])
def test_sentence_allows_refuses_only_sentences_carrying_another_fields_cue(text, allowed):
    assert sentence_allows("probation_period", text) is allowed


def test_sentence_allows_does_not_restrict_fields_without_a_competing_cue():
    for field in ("salary", "annual_leave", "working_hours", "job_title"):
        assert sentence_allows(field, "anything at all, 30 days' notice included") is True


# --------------------------------------------------------------------------- honesty guarantees
def test_nothing_is_read_from_a_document_that_does_not_say_it(prose):
    fields = extract_semantic_fields(prose([
        "The company values punctuality and professional conduct at all times.",
        "This letter is issued in duplicate.",
    ]))
    assert fields == {}


def test_every_value_is_quoted_from_the_text_it_was_read_from(prose):
    fields = extract_semantic_fields(prose([PROBATION, NOTICE_AFTER, "You will get 24 paid leave per annum."]))
    for name, field in fields.items():
        assert field.source_text, name
        assert field.sources and all(s.section_id or s.table_id for s in field.sources), name
        flat_source = " ".join(field.source_text.split()).casefold()
        assert " ".join(field.raw_value.split()).casefold() in flat_source, name


def test_an_assumed_unit_is_recorded_in_the_notes(prose):
    """"24 paid leave per annum" states no unit. Reading it as days is defensible, but the reader
    has to be able to see that it was an assumption."""
    field = extract_semantic_fields(prose(["You will get 24 paid leave per annum."]))["annual_leave"]
    assert field.normalized_value == "24 day"
    assert any("without a unit" in note for note in field.notes)


def test_two_numbers_in_one_leave_statement_are_not_silently_settled(prose):
    field = extract_semantic_fields(prose(["Annual leave is thirty (21) days."]))
    assert field.get("annual_leave") is None or field["annual_leave"].status is not FOUND


def test_a_sentence_about_the_companys_hours_is_not_the_employees_hours(prose):
    """"The company will work five days a week, 24 hours a day" describes opening hours."""
    fields = extract_semantic_fields(prose(["The company will work five days a week, 24 hours a day."]))
    assert "working_hours" not in fields
    assert fields["working_days"].confidence == "medium"
    assert fields["working_days"].notes


# --------------------------------------------------------------------------- the two paths together
def test_a_label_is_never_overruled_by_prose(prose):
    """A labeled value is the most explicit thing a document can say, so prose does not replace it."""
    document = prose([
        "Job Title: Accountant",
        "You shall be appointed to the position of Senior Engineer at the Riyadh office.",
    ])
    result = extract_contract(document)
    assert result.job_title.status is FOUND
    assert result.job_title.value == "Accountant"


def test_a_document_that_states_two_probation_periods_stays_ambiguous(prose):
    """Prose is not a tie-breaker for a real contradiction. A letter that says both 90 and 180 days
    is contradictory, and saying so is more useful than silently preferring one of them."""
    document = prose([
        "Probation Period: 90 days",
        "The initial period of probation is 180 days from your date of joining.",
    ])
    result = extract_contract(document)
    assert result.probation_period.status is AMBIGUOUS
    assert result.probation_period.value is None
    written = {c.raw_value for c in result.probation_period.candidates}
    assert {"90 days", "180 days"} <= written


def test_prose_fills_a_field_the_label_path_left_empty(prose):
    document = prose([
        "Employee Name: Reem Test",
        "Your initial place of work shall be in Jeddah.",
    ])
    result = extract_contract(document)
    assert result.employee_name.status is FOUND and result.employee_name.value == "Reem Test"
    assert result.work_location.status is FOUND and result.work_location.value == "Jeddah"
    assert "read from a sentence rather than a label" in result.work_location.notes
