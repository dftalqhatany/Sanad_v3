"""Mandatory extraction regression: the real Aramco appointment letter.

This is a narrative appointment letter, not a labeled contract. Before the narrative extraction
path existed the label-driven extractor read exactly one of its twenty-one fields, and that one
reading was wrong in kind: it merged two legally distinct notice periods into a single field, while
the 180-day probation was lost because the notice-during-probation sentence was credited to
`probation_period` and made it ambiguous.

Every assertion below is checked against the document's own words: the sentence a value was read
from is asserted verbatim, so a rule that starts matching the wrong sentence fails here even if it
happens to produce the same number.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from extraction.contract import extract_contract
from models.common import ResultStatus
from models.extraction import FieldStatus

FOUND = FieldStatus.FOUND
FIXTURE = "aramco_appointment_letter.pdf"
# A real document, not one of the synthetic builders: it is checked in as-is so that the numbers
# below are the ones a user's own upload would produce.
LETTER_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "documents" / FIXTURE


@pytest.fixture(scope="module")
def letter():
    if not LETTER_PATH.exists():
        pytest.skip(f"{FIXTURE} is not checked in at {LETTER_PATH}")
    return LETTER_PATH


@pytest.fixture(scope="module")
def extraction(processor, letter):
    document = processor.parse_bytes(letter.read_bytes(), FIXTURE)
    assert document.status.has_text, "the letter must parse before anything can be extracted"
    return extract_contract(document)


def test_the_letter_is_extracted_successfully(extraction):
    assert extraction.status is ResultStatus.SUCCESS
    assert extraction.document_type == "employment_contract"


# --------------------------------------------------------------------------- the required fields
#   field -> (normalized value, a phrase that must appear in the sentence it was read from)
REQUIRED = {
    "employee_name": ("Akram Hosen Ovi", "Dear Mr./Ms."),
    "job_title": ("Oil truck driver's helper", "appointed to the position of"),
    "work_location": ("Riyadh(Saudi Arab)", "initial place of work shall be"),
    "start_date": ("2025-07-24", "effective from July 24, 2025"),
    "probation_period": ("180 day", "The initial period of probation is 180 days"),
    "notice_period_during_probation": ("90 day", "During the period of probation"),
    "notice_period_after_confirmation": ("90 day", "After confirmation"),
    "working_hours": ("40 h/week", "required to work up to 40 hours a week"),
    "weekly_rest": ("2 day", "weekly off would be for 2 days"),
    "annual_leave": ("24 day", "24 paid leave per annum"),
    "in_hand_salary": ("SAR 1800 monthly", "In hand monthly- 1800 Riyal"),
}


@pytest.mark.parametrize("name, expected, quote", [(n, *v) for n, v in REQUIRED.items()])
def test_required_field_is_read_from_the_sentence_that_states_it(extraction, name, expected, quote):
    field = getattr(extraction, name)
    assert field.status is FOUND, (name, field.notes)
    assert field.normalized_value == expected, name
    flat_source = " ".join((field.source_text or "").split())
    assert " ".join(quote.split()) in flat_source, (name, flat_source)


def test_benefits_are_listed_as_written(extraction):
    benefits = extraction.benefits
    assert benefits.status is FOUND
    listed = [item.text for item in benefits.value]
    assert "Free Transportation" in listed
    assert "Life Insurance" in listed
    assert len(listed) >= 5


# --------------------------------------------------------------------------- the field-mixing rule
def test_probation_is_180_days_and_is_not_the_90_day_notice(extraction):
    """The specific bug this stage exists to fix."""
    probation = extraction.probation_period
    assert probation.status is FOUND, probation.notes
    assert probation.value.count == 180.0 and probation.value.unit == "day"
    assert probation.value.count != 90.0
    assert "90 days" not in " ".join(probation.source_text.split())


def test_the_two_notice_periods_are_separate_fields_with_their_own_sources(extraction):
    during = extraction.notice_period_during_probation
    after = extraction.notice_period_after_confirmation
    assert during.status is FOUND and after.status is FOUND
    assert during.value.count == 90.0 and after.value.count == 90.0
    # Same number, different clauses: the sources must not be the same sentence.
    assert during.source_text != after.source_text
    assert "During the period of probation" in during.source_text
    assert "After confirmation" in after.source_text


def test_probation_and_notice_never_share_a_source_sentence(extraction):
    probation_sources = {s.text for s in extraction.probation_period.sources}
    for name in ("notice_period_during_probation", "notice_period_after_confirmation"):
        notice_sources = {s.text for s in getattr(extraction, name).sources}
        assert not (probation_sources & notice_sources), name


# --------------------------------------------------------------------------- provenance contract
def test_every_found_field_carries_value_source_page_status_and_confidence(extraction):
    for name, field in extraction.fields().items():
        if field.status is not FOUND:
            assert field.value is None and field.normalized_value is None, name
            continue
        assert field.value is not None, name
        assert field.normalized_value, name
        assert field.source_text, name
        assert field.page_number is not None and field.page_number >= 1, name
        assert field.confidence in ("high", "medium", "low"), name
        assert field.sources, name


def test_no_value_is_invented(extraction):
    """Every reading is literally present in the text it cites."""
    for name, field in extraction.fields().items():
        if field.status is not FOUND or isinstance(field.value, list):
            continue
        flat_raw = " ".join(field.raw_value.split()).casefold()
        flat_source = " ".join(field.source_text.split()).casefold()
        assert flat_raw in flat_source, (name, field.raw_value, field.source_text)


def test_fields_the_letter_does_not_state_stay_not_found(extraction):
    """The letter states no contract type, end date or housing allowance. Absence is not a value."""
    for name in ("contract_type", "end_date", "contract_duration", "housing_allowance",
                 "transportation_allowance", "nationality"):
        field = getattr(extraction, name)
        assert field.status is not FOUND, (name, field.value)
        assert field.value is None, name


def test_the_narrative_path_reads_far_more_than_the_label_path_alone(extraction):
    """Guards the headline result: 1 of 21 fields before this stage, 14 of 26 after."""
    found = extraction.fields_with_status(FOUND)
    assert len(found) >= 13, found
