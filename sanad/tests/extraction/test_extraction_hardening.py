"""Extraction Hardening regression suite (Stage: Multi-Format Contract Intelligence).

Four new real/synthetic-narrative documents, on top of the Aramco letter already covered by
test_aramco_regression.py, exercise the layouts that stage this fixed:

  * contract_shifa_albahr_pharmacist.pdf         - real bilingual PDF, colon-less bold "heading-style"
                                                    label rows, an explicit English probation waiver.
  * contract_taibah_grand_front_office_manager.pdf - real bilingual PDF, "Label: value" rows where the
                                                    PDF text layer glues an Arabic mirror onto the same
                                                    line with no separator, an explicit Arabic-context
                                                    probation waiver (word order corrupted by the PDF's
                                                    own right-to-left text-layer extraction).
  * data_engineer_contract_1.docx / _3.docx      - clean, colon-labeled Arabic narrative contracts
                                                    ("المادة ..." articles), no bilingual glue at all.

None of these fixtures existed before this stage; before it, the bilingual PDFs above produced either
false NOT_FOUND (no colon at all) or a silently wrong value (a colon whose "value" swallowed a glued
Arabic mirror), and the two Arabic docx contracts produced almost nothing outside the compensation
table, because extraction.text had no bilingual-script-boundary handling and extraction.semantic had
no Arabic narrative rules at all. Every assertion below is checked against the fixture's own words,
not against a hand-picked expectation, and no fixture is special-cased in production code (see
test_no_document_is_special_cased_in_the_rule_engine in test_aramco_legal_findings.py for the existing
guard that already covers this file's fixtures too, since it scans by module, not by filename).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from extraction.contract import extract_contract
from extraction.semantic import detect_negations
from extraction.text import flat, label_keys
from models.common import ResultStatus
from models.extraction import FieldStatus
from tests.fixtures.documents.builders import docx_from_paragraphs

FOUND, NOT_FOUND, AMBIGUOUS, NOT_APPLICABLE = (
    FieldStatus.FOUND, FieldStatus.NOT_FOUND, FieldStatus.AMBIGUOUS, FieldStatus.NOT_APPLICABLE,
)
FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "documents"


def _load(processor, filename):
    path = FIXTURES_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} is not checked in at {path}")
    document = processor.parse_bytes(path.read_bytes(), filename)
    assert document.status.has_text, f"{filename} must parse before anything can be extracted"
    return document


def _contract(processor, paragraphs):
    """A minimal synthetic docx, for the handful of schema/alias checks below that no single real
    fixture happens to state in a clean, unambiguous form (see each test's own docstring)."""
    return extract_contract(processor.parse_bytes(docx_from_paragraphs(paragraphs), "c.docx"))


@pytest.fixture(scope="module")
def shifa(processor):
    return _load(processor, "contract_shifa_albahr_pharmacist.pdf")


@pytest.fixture(scope="module")
def taibah(processor):
    return _load(processor, "contract_taibah_grand_front_office_manager.pdf")


@pytest.fixture(scope="module")
def data_engineer_1(processor):
    return _load(processor, "data_engineer_contract_1.docx")


@pytest.fixture(scope="module")
def data_engineer_3(processor):
    return _load(processor, "data_engineer_contract_3.docx")


@pytest.fixture(scope="module")
def shifa_extraction(shifa):
    return extract_contract(shifa)


@pytest.fixture(scope="module")
def taibah_extraction(taibah):
    return extract_contract(taibah)


@pytest.fixture(scope="module")
def de1_extraction(data_engineer_1):
    return extract_contract(data_engineer_1)


@pytest.fixture(scope="module")
def de3_extraction(data_engineer_3):
    return extract_contract(data_engineer_3)


def test_all_four_documents_parse_and_extract_successfully(
    shifa_extraction, taibah_extraction, de1_extraction, de3_extraction,
):
    for extraction in (shifa_extraction, taibah_extraction, de1_extraction, de3_extraction):
        assert extraction.status is ResultStatus.SUCCESS


# --------------------------------------------------------------------------- A. bare-label PDF rows
# Shifa Al-Bahr states identity fields as a bold "label" line (no colon) with the bilingual value on
# the line(s) that follow - before this stage, extraction.text only read "Label: value" and table
# rows, so every one of these came back NOT_FOUND even though the document plainly states them.
def test_shifa_bare_label_rows_are_read(shifa_extraction):
    result = shifa_extraction
    # employee_name resolves to AMBIGUOUS rather than a clean FOUND: the document's own signature
    # block ("First Party (Employer) | <mirrored Arabic caption>") reads as a second candidate
    # alongside the real name, because the PDF's own text layer stores that caption's Arabic side
    # word-reordered - a known limitation documented in the completion report, section "known
    # limitations". What matters here is that the real value is present and traceable, not silently
    # dropped in favour of - or lost behind - the spurious one.
    assert result.employee_name.status is AMBIGUOUS
    assert "Turki Salem Al-Harbi" in [c.value for c in result.employee_name.candidates]
    assert result.job_title.status is FOUND and result.job_title.value == "Pharmacist"
    assert result.nationality.status is FOUND and result.nationality.value == "Saudi"
    # work_location has the same signature-block interference as employee_name above.
    assert result.work_location.status is AMBIGUOUS
    assert "Jeddah" in [c.value for c in result.work_location.candidates]
    assert result.contract_type.status is FOUND and result.contract_type.value.normalized == "fixed_term"


def test_shifa_working_pattern_and_leave_are_read(shifa_extraction):
    result = shifa_extraction
    assert result.working_hours.status is FOUND
    assert result.working_hours.value.hours_per_week == 48.0
    assert result.working_days.status is FOUND and result.working_days.value.days_per_week == 6
    assert result.annual_leave.status is FOUND and result.annual_leave.value.count == 30.0


def test_shifa_housing_allowance_is_read_from_an_arabic_worded_amount(shifa_extraction):
    """The label is "9.1.1.2 Housing Allowance )" - a clause-numbered, stray-parenthesis-decorated
    English label (the lone ")" is the source PDF's own RTL/bidi corruption: the Arabic "monthly"
    gloss that follows has its mirrored parentheses swapped) - whose value is the Arabic-worded
    amount immediately adjacent to it ('ريال سعودي2,500.00'), not written in the same script as the
    label. A few lines later, a *different* clause ("9.1.1.6 Deduction for Public Authority for
    Social Insurance Subscription") computes "9.75% of the basic wage plus housing allowance = SAR
    1,218.75" - a real amount, but the deduction, not the allowance; a keyword-matched sentence must
    never be allowed to out-vote the field's own explicit label."""
    housing = shifa_extraction.housing_allowance
    assert housing.status is FOUND, (housing.notes, housing.candidates)
    assert housing.value.amount == 2500.0


# --------------------------------------------------------------------------- B. explicit negation
def test_shifa_and_taibah_probation_waiver_is_not_applicable_not_not_found(shifa_extraction, taibah_extraction):
    """Both documents explicitly state the employee is NOT subject to a probationary period. Reporting
    NOT_FOUND there would misrepresent a document that directly addresses the topic. Taibah's own
    Arabic clause has its word order corrupted by the source PDF's right-to-left text-layer extraction
    (independently confirmed by dumping the raw parsed text), which is exactly why the Arabic side of
    the negation check is order-independent rather than one fixed phrase."""
    for result in (shifa_extraction, taibah_extraction):
        probation = result.probation_period
        assert probation.status is NOT_APPLICABLE, probation.notes
        assert probation.value is None
        assert probation.sources, "a not_applicable determination must still be traceable to a source"
        assert "تجرب" in probation.sources[0].text or "probation" in probation.sources[0].text.lower()


def test_negation_never_fires_on_an_unrelated_field(shifa_extraction):
    """A document-wide negation table must not spill over onto fields it was never written for."""
    assert shifa_extraction.annual_leave.status is FOUND  # nothing here was negated


def test_negation_detector_is_additive_and_keyed_by_field_name(shifa):
    negations = detect_negations(shifa)
    assert set(negations) <= {"probation_period"}
    assert negations["probation_period"].status is NOT_APPLICABLE


# --------------------------------------------------------------------------- C. Arabic narrative contracts
# The two data-engineer contracts are pure Arabic narrative ("المادة ..." articles): no labels, no
# tables, for most fields. Before this stage's Arabic narrative rules, only the compensation table and
# the (already colon-labeled) nationality/duration fields were read.
@pytest.mark.parametrize("extraction_name", ["de1_extraction", "de3_extraction"])
def test_arabic_narrative_identity_and_dates_are_read(request, extraction_name):
    result = request.getfixturevalue(extraction_name)
    assert result.employee_name.status is FOUND
    assert result.employee_name.value == "خالد محمد العتيبي"
    assert result.job_title.status is FOUND and result.job_title.value == "Data Engineer"
    assert result.start_date.status is FOUND and result.start_date.value.iso_date == "2026-11-15"
    assert result.end_date.status is FOUND
    assert result.contract_duration.status is FOUND
    assert result.nationality.status is FOUND


def test_arabic_narrative_probation_and_salary_components_stay_distinct(de1_extraction, de3_extraction):
    assert de1_extraction.probation_period.value.count == 90.0
    assert de3_extraction.probation_period.value.count == 180.0
    for result, basic, housing, transport in (
        (de1_extraction, 15000.0, 3750.0, 1000.0),
        (de3_extraction, 17000.0, 4250.0, 1000.0),
    ):
        assert result.salary.status is FOUND and result.salary.value.amount == basic
        assert result.housing_allowance.status is FOUND and result.housing_allowance.value.amount == housing
        assert result.transportation_allowance.status is FOUND
        assert result.transportation_allowance.value.amount == transport
        # basic, housing and transport must never collapse into one another or into the total line
        amounts = {result.salary.value.amount, result.housing_allowance.value.amount,
                   result.transportation_allowance.value.amount}
        assert len(amounts) == 3


def test_job_title_prefers_the_english_gloss_over_the_arabic_phrase(de1_extraction):
    """'بمسمى مهندس بيانات (Data Engineer)' states the same fact two ways in one sentence; this must
    resolve to one clean value, not AMBIGUOUS between the Arabic phrase and its own English gloss."""
    job_title = de1_extraction.job_title
    assert job_title.status is FOUND
    assert job_title.value == "Data Engineer"
    assert not job_title.candidates


# --------------------------------------------------------------------------- D. cross-language regression
# The same concept, stated in Arabic-only (data-engineer contracts), Arabic+English bilingual (Shifa,
# Taibah) and English-only (Aramco, covered elsewhere) documents, must all normalize into the same
# canonical field rather than separate per-language schemas.
def test_probation_period_normalizes_the_same_way_regardless_of_source_language(de1_extraction, de3_extraction):
    assert de1_extraction.probation_period.normalized_value == "90 day"
    assert de3_extraction.probation_period.normalized_value == "180 day"


def test_nationality_is_read_in_both_languages(shifa_extraction, taibah_extraction, de1_extraction):
    assert shifa_extraction.nationality.value == "Saudi"
    assert taibah_extraction.nationality.value == "Jordanian"
    assert de1_extraction.nationality.value == "سعودي"


# --------------------------------------------------------------------------- E. provenance contract
@pytest.mark.parametrize("extraction_name", [
    "shifa_extraction", "taibah_extraction", "de1_extraction", "de3_extraction",
])
def test_every_found_value_is_literally_present_in_its_cited_source(request, extraction_name):
    result = request.getfixturevalue(extraction_name)
    for name, field in result.fields().items():
        if field.status is not FOUND:
            assert field.value is None, name
            continue
        assert field.source_text, name
        if not isinstance(field.value, list):
            assert flat(field.raw_value) in flat(field.source_text), name
        assert field.sources, name


@pytest.mark.parametrize("extraction_name", [
    "shifa_extraction", "taibah_extraction", "de1_extraction", "de3_extraction",
])
def test_ambiguous_fields_carry_their_candidates(request, extraction_name):
    """Section 22: never silently choose one. An AMBIGUOUS field always keeps every candidate it saw
    - whether that is two conflicting values (the real bilingual PDFs' signature-block role captions
    read as a second, spurious candidate alongside the real value - a known limitation of the source
    PDFs' own right-to-left text-layer corruption, documented in the completion report) or a single
    candidate whose own raw text could not be read reliably (e.g. two dates written in one span) -
    and the field itself never carries a guessed value."""
    result = request.getfixturevalue(extraction_name)
    for name, field in result.fields().items():
        if field.status is AMBIGUOUS:
            assert len(field.candidates) >= 1, name
            assert field.value is None, name


# --------------------------------------------------------------------------- F. downstream compatibility
def test_extraction_output_still_feeds_clause_segmentation(shifa_extraction, taibah_extraction):
    """Stage 3's ClauseSegmenter/RegulatoryEvidenceCollector consume `extraction.clauses`; this stage
    must not have broken that handoff for the new document shapes."""
    for result in (shifa_extraction, taibah_extraction):
        assert result.clauses, "a 12-article bilingual contract must yield clause segments"
        assert all(c.text for c in result.clauses)


# --------------------------------------------------------------------------- G. hardening follow-up
# A second pass over the same five fixtures, after a targeted quality-gate review found eight
# remaining issues in the extraction above. Each fix stays generalised - a dictionary/structural
# signal usable on any document, never one fixture's exact wording - and every test here is checked
# against a fixture's own words or the shared alias table, never a hand-picked expectation.

def test_shifa_housing_allowance_excludes_the_gosi_deduction_amount(shifa_extraction):
    """A few lines below the real housing-allowance label, a *different*, later clause ("9.1.1.6
    Deduction for Public Authority for Social Insurance Subscription") computes "Deducted at a rate
    of (9.75%) of the basic wage plus housing allowance = SAR 1,218.75" - a real amount, but the
    deduction, not the allowance. The PDF's own layout splits that clause's "Deducted ... 9.75%"
    wording and its "= SAR 1,218.75" figure into two separate text units, so the disqualifying cue is
    not always in the very same sentence as the number it explains; the keyword-sentence scan must
    still recognise it via the immediately preceding unit and never offer 1,218.75 as a competing
    housing_allowance candidate."""
    housing = shifa_extraction.housing_allowance
    assert housing.status is FOUND, (housing.notes, housing.candidates)
    assert housing.value.amount == 2500.0
    assert not any(getattr(c.value, "amount", None) == 1218.75 for c in housing.candidates)


def test_label_keys_strips_multi_level_clause_numbers_before_matching():
    """Schema/structural regression for the housing-allowance label bug: "9.1.1.2 Housing Allowance
    )" (Shifa's own label, stray trailing parenthesis - the source PDF's own RTL/bidi corruption -
    and all) must reduce to the same key as the plain alias "housing allowance". A dotted, multi-
    level clause number ("9.1.1.2") is the hard case - by the time a naive ordinal-strip runs on the
    already-normalised key, its dots have already become spaces, so only the leading "9" was ever
    stripped and "1 1 2 housing allowance" never matched anything. This checks the mechanism
    directly, not only through one fixture's exact wording."""
    assert label_keys("9.1.1.2 Housing Allowance )") == {"housing allowance"}
    assert label_keys("6.1 Probationary Period") == {"probationary period"}
    assert label_keys("9.1.1.1 Basic Wage") == {"basic wage"}


def test_taibah_annual_leave_excludes_the_postponement_limit(taibah_extraction):
    """The document grants 21 days of annual leave in one clause and, separately, lets the employer
    postpone that leave "for a period not exceeding (90) days" in another. The 90-day postponement
    limit must never be merged into, or offered as a competing candidate for, the leave entitlement
    itself - "annual leave" is not enough of a cue on its own when the sentence is really about
    deferring it."""
    leave = taibah_extraction.annual_leave
    assert leave.status is FOUND, (leave.notes, leave.candidates)
    assert leave.value.count == 21.0
    assert not any(getattr(c.value, "count", None) == 90.0 for c in leave.candidates)


def test_taibah_transportation_allowance_reads_a_label_wrapped_across_two_lines(taibah_extraction):
    """The label itself is written across three physical lines ("9.1.1.3" / "Transportation" /
    "Allowance"). "Transportation" is already, on its own, one of the field's own aliases, so before
    this fix it was promoted as a complete label immediately, and the very next line ("Allowance")
    was taken as its value and failed to parse as an amount - leaving the real "SAR 500.00" one line
    further down unreachable. The fix must recognise that appending the next line turns the label
    into a *different*, longer recognised alias ("transportation allowance") before accepting any
    value for it."""
    transport = taibah_extraction.transportation_allowance
    assert transport.status is FOUND, (transport.notes, transport.candidates)
    assert transport.value.amount == 500.0
    assert transport.value.currency == "SAR"


def test_net_salary_field_exists_and_stays_well_formed_on_every_fixture(
    shifa_extraction, taibah_extraction, de1_extraction, de3_extraction,
):
    """`net_salary` is a new canonical field: previously there was no way to represent a document's
    take-home wage separately from `salary` (basic) and `total_salary` (gross) at all. Taibah and the
    two data-engineer contracts don't write it in a form any extraction path can reach, so those three
    are expected to report NOT_FOUND, not crash or silently attach the wrong amount.

    Shifa is the one exception, and it reaches `net_salary` through a *different* path than the one the
    other three fail on: its bare-label "Net Wage" line glues a colon into the value text itself, which
    the label mechanism correctly refuses to treat as a single value (a colon inside a candidate means
    "this looks like its own label: value pair", a pre-existing and unrelated safeguard) - but the same
    figure is *also* stated as its own PDF table row ("9.1.1.7 Net Wage | ... | 12,781.25"), which
    PyMuPDF's table detector picks up as a distinct table-cell candidate, unaffected by that safeguard.
    (Stage 4.1 note: this was true before Stage 4.1 and is unrelated to clause-to-field linkage - it was
    simply never observed until a real, unpatched parse of the Shifa PDF was run against this suite.)"""
    for result in (taibah_extraction, de1_extraction, de3_extraction):
        assert result.net_salary.status is NOT_FOUND
        assert result.net_salary.value is None

    assert shifa_extraction.net_salary.status is FOUND
    assert shifa_extraction.net_salary.value.amount == 12781.25
    assert shifa_extraction.net_salary.value.currency == "SAR"


def test_net_salary_is_read_from_its_english_and_arabic_aliases(processor):
    """Alias regression for the new field: a plain "Label: amount" line must reach `net_salary` the
    same way every other money field is already reached, in both languages."""
    english = _contract(processor, ["Net Salary: 9,000 SAR"])
    assert english.net_salary.status is FOUND
    assert english.net_salary.value.amount == 9000.0

    arabic = _contract(processor, ["صافي الراتب: 9000 ريال"])
    assert arabic.net_salary.status is FOUND
    assert arabic.net_salary.value.amount == 9000.0


def test_total_salary_is_read_via_the_total_monthly_wage_alias(de1_extraction, de3_extraction):
    """Both Arabic narrative contracts state "إجمالي الأجر الشهري" ("total monthly wage") - standard
    Saudi labour-law wording for the same concept as "total salary" / "gross salary" - which no
    existing alias matched before this fix."""
    assert de1_extraction.total_salary.status is FOUND
    assert de1_extraction.total_salary.value.amount == 19750.0
    assert de3_extraction.total_salary.status is FOUND
    assert de3_extraction.total_salary.value.amount == 22250.0


def test_total_salary_is_read_from_the_english_total_monthly_wage_alias(processor):
    result = _contract(processor, ["Total Monthly Wage: 12,500 SAR"])
    assert result.total_salary.status is FOUND
    assert result.total_salary.value.amount == 12500.0


def test_probation_status_is_applicable_when_a_period_is_stated(de1_extraction, de3_extraction):
    """The new `probation_status` field sits alongside `probation_period` without replacing it:
    `probation_period` keeps the duration, `probation_status` states plainly that probation applies."""
    for result, days in ((de1_extraction, 90.0), (de3_extraction, 180.0)):
        assert result.probation_period.status is FOUND and result.probation_period.value.count == days
        assert result.probation_status.status is FOUND
        assert result.probation_status.value == "applicable"


def test_probation_status_mirrors_probation_periods_not_applicable_waiver(shifa_extraction, taibah_extraction):
    """A document that explicitly waives probation must be NOT_APPLICABLE on both fields, never FOUND
    on one and silent (or, worse, disagreeing) on the other."""
    for result in (shifa_extraction, taibah_extraction):
        assert result.probation_period.status is NOT_APPLICABLE
        assert result.probation_status.status is NOT_APPLICABLE
        assert result.probation_status.value is None


def test_shifa_and_taibah_end_date_merges_the_paired_hijri_date(shifa_extraction, taibah_extraction):
    """Both contracts write the end date as "Gregorian (Hijri H)" - two representations of one event,
    not two competing dates. Before this fix, the Hijri echo was read as a second, unreadable date
    and the whole labeled value collapsed to a single "several dates are written" candidate, forcing
    end_date to AMBIGUOUS even though only one calendar's date needs to be normalised."""
    for result, iso in ((shifa_extraction, "2027-10-31"), (taibah_extraction, "2028-05-31")):
        end = result.end_date
        assert end.status is FOUND, (end.notes, end.candidates)
        assert end.value.iso_date == iso
        assert end.value.calendar == "gregorian"
        assert any("Hijri" in note for note in end.notes)


def test_genuinely_conflicting_dates_still_resolve_to_ambiguous(processor):
    """The Hijri/Gregorian pairing fix must not swallow a real conflict: two different Gregorian
    dates written for the same labeled field stay AMBIGUOUS, exactly as before - only a Hijri-
    calendar reading immediately paired with a Gregorian one is ever folded into one candidate."""
    result = _contract(processor, ["Start Date: 01/01/2025", "Start Date: 01/06/2025"])
    assert result.start_date.status is AMBIGUOUS
    assert len(result.start_date.candidates) == 2
