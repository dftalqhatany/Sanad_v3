"""Clause segmentation and classification: the document's own words, split by contractual rule."""

from __future__ import annotations

import pytest

from extraction.clauses import ClauseSegmenter, classify_heading, classify_sentence, segment_clauses
from extraction.text import flat
from models.extraction import ClauseType
from tests.fixtures.documents.builders import docx_from_paragraphs

PROBATION = "The initial period of probation is 180 days from your date of joining."
NOTICE_DURING = ("During the period of probation, either side can terminate this employment contract "
                 "by giving the other 90 days' notice or their gross salary in lieu thereof.")
NOTICE_AFTER = ("After confirmation, either side may terminate your services by giving 90 days' notice "
                "in writing or by paying your gross salary in lieu thereof.")


@pytest.fixture
def prose(processor):
    def build(paragraphs):
        return processor.parse_bytes(docx_from_paragraphs(paragraphs), "letter.docx")
    return build


def _of_type(clauses, clause_type):
    return [c for c in clauses if c.clause_type is clause_type]


# --------------------------------------------------------------------------- A. segmentation
def test_one_paragraph_stating_several_rules_becomes_several_clauses(prose):
    """Requirement 6: the unit is the contractual rule, not the paragraph."""
    document = prose([f"{PROBATION} {NOTICE_DURING} {NOTICE_AFTER}"])
    clauses = segment_clauses(document)
    assert len(_of_type(clauses, ClauseType.PROBATION)) == 1
    assert len(_of_type(clauses, ClauseType.NOTICE)) == 2


def test_sentences_belonging_to_one_rule_are_not_split(prose):
    extra = "You will be deemed to be confirmed at the end of the probationary period."
    clauses = segment_clauses(prose([f"{PROBATION} {extra}"]))
    probation = _of_type(clauses, ClauseType.PROBATION)
    assert len(probation) == 1
    assert "180 days" in probation[0].text and "deemed to be confirmed" in probation[0].text


def test_clause_ids_are_assigned_in_document_order(prose):
    clauses = segment_clauses(prose([PROBATION, NOTICE_AFTER]))
    assert [c.clause_id for c in clauses] == [f"C{i:02d}" for i in range(1, len(clauses) + 1)]


# --------------------------------------------------------------------------- B. classification
@pytest.mark.parametrize("sentence, expected", [
    (PROBATION, ClauseType.PROBATION),
    (NOTICE_DURING, ClauseType.NOTICE),
    (NOTICE_AFTER, ClauseType.NOTICE),
    ("You will be required to work up to 40 hours a week.", ClauseType.WORKING_HOURS),
    ("Your weekly off would be for 2 days.", ClauseType.WEEKLY_REST),
    ("You will get 24 paid leave per annum.", ClauseType.ANNUAL_LEAVE),
    ("Your initial place of work shall be in Riyadh.", ClauseType.WORK_LOCATION),
    ("You shall not disclose any confidential information to third parties.", ClauseType.CONFIDENTIALITY),
    ("You will be entitled to sick leave as per company policy.", ClauseType.SICK_LEAVE),
])
def test_a_sentence_is_classified_by_the_rule_it_states(sentence, expected):
    assert classify_sentence(sentence)[0] is expected


def test_a_sentence_stating_no_rule_is_not_forced_into_a_type():
    assert classify_sentence("This letter is issued in duplicate.")[0] is None
    assert classify_sentence("The company values punctuality.")[0] is None


def test_headings_name_the_block_beneath_them():
    assert classify_heading("3. Job Duties & Responsibilities") is ClauseType.DUTIES
    assert classify_heading("4. Confidentiality") is ClauseType.CONFIDENTIALITY
    assert classify_heading("Compensation and Salary") is ClauseType.SALARY
    assert classify_heading("Annexure A") is None


def test_an_unclassifiable_section_produces_no_clause_rather_than_a_guess(prose):
    clauses = segment_clauses(prose(["This letter is issued in duplicate. Please retain a copy."]))
    assert clauses == []


# --------------------------------------------------------------------------- C. provenance
def test_every_clause_quotes_the_document_and_cites_where(prose):
    document = prose([PROBATION, NOTICE_DURING, "You will get 24 paid leave per annum."])
    full = flat(document.full_text)
    for clause in segment_clauses(document):
        assert clause.clause_id
        assert clause.clause_name
        assert clause.source_span is not None
        assert clause.section_id
        assert clause.extraction_method is not None
        assert clause.confidence in ("high", "medium", "low")
        assert flat(clause.text) in full, clause.text


def test_clause_text_is_never_paraphrased(prose):
    clauses = segment_clauses(prose([PROBATION]))
    assert clauses[0].text.startswith("The initial period of probation is 180 days")


# --------------------------------------------------------------------------- D/E. subject binding
def test_the_notice_during_probation_sentence_is_a_notice_clause_not_a_probation_one(prose):
    """A keyword is not a subject. This sentence mentions probation and states a notice period."""
    clauses = segment_clauses(prose([NOTICE_DURING]))
    assert _of_type(clauses, ClauseType.PROBATION) == []
    notice = _of_type(clauses, ClauseType.NOTICE)
    assert len(notice) == 1 and "90 days" in notice[0].text


def test_the_probation_clause_excludes_the_notice_sentence(prose):
    clauses = segment_clauses(prose([f"{PROBATION} {NOTICE_DURING}"]))
    probation = _of_type(clauses, ClauseType.PROBATION)[0]
    assert "180 days" in probation.text
    assert "90 days" not in probation.text
    assert "notice" not in probation.text.casefold()


def test_the_two_notice_clauses_are_named_by_the_phase_they_govern(prose):
    clauses = _of_type(segment_clauses(prose([NOTICE_DURING, NOTICE_AFTER])), ClauseType.NOTICE)
    assert {c.clause_name for c in clauses} == {"Notice during probation", "Notice after confirmation"}


# --------------------------------------------------------------------------- F. repeated keywords
def test_two_clauses_stating_different_probation_periods_both_survive(prose):
    """Requirement 12: segmentation never silently picks a winner."""
    other = "The probation period shall be 90 days for all administrative staff."
    clauses = _of_type(segment_clauses(prose([PROBATION, other])), ClauseType.PROBATION)
    assert len(clauses) == 2
    texts = " | ".join(c.text for c in clauses)
    assert "180 days" in texts and "90 days" in texts
    assert clauses[0].source_span.section_id != clauses[1].source_span.section_id


def test_repeated_identical_rules_are_kept_as_separate_clauses(prose):
    clauses = _of_type(segment_clauses(prose([NOTICE_AFTER, NOTICE_AFTER])), ClauseType.NOTICE)
    assert len(clauses) == 2
    assert clauses[0].clause_id != clauses[1].clause_id


# --------------------------------------------------------------------------- foreign references
def test_foreign_legal_references_are_preserved_verbatim(prose):
    """Requirement 11: Stage 3 needs to see these, so nothing is normalised away."""
    sentence = ("A female employee will be eligible for all the benefits, as applicable under the "
                "provisions of the Maternity Benefit Act, 1961, and the Rules made thereunder.")
    clauses = segment_clauses(prose([sentence]))
    assert any("Maternity Benefit Act, 1961" in c.text for c in clauses)


# --------------------------------------------------------------------------- configuration
def test_very_short_fragments_are_not_promoted_to_clauses(prose):
    segmenter = ClauseSegmenter(min_clause_chars=1000)
    assert segmenter.segment(prose([PROBATION, NOTICE_AFTER])) == []


# --------------------------------------------------------------------------- clause boundaries
RETIREMENT = "15. You will automatically retire on reaching the age of 62."


def test_a_new_numbered_item_is_never_absorbed_into_the_clause_above(prose):
    """An unclassified sentence continues the rule above it only within the same item.

    "Your weekly off would be for 2 days. 15. You will automatically retire..." are two different
    rules; folding the retirement age into the weekly-rest clause would put it in front of the
    weekly-rest article as though the contract said it there.
    """
    clauses = segment_clauses(prose([f"Your weekly off would be for 2 days. {RETIREMENT}"]))
    weekly = _of_type(clauses, ClauseType.WEEKLY_REST)
    assert len(weekly) == 1
    assert "automatically retire" not in weekly[0].text
    assert "62" not in weekly[0].text


def test_an_unclassified_new_item_is_preserved_as_its_own_clause(prose):
    clauses = segment_clauses(prose([f"Your weekly off would be for 2 days. {RETIREMENT}"]))
    other = _of_type(clauses, ClauseType.OTHER)
    assert len(other) == 1
    assert "automatically retire" in other[0].text


def test_an_unclassified_sentence_inside_the_same_item_still_continues_it(prose):
    """The boundary rule must not fragment a rule written over two sentences."""
    clauses = segment_clauses(prose([
        "The initial period of probation is 180 days from your date of joining. "
        "You will be deemed to be confirmed at the end of that period."]))
    probation = _of_type(clauses, ClauseType.PROBATION)
    assert len(probation) == 1
    assert "deemed to be confirmed" in probation[0].text


@pytest.mark.parametrize("sentence", [
    "15. You will automatically retire on reaching the age of 62.",
    "(3) All company assets must be returned on the last day of employment.",
    "• You must notify us of your acceptance within two business days.",
])
def test_item_openers_are_recognised(sentence):
    from extraction.clauses import starts_new_item
    assert starts_new_item(sentence) is True


def test_an_ordinary_sentence_is_not_an_item_opener():
    from extraction.clauses import starts_new_item
    assert starts_new_item("Your weekly off would be for 2 days.") is False


# --------------------------------------------------------------------------- weak classification
@pytest.mark.parametrize("sentence", [
    # Names benefits, grants none.
    "Your compensation and benefits are attached as Annexure A.",
    # A modal governing "the terms", not the allowances listed inside the parenthesis.
    ("In the event that you are transferred by the company to another position, the terms and "
     "conditions applicable to the new location (including, but not limited to, the compensation "
     "and benefits, allowances, entitlements) shall apply to you."),
    # An enumeration of component names, with no figure and no entitlement.
    "CTC component - Basic Pay, House Rent Allowance (HRA), Conveyance Allowance, Food Allowance.",
    # Says pay is private; says nothing about what the pay is.
    "Your individual remuneration is purely a matter between yourself and the company.",
    # A document checklist, not an appointment rule.
    "Records of previous employment (offer letters, appointment letters, salary slips).",
])
def test_a_topic_word_alone_does_not_create_a_regulatory_classification(sentence):
    """Requirement 2: prefer no type over a wrong one."""
    clause_type, _ = classify_sentence(sentence)
    assert clause_type is None, f"{sentence[:60]!r} was classified as {clause_type}"


@pytest.mark.parametrize("sentence, expected", [
    ("A female employee will be eligible for all the benefits under the Maternity Benefit Act, 1961.",
     ClauseType.BENEFITS),
    ("As an employee of Aramco Group you will be eligible for the below mentioned perks.",
     ClauseType.BENEFITS),
    ("Your housing allowance shall be 2,500 SAR per month.", ClauseType.HOUSING),
    ("You will be provided a transportation allowance as per company policy.", ClauseType.TRANSPORTATION),
    ("Your earning potential will be -10.14 LPA.", ClauseType.SALARY),
    ("You will be entitled to sick leave as per company policy.", ClauseType.SICK_LEAVE),
    ("You shall not disclose any confidential information to third parties.", ClauseType.CONFIDENTIALITY),
])
def test_genuine_subject_bound_matches_still_classify(sentence, expected):
    """The tightening must not cost a legitimate match."""
    assert classify_sentence(sentence)[0] is expected
