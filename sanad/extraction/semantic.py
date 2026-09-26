"""Narrative ("semantic") contract field extraction.

The label-driven extractor in `extraction.contract` reads "Job Title: X" lines and two-column table
rows. Appointment letters and ordinary contract prose carry the same facts in sentences instead
("You shall be appointed to the position of X"), so that extractor finds nothing in them. This module
adds a second, additive path: a table of narrative rules, each of which

  * matches one sentence / heading / list item,
  * captures the value **as written** (the captured span is always a verbatim substring of the unit,
    so `extraction.fields.candidate` can reject anything that is not literally in the document),
  * hands that span to the existing readers in `extraction.values` for normalisation.

No legal interpretation happens here, nothing is inferred from outside the document, and no LLM is
used: every rule is an explicit, reviewable pattern.

Role binding (why a rule is not just a keyword)
-----------------------------------------------
A keyword scan for "probation" also matches "During the period of probation, either side can
terminate ... by giving 90 days' notice", which states a *notice* period, not the probation period.
Every duration rule here therefore binds the number to its own subject syntactically
("period of probation **is** 180 days"), and `sentence_allows` exports that binding so the existing
keyword scan can apply it too. That is the root-cause fix for probation/notice field mixing.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Callable

from extraction import values as readers
from extraction.fields import candidate, not_applicable, resolve, resolve_list
from extraction.text import BULLET_CHARS, TextUnit, contains_keyword, iter_lines, iter_sentences, matching_key
from models.documents import ParsedDocument
from models.extraction import (
    DateValue,
    DurationValue,
    ExtractedField,
    ExtractionMethodName,
    FieldCandidate,
    ListItem,
    MoneyValue,
    SourceSpan,
    WorkingDaysValue,
    WorkingHoursValue,
)

# --------------------------------------------------------------------------- shared pattern pieces
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
    "forty": 40, "forty-five": 45, "sixty": 60, "ninety": 90,
}
_NUM = r"(?:\d{1,4}(?:\.\d+)?|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")"
# Plural first: regex alternation is ordered, so "day|days" would match "day" and leave the "s" behind.
_UNIT = r"(?:calendar\s+|working\s+|business\s+)?(?:days|day|weeks|week|months|month|years|year)"
_DURATION = rf"{_NUM}\s*(?:\(\s*\d+\s*\)\s*)?{_UNIT}"
# "90 days'" / "90 days’" - the apostrophe is not part of the duration but must not break the match.
_APOS = r"[’'`´]?"

_MONTH = (r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
          r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)")
_DATE = (rf"(?:{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"
         rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}\.?,?\s+\d{{4}}"
         r"|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})")

_MONEY = r"(?:[A-Z]{3}\s*)?[\d,]+(?:\.\d+)?\s*(?:riyals?|SAR|SR|ريال|USD|dollars?|LPA|lakhs?)?"


def _flags() -> int:
    return re.IGNORECASE | re.UNICODE


# --------------------------------------------------------------------------- rules
@dataclass(frozen=True)
class NarrativeRule:
    """One narrative pattern for one field.

    `pattern` must expose a group named ``v`` holding the value **exactly as written**.
    `excludes` disqualify the whole text unit (defence in depth: the pattern is already
    subject-bound, so an exclude is a second, independent reason to skip).
    """

    field: str
    kind: str  # duration | date | text | money | hours | days
    pattern: re.Pattern[str]
    excludes: tuple[re.Pattern[str], ...] = ()
    confidence: str = "high"
    note: str | None = None
    max_chars: int = 160

    def search(self, text: str) -> re.Match[str] | None:
        if any(bad.search(text) for bad in self.excludes):
            return None
        match = self.pattern.search(text)
        if match is None or not (match.group("v") or "").strip():
            return None
        return match if len(match.group("v")) <= self.max_chars else None


def _rule(field: str, kind: str, pattern: str, *, excludes: tuple[str, ...] = (), confidence: str = "high",
          note: str | None = None, max_chars: int = 160) -> NarrativeRule:
    return NarrativeRule(field, kind, re.compile(pattern, _flags()),
                         tuple(re.compile(e, _flags()) for e in excludes), confidence, note, max_chars)


# Sentences that state a *notice* period. A duration inside one of these never belongs to probation.
_NOTICE_SENTENCE = (
    r"notice",
    r"in\s+lieu\s+(?:there)?of",
    r"terminate\b",
)
# Sentences that state a *leave* entitlement, not a probation length.
_LEAVE_SENTENCE = (r"\bleave\b", r"\bpaid\s+days?\b", r"\bcasual\b", r"\bentitled\b", r"\beligible\s+for\b")

NARRATIVE_RULES: tuple[NarrativeRule, ...] = (
    # ---------------------------------------------------------------- parties and role
    _rule("employee_name", "text",
          r"\bDear\s+(?:Mr\.?\s*/\s*Ms\.?|Mrs?\.?|Ms\.?|Miss)\s*\.?\s*"
          r"(?P<v>[A-Z][\w.'’-]*(?:\s+[A-Z][\w.'’-]*){0,4})",
          excludes=(r"@",), max_chars=60),
    _rule("job_title", "text",
          r"(?:appointed\s+to\s+the\s+position\s+of|appointed\s+as|"
          r"position\s+(?:of|is)|designated\s+as|the\s+role\s+of|employed\s+as|"
          r"join\s+(?:us\s+)?as)\s+"
          r"(?P<v>[^.;:]+?)"
          r"(?=\s+(?:at|with|in|for|under|reporting)\b|\s*[.,;:]|$)",
          max_chars=90),
    _rule("work_location", "text",
          r"(?:place\s+of\s+work|work\s+location|posted|based|stationed|location\s+of\s+work)"
          r"[^.;]{0,30}?(?:shall\s+be|will\s+be|is|be)\s+(?:in|at)\s+"
          r"(?P<v>[^.;]+?)(?=\s*[.;]|$)",
          max_chars=90),

    # ---------------------------------------------------------------- dates
    _rule("start_date", "date",
          rf"(?:appointment|employment|engagement|contract|joining|services?)\b[^.]{{0,80}}?"
          rf"(?:effective|commenc\w+|start\w*|begin\w*)\s+(?:from|on)\s+(?P<v>{_DATE})"),
    _rule("start_date", "date",
          rf"(?:date\s+of\s+joining|joining\s+date|date\s+of\s+commencement)\s*(?:is|shall\s+be|:|-)?\s*"
          rf"(?P<v>{_DATE})"),

    # ---------------------------------------------------------------- probation (subject-bound)
    _rule("probation_period", "duration",
          rf"(?:period\s+of\s+probation|probation(?:ary)?\s+period|probation)\s*"
          rf"(?:is|shall\s+be|will\s+be|of|:|-)\s*(?P<v>{_DURATION})",
          excludes=_NOTICE_SENTENCE + _LEAVE_SENTENCE),
    _rule("probation_period", "duration",
          rf"(?:you\s+(?:shall|will)\s+be\s+on\s+probation\s+for|on\s+probation\s+for)\s+(?P<v>{_DURATION})",
          excludes=_NOTICE_SENTENCE + _LEAVE_SENTENCE),

    # ---------------------------------------------------------------- Arabic narrative (المادة sentences)
    # A Saudi contract's Arabic job-title sentence very often glosses the title in English in
    # parentheses right after it ("بمسمى مهندس بيانات (Data Engineer)") specifically so the title is
    # unambiguous; that English span is untouched by Arabic reshaping/reordering, so it is preferred
    # when present. The Arabic-only fallback below still fires when no such gloss exists.
    _rule("job_title", "text",
          r"(?:بمسمى|بوظيفة|بمنصب|بصفته|كـ)\s+[^()،,.؛]*?\(\s*(?P<v>[A-Za-z][A-Za-z\s/&.-]*[A-Za-z])\s*\)",
          max_chars=60),
    _rule("job_title", "text",
          r"(?:بمسمى|بوظيفة|بمنصب)\s+(?P<v>[^()،,.؛]+?)(?=\s*\(|[،,.؛]|$)",
          excludes=(r"(?:بمسمى|بوظيفة|بمنصب|بصفته|كـ)\s+[^()،,.؛]*?\([^()]*[A-Za-z][^()]*\)",),
          confidence="medium", max_chars=60),
    _rule("start_date", "date",
          r"تبدأ\s+من\s+(?:تاريخ\s+)?(?P<v>\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})"),
    _rule("end_date", "date",
          r"(?:و\s*)?تنتهي\s+في\s+(?P<v>\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})"),

    # ---------------------------------------------------------------- notice, split by phase
    _rule("notice_period_during_probation", "duration",
          rf"(?:during|within|in)\s+(?:the\s+)?(?:period\s+of\s+probation|probation(?:ary)?\s+period|probation)"
          rf"[^.]{{0,120}}?(?P<v>{_DURATION}){_APOS}\s*(?:notice|prior\s+notice|written\s+notice)"),
    _rule("notice_period_during_probation", "duration",
          rf"(?P<v>{_DURATION}){_APOS}\s*(?:notice|prior\s+notice)"
          rf"[^.]{{0,80}}?during\s+(?:the\s+)?probation"),
    _rule("notice_period_after_confirmation", "duration",
          rf"(?:after|upon|post|on)\s+(?:your\s+)?confirmation"
          rf"[^.]{{0,120}}?(?P<v>{_DURATION}){_APOS}\s*(?:notice|prior\s+notice|written\s+notice)"),
    _rule("notice_period_after_confirmation", "duration",
          rf"(?:once\s+confirmed|after\s+being\s+confirmed|following\s+confirmation)"
          rf"[^.]{{0,120}}?(?P<v>{_DURATION}){_APOS}\s*notice"),

    # ---------------------------------------------------------------- time at work (employee subject)
    _rule("working_hours", "hours",
          rf"(?:you\s+(?:will|shall|are)\s+(?:be\s+)?(?:required\s+to\s+)?work|"
          rf"your\s+working\s+hours?|hours\s+of\s+work\s+(?:shall|will)\s+be)"
          rf"[^.;]{{0,40}}?(?P<v>{_NUM}\s*hours?\s*(?:a|per|each)\s*(?:day|week))",
          excludes=(r"^\s*the\s+company\s+(?:will\s+)?work",)),
    _rule("working_days", "days",
          rf"(?P<v>{_NUM}\s*days?\s*(?:a|per|each)\s*week)",
          confidence="medium",
          note="read from a sentence about the working week; confirm whether it states the company's "
               "operating days or the employee's."),
    _rule("weekly_rest", "duration",
          rf"(?:your\s+)?weekly\s+(?:off|rest|holiday)\s*(?:day)?s?\s*"
          rf"(?:would\s+be|will\s+be|shall\s+be|is|are|of)?\s*(?:for\s+)?(?P<v>{_DURATION})"),
    _rule("weekly_rest", "duration",
          rf"(?P<v>{_DURATION})\s*(?:of\s+)?weekly\s+(?:off|rest)"),

    # ---------------------------------------------------------------- leave (bound to a year)
    _rule("annual_leave", "leave",
          rf"(?:you\s+(?:will|shall)\s+(?:get|be\s+entitled\s+to|receive|be\s+eligible\s+for)|"
          rf"entitled\s+to|eligible\s+for)\s+(?P<v>{_NUM}\s*(?:paid\s+|annual\s+|calendar\s+)*"
          rf"(?:days?\s+(?:of\s+)?)?(?:leave|days|vacation|holidays?))"
          rf"[^.]{{0,40}}?(?:per\s+annum|per\s+year|a\s+year|annually|each\s+year)"),
    _rule("annual_leave", "leave",
          rf"annual\s+(?:paid\s+)?(?:leave|vacation)\s*(?:entitlement)?\s*"
          rf"(?:is|shall\s+be|will\s+be|of|:|-)\s*(?P<v>{_DURATION})"),

    # ---------------------------------------------------------------- pay
    _rule("in_hand_salary", "money",
          rf"in[\s-]*hand\s*(?:salary)?\s*(?P<period>monthly|per\s+month|a\s+month|annually|per\s+annum)?"
          rf"\s*[-–:]?\s*(?P<v>{_MONEY})",
          max_chars=40),
    _rule("salary", "money",
          rf"(?:basic\s+(?:salary|pay|wage)|monthly\s+(?:basic\s+)?salary)\s*"
          rf"(?:shall\s+be|will\s+be|is|of|:|-)\s*(?P<v>{_MONEY})",
          max_chars=40),
    _rule("total_salary", "money",
          rf"(?:gross\s+(?:salary|pay)|total\s+(?:salary|compensation)|CTC)\s*"
          rf"(?:shall\s+be|will\s+be|is|of|:|-)\s*(?P<v>{_MONEY})",
          max_chars=40),
)

_BY_FIELD: dict[str, list[NarrativeRule]] = {}
for _r in NARRATIVE_RULES:
    _BY_FIELD.setdefault(_r.field, []).append(_r)

SEMANTIC_FIELDS: tuple[str, ...] = tuple(_BY_FIELD)


# --------------------------------------------------------------------------- role binding, exported
# A sentence matching one of these states some *other* field's duration, even though it mentions
# this field's keyword. Both languages are listed: the guard must not silently disable itself on an
# Arabic contract just because the narrative rules above are written for English prose.
# A sentence describing a payroll *deduction* ("Deducted at a rate of (9.75%) of the basic wage
# plus housing allowance = SAR 1,218.75") mentions "housing allowance" only to compute what is taken
# out of it; the number it states is the deduction, not the allowance itself. This is an ordinary,
# generic GOSI/social-insurance clause shape found across Saudi contracts, not one document's wording.
_DEDUCTION_SENTENCE = (
    r"\bdeduct(?:ed|ion|ions)?\b", r"\bwithhold(?:ing|s)?\b", r"\bGOSI\b",
    r"\bsocial\s+insurance\b", r"\b(?:insurance|hazard)\s+subscription\b", r"\brate\s+of\s*\(?\s*\d",
    r"يخصم", r"خصم", r"استقطاع", r"اشتراك", r"(?:التأمينات|التامينات)\s+الاجتماعي",
)
# A sentence about *postponing* or deferring leave ("the right to postpone the annual leave ... for
# a period not exceeding (90) days") states a limit on deferral, not the leave entitlement itself.
_POSTPONEMENT_SENTENCE = (
    r"\bpostpone(?:ment|d|s)?\b", r"\bdefer(?:ral|red|s)?\b", r"\bcarr(?:y|ied)\s+(?:over|forward)\b",
    r"يؤجل", r"(?:تأجيل|تاجيل)", r"ترحيل",
)

_DISQUALIFIERS: dict[str, tuple[re.Pattern[str], ...]] = {
    "probation_period": tuple(re.compile(p, _flags()) for p in (
        r"\bnotice\b", r"in\s+lieu\s+(?:there)?of", r"\bterminate\b", r"\btermination\b",
        r"\bleave\b", r"\bpaid\s+days?\b", r"\bcasual\b", r"\bentitled\s+to\b", r"\beligible\s+for\b",
        r"إشعار", r"إنذار", r"إنهاء", r"فسخ", r"إجازة",
    )),
    "housing_allowance": tuple(re.compile(p, _flags()) for p in _DEDUCTION_SENTENCE),
    "annual_leave": tuple(re.compile(p, _flags()) for p in _POSTPONEMENT_SENTENCE),
}


def sentence_allows(field: str, text: str) -> bool:
    """Whether a keyword-matched sentence may contribute a value to `field`.

    Used by the label/keyword extractor. A sentence is refused only when it carries a cue belonging
    to a *different* field ("...90 days' notice") and does not itself state this field's value in a
    subject-bound form ("the period of probation is 180 days"). Fields with no disqualifier, and
    sentences with no competing cue, are unaffected - so contracts that the keyword scan already
    read correctly, in any language, keep behaving exactly as before.
    """
    disqualifiers = _DISQUALIFIERS.get(field)
    if not disqualifiers or not any(bad.search(text) for bad in disqualifiers):
        return True
    return any(rule.search(text) is not None for rule in _BY_FIELD.get(field, ()))


# --------------------------------------------------------------------------- explicit negation ("F")
# A document that states a topic does not apply ("not subject to a probationary period", "لا يخضع
# ... لفترة تجربة") is not silent about it: reporting NOT_FOUND there would misrepresent a document
# that in fact addresses the topic directly. `detect_negations` looks for that explicit statement and
# turns it into NOT_APPLICABLE, with the statement itself as the field's source - never a guessed value.
#
# The Arabic check is deliberately word-order independent (the keyword and a negation marker must
# appear in the same unit, not in one fixed phrase): PDF text layers extracted from a right-to-left
# run are frequently word-reordered or glyph-broken by the exporting tool (seen directly in the real
# bilingual fixtures this was built against - e.g. "6.1 The Second Party is not subject to a
# probationary period" comes through as "6.1 [entire clause, word order scrambled]" on the Arabic
# side), so a single fixed Arabic phrase would silently stop matching the moment a document's text
# layer reorders it. English negation markers keep normal reading order, since English text layers are
# not subject to the same corruption, so those stay precise, ordered phrase patterns.
@dataclass(frozen=True)
class NegationRule:
    field: str
    topic_keys: tuple[str, ...]              # the topic, for the order-independent Arabic check
    english: tuple[re.Pattern[str], ...] = ()  # precise, ordered phrase match against the raw sentence
    arabic_markers: tuple[re.Pattern[str], ...] = ()  # topic + any one of these, same unit, any order


def _ar(word: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(word)}(?!\w)")


NEGATION_RULES: tuple[NegationRule, ...] = (
    NegationRule(
        field="probation_period",
        topic_keys=("probation", "تجربة"),
        english=tuple(re.compile(p, _flags()) for p in (
            r"\bnot\s+subject\s+to\s+(?:a\s+|any\s+)?probation",
            r"\bno\s+probation(?:ary)?\s+period\b",
            r"\bwithout\s+(?:a\s+)?probation(?:ary)?\s+period\b",
            r"\bprobation(?:ary)?\s+period\b[^.]{0,40}\bdoes\s+not\s+apply\b",
        )),
        arabic_markers=(_ar("لا"), _ar("غير"), _ar("دون")),
    ),
)


def detect_negations(document: ParsedDocument) -> dict[str, ExtractedField]:
    """One ExtractedField (status NOT_APPLICABLE) per field explicitly negated somewhere in the
    document. Additive and independent of the label/narrative paths - see `NEGATION_RULES` above."""
    found: dict[str, ExtractedField] = {}
    for unit in (*iter_sentences(document), *iter_lines(document)):
        text = unit.text
        if not text.strip():
            continue
        text_key = matching_key(text)
        for rule in NEGATION_RULES:
            if rule.field in found:
                continue
            hit = any(pattern.search(text) for pattern in rule.english)
            if not hit and rule.arabic_markers and any(
                contains_keyword(text_key, matching_key(key)) for key in rule.topic_keys
            ):
                hit = any(marker.search(text) for marker in rule.arabic_markers)
            if hit:
                source = unit.source(ExtractionMethodName.PATTERN)
                found[rule.field] = not_applicable(
                    rule.field, source, "the document explicitly states this does not apply")
    return found


# --------------------------------------------------------------------------- value builders
def _duration(raw: str) -> tuple[Any | None, list[str]]:
    readings = readers.find_durations(raw, allow_bare_singular=True)
    if not readings or readings[0].value is None:
        return None, ["a duration could not be read from the matched text"]
    return DurationValue(**readings[0].value), list(readings[0].notes)


def _date(raw: str) -> tuple[Any | None, list[str]]:
    readings = readers.find_dates(raw)
    if not readings or readings[0].value is None:
        return None, ["a date could not be read from the matched text"]
    return DateValue(**readings[0].value), list(readings[0].notes)


def _money(raw: str) -> tuple[Any | None, list[str]]:
    amounts, _ = readers.find_money(raw, require_currency=False)
    if not amounts or amounts[0].value is None:
        return None, ["an amount could not be read from the matched text"]
    return MoneyValue(**amounts[0].value), list(amounts[0].notes)


def _hours(raw: str) -> tuple[Any | None, list[str]]:
    readings = readers.find_hours(raw)
    if not readings or readings[0].value is None:
        return None, ["working hours could not be read from the matched text"]
    value = readings[0].value
    return WorkingHoursValue(hours_per_day=value["hours"] if value["period"] == "day" else None,
                             hours_per_week=value["hours"] if value["period"] == "week" else None,
                             hours=value["hours"] if value["period"] is None else None), list(readings[0].notes)


def _days(raw: str) -> tuple[Any | None, list[str]]:
    readings = readers.find_working_days(raw)
    if not readings or readings[0].value is None:
        return None, ["working days could not be read from the matched text"]
    return WorkingDaysValue(**readings[0].value), list(readings[0].notes)


def _text(raw: str) -> tuple[Any | None, list[str]]:
    cleaned = " ".join(raw.split()).strip(" .,;:-–")
    return (cleaned, []) if cleaned else (None, ["the matched text is empty"])


_BARE_LEAVE = re.compile(rf"^\s*(?P<n>{_NUM})\b", _flags())
_NUMBER_WORD_RE = r"(?<![a-z])(?:" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")(?![a-z])"


def _leave(raw: str) -> tuple[Any | None, list[str]]:
    """Leave entitlements are often written without a unit ("24 paid leave per annum").

    A duration is read normally when the document states a unit. When it does not, the count is kept
    and the unit recorded as days - the only unit leave is counted in - with an explicit note, so the
    assumption is visible to the reader rather than silently baked into the value.
    """
    value, notes = _duration(raw)
    if value is not None:
        return value, notes
    # The fallback only applies to a plainly written single count. "thirty (21) days" states two
    # different numbers, which is a real contradiction in the document: it must stay unreadable here
    # so the field resolves to ambiguous, rather than being quietly settled in favour of one of them.
    if len(re.findall(r"\d+(?:\.\d+)?", raw)) + len(re.findall(_NUMBER_WORD_RE, raw)) > 1:
        return None, ["the matched text states more than one number"]
    match = _BARE_LEAVE.search(raw)
    if match is None:
        return None, ["a leave entitlement could not be read from the matched text"]
    count = _NUMBER_WORDS.get(match.group("n").lower())
    if count is None:
        try:
            count = float(match.group("n"))
        except ValueError:
            return None, ["a leave entitlement could not be read from the matched text"]
    return (DurationValue(count=float(count), unit="day"),
            [f"the document writes '{' '.join(raw.split())}' without a unit; read as days"])


_BUILDERS: dict[str, Callable[[str], tuple[Any | None, list[str]]]] = {
    "duration": _duration, "date": _date, "money": _money,
    "hours": _hours, "days": _days, "text": _text, "leave": _leave,
}

_PERIOD_WORDS = {"monthly": "monthly", "per month": "monthly", "a month": "monthly",
                 "annually": "annual", "per annum": "annual"}


# --------------------------------------------------------------------------- extractor
@dataclass
class SemanticExtractor:
    """Narrative extraction over a parsed document. Additive: it never removes a label-derived value."""

    rules: tuple[NarrativeRule, ...] = NARRATIVE_RULES

    def extract_fields(self, document: ParsedDocument) -> dict[str, ExtractedField]:
        found: dict[str, list[FieldCandidate | None]] = {name: [] for name in _BY_FIELD}
        confidences: dict[str, set[str]] = {name: set() for name in _BY_FIELD}
        notes: dict[str, list[str]] = {name: [] for name in _BY_FIELD}

        seen: set[tuple[str, str, str]] = set()
        for unit in self._units(document):
            text = unit.text
            for rule in self.rules:
                match = rule.search(text)
                if match is None:
                    continue
                # Line breaks inside a wrapped PDF line are collapsed; the wording itself is untouched,
                # and `candidate` still compares against the source text on flattened whitespace.
                raw = " ".join(match.group("v").split())
                value, build_notes = _BUILDERS[rule.kind](raw)
                value, period_notes = _apply_period(value, match)
                source = unit.source(ExtractionMethodName.PATTERN)
                item = candidate(value, raw, source, [*build_notes, *period_notes])
                if item is None:  # the reading is not literally in the source text: never guess
                    continue
                # The same fact is usually reachable both as a sentence and as a line; one reading is
                # enough. Two *different* readings are kept, so a real conflict still shows as ambiguous.
                marker = (rule.field, raw.casefold(), repr(value))
                if marker in seen:
                    continue
                seen.add(marker)
                found[rule.field].append(item)
                confidences[rule.field].add(rule.confidence)
                if rule.note:
                    notes[rule.field].append(rule.note)

        fields: dict[str, ExtractedField] = {}
        for name, items in found.items():
            if not items:
                continue
            resolved = resolve(name, items)
            extra = [n for n in notes[name] if n]
            if extra or confidences[name]:
                resolved = resolved.model_copy(update={
                    "notes": list(dict.fromkeys([*resolved.notes, *extra])),
                    "confidence": _weakest(resolved.confidence, confidences[name]),
                })
            fields[name] = resolved

        benefits = self._benefits(document)
        if benefits is not None:
            fields["benefits"] = benefits
        return fields

    # ------------------------------------------------------------------ units
    @staticmethod
    def _units(document: ParsedDocument) -> Iterator[TextUnit]:
        """Sentences (prose) plus every line (headings and list items carry facts too)."""
        yield from iter_sentences(document)
        yield from iter_lines(document)

    # ------------------------------------------------------------------ benefits
    @staticmethod
    def _benefits(document: ParsedDocument) -> ExtractedField | None:
        cue = re.compile(r"\bbenefits?\b|\bperks?\b", _flags())
        items: list[ListItem] = []
        sources: list[SourceSpan] = []
        collecting = False
        for unit in iter_lines(document):
            text = unit.text.strip()
            if not text:
                continue
            bullet = text[:1] in BULLET_CHARS
            if cue.search(text) and not bullet:
                collecting = True
                continue
            if not collecting:
                continue
            if bullet:
                entry = text.lstrip(BULLET_CHARS).strip()
                if entry and not any(i.text == entry for i in items):
                    source = unit.source(ExtractionMethodName.SECTION_CONTENT)
                    items.append(ListItem(text=entry, source=source))
                    sources.append(source)
            else:
                collecting = False
        if not items:
            return None
        return resolve_list("benefits", items, sources)


def _apply_period(value: Any, match: re.Match[str]) -> tuple[Any, list[str]]:
    """Attach a pay period the rule captured outside the value span ("In hand monthly- 1800 Riyal")."""
    if not isinstance(value, MoneyValue) or value.period is not None:
        return value, []
    written = (match.groupdict().get("period") or "").strip().casefold()
    period = _PERIOD_WORDS.get(" ".join(written.split()))
    if period is None:
        return value, []
    return value.model_copy(update={"period": "monthly" if period == "monthly" else "annual"}), []


def _weakest(resolved: str | None, seen: set[str]) -> str | None:
    """A field is only as confident as its least confident contributing rule."""
    order = ["low", "medium", "high"]
    levels = [c for c in seen if c in order]
    if resolved in order:
        levels.append(resolved)
    return min(levels, key=order.index) if levels else resolved


def extract_semantic_fields(document: ParsedDocument) -> dict[str, ExtractedField]:
    return SemanticExtractor().extract_fields(document)
