"""Stage 3: structured legal rules and deterministic comparison.

CONTRACT CLAUSE -> EXTRACTED FACT -> REGULATORY EVIDENCE -> STRUCTURED LEGAL RULE
-> DETERMINISTIC COMPARISON -> STATUS -> EXPLANATION

Nothing here asks an LLM whether a clause is compliant. A LegalRule is read from the retrieved
article's own Arabic text by pattern matching only (`derive_legal_rule`); when the text does not
match a known, testable pattern, the rule is UNKNOWN and the finding is REQUIRES_REVIEW - never a
guess. The arithmetic in `compare_fact_to_rule` and the finite state table in
`_status_from_comparison` are the only things that ever produce COMPLIANT or NON_COMPLIANT. An
optional `explain` hook may be passed to `build_clause_legal_finding` to reword the explanation
(e.g. with an LLM); it is called after the status is already fixed and can only change the
explanation text - see `build_clause_legal_finding` and `_apply_explainer`.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from agents.regulatory import CONTRACT_TOPICS, RegulatoryTopic
from extraction.clauses import FIELD_CLAUSE_TYPES
from extraction.text import matching_key
from models.analysis import ClauseCheck, ClauseLegalFinding, EvidenceReference, FindingStatus
from models.extraction import ClauseType, ClauseValue, ContractExtraction, ExtractedField, FieldStatus
from models.legal_rules import Comparator, ComparisonResult, ConditionStatus, ContractFact, LegalRule, RuleCondition, RuleType

# --------------------------------------------------------------------------- Arabic number words
# Deliberately small and closed: a word this table does not recognise is never guessed at, and a
# rule that would need it stays UNKNOWN (see _parse_number_phrase). Keys are post-matching_key forms
# (diacritics stripped, alef/ta-marbuta unified, case-folded) of the words Saudi Labor Law actually
# uses for the thresholds this module knows about.
_ONES: dict[str, int] = {
    "واحد": 1, "احد": 1, "اثنان": 2, "اثنين": 2, "ثلاثه": 3, "ثلاث": 3, "اربعه": 4, "اربع": 4,
    "خمسه": 5, "خمس": 5, "سته": 6, "ست": 6, "سبعه": 7, "سبع": 7, "ثمانيه": 8, "ثمان": 8, "ثمانى": 8, "ثماني": 8,
    "تسعه": 9, "تسع": 9,
}
_TENS: dict[str, int] = {
    "عشره": 10, "عشر": 10, "عشرون": 20, "عشرين": 20, "ثلاثون": 30, "ثلاثين": 30,
    "اربعون": 40, "اربعين": 40, "خمسون": 50, "خمسين": 50, "ستون": 60, "ستين": 60,
    "سبعون": 70, "سبعين": 70, "ثمانون": 80, "ثمانين": 80, "تسعون": 90, "تسعين": 90,
}
_HUNDREDS: dict[str, int] = {"مائه": 100, "مئه": 100, "مايه": 100}  # matching_key folds hamza (مائة) to ي: "مايه"

# unit word (normalised) -> canonical unit. "ساعه/ساعات" is refined to hour_per_day/hour_per_week
# by _refine_hour_unit() when the sentence says which standard it is.
_UNIT_WORDS: dict[str, str] = {
    "يوما": "day", "يوم": "day", "ايام": "day",
    "اسبوع": "week", "اسابيع": "week",
    "ساعه": "hour", "ساعات": "hour",
    "شهر": "month", "اشهر": "month",
    "سنه": "year", "سنوات": "year",
}
_UNIT_PATTERN = "|".join(sorted(_UNIT_WORDS, key=len, reverse=True))


def _parse_number_phrase(phrase: str) -> int | None:
    """A small, closed Arabic-number-word parser. Returns None rather than guessing."""
    joined = "".join(phrase.split())
    if joined.isdigit():
        return int(joined)
    total, matched = 0, False
    for token in phrase.split():
        token = token[1:] if token.startswith("و") and len(token) > 1 else token  # strip the "and" clitic
        if token in _HUNDREDS:
            total, matched = total + _HUNDREDS[token], True
        elif token in _TENS:
            total, matched = total + _TENS[token], True
        elif token in _ONES:
            total, matched = total + _ONES[token], True
    return total if matched else None


def _refine_hour_unit(text: str, match_end: int) -> str:
    tail = text[match_end: match_end + 20]
    if re.search(r"\bاليوم\b", tail):
        return "hour_per_day"
    if re.search(r"\bالاسبوع\b", tail):
        return "hour_per_week"
    return "hour"


_NUMBER_PHRASE = rf"([^\d.,;()]{{1,20}}?)\s+({_UNIT_PATTERN})\b"

# "لا/الا ... يزيد/تزيد ... على/عن NUM UNIT" and "اكثر من NUM UNIT" both state a MAXIMUM.
# Patterns are written with "علي" (ya), not "على" (alef maksura): matching_key() folds the maksura
# alef to ya (see extraction.text._LETTER_MAP), so the normalised text this module matches against
# always spells it that way.
_MAX_PATTERNS = (
    re.compile(rf"(?:يزيد|تزيد)[^.]{{0,40}}?(?:علي|عن)\s+{_NUMBER_PHRASE}"),
    re.compile(rf"اكثر\s+من\s+{_NUMBER_PHRASE}"),
)
# "لا/ولا ... يقل/تقل ... عن NUM UNIT" and "قبل NUM UNIT علي الاقل" both state a MINIMUM.
_MIN_PATTERNS = (
    re.compile(rf"(?:يقل|تقل)[^.]{{0,40}}?عن\s+{_NUMBER_PHRASE}"),
    re.compile(rf"قبل\s+{_NUMBER_PHRASE}\s+علي\s+الاقل"),
)
_PROHIBITION = re.compile(r"لا\s+يجوز")
_REQUIREMENT = re.compile(r"\bيجب\b|\bيلزم\b|\bتلزم\b")
# A genuine condition-precedent (not merely "which of two flat thresholds applies"): the rule only
# holds if some other fact - stated nowhere in a duration/money value - is also true.
_CONDITION_PRECEDENT = re.compile(r"الا\s+(?:اذا|بموافقه|بشرط)|بموافقه\s+كتابيه|بشرط\s+موافقه")
_WRITTEN_AGREEMENT_CUES = re.compile(r"written|in\s+writing|agreed|consent|مكتوب|كتاب[ةه]|موافق")


def _find_numeric_thresholds(text: str, patterns: tuple[re.Pattern[str], ...]) -> list[tuple[float, str]]:
    found: list[tuple[float, str]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            phrase, unit_word = match.group(1), match.group(2)
            number = _parse_number_phrase(phrase)
            if number is None:
                continue
            unit = _UNIT_WORDS[unit_word]
            if unit == "hour":
                unit = _refine_hour_unit(text, match.end())
            found.append((float(number), unit))
    # de-duplicate identical (value, unit) pairs while keeping first-seen order
    seen: set[tuple[float, str]] = set()
    unique = []
    for item in found:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def derive_legal_rule(rule_id: str, topic: str, evidence: EvidenceReference) -> LegalRule:
    """A deterministic reading of one retrieved article. Never invents a number (see model validator)."""
    text = matching_key(evidence.arabic_text or "")
    base = dict(rule_id=rule_id, topic=topic, article_citation=evidence.citation,
               article_number=evidence.article_number, source_evidence_id=evidence.evidence_id,
               source_text=evidence.arabic_text)

    maxima = _find_numeric_thresholds(text, _MAX_PATTERNS)
    minima = _find_numeric_thresholds(text, _MIN_PATTERNS)
    condition_precedent = _CONDITION_PRECEDENT.search(text) is not None

    if maxima and minima and len(maxima) == 1 and len(minima) == 1 and maxima[0][1] == minima[0][1]:
        (max_v, unit), (min_v, _) = maxima[0], minima[0]
        return LegalRule(**base, rule_type=RuleType.RANGE, comparator=Comparator.RANGE,
                         value=min(min_v, max_v), value_max=max(min_v, max_v), unit=unit, confidence="medium")

    all_thresholds = [("maximum", Comparator.LE, v, u) for v, u in maxima] + \
                     [("minimum", Comparator.GE, v, u) for v, u in minima]

    if condition_precedent and len(all_thresholds) == 1:
        kind, comparator, value, unit = all_thresholds[0]
        condition = RuleCondition(
            description=f"The evidence states this {kind} ({value:g} {unit}) applies only under a stated condition "
                        "(e.g. the worker's written agreement); the retrieved text does not itself confirm whether "
                        "that condition is met.",
            value=value, unit=unit, comparator=comparator)
        return LegalRule(**base, rule_type=RuleType.CONDITIONAL, comparator=Comparator.CONDITIONAL,
                         conditions=[condition], confidence="medium",
                         notes=["A condition precedent was detected in the article text; see the condition's own "
                                "value/unit for the numeric threshold it attaches to."])

    if len(all_thresholds) == 1:
        kind, comparator, value, unit = all_thresholds[0]
        rule_type = RuleType.MAXIMUM if kind == "maximum" else RuleType.MINIMUM
        return LegalRule(**base, rule_type=rule_type, comparator=comparator, value=value, unit=unit,
                         confidence="high")

    if len(all_thresholds) > 1:
        conditions = [
            RuleCondition(description=f"A {kind} of {value:g} {unit} applies under the circumstances this "
                                      f"sub-clause of the article describes.",
                         value=value, unit=unit, comparator=comparator)
            for kind, comparator, value, unit in all_thresholds
        ]
        return LegalRule(**base, rule_type=RuleType.CONDITIONAL, comparator=Comparator.CONDITIONAL,
                         conditions=conditions, confidence="medium",
                         notes=[f"The article states {len(conditions)} distinct thresholds; which applies depends "
                                "on facts (e.g. which party is acting, or length of service) that a single clause "
                                "does not by itself establish. Each is checked against the contract value on its own."])

    if _PROHIBITION.search(text):
        return LegalRule(**base, rule_type=RuleType.PROHIBITION, comparator=Comparator.PROHIBITED, confidence="medium")
    if _REQUIREMENT.search(text):
        return LegalRule(**base, rule_type=RuleType.REQUIREMENT, comparator=Comparator.REQUIRED, confidence="medium")

    return LegalRule(**base, rule_type=RuleType.UNKNOWN, comparator=Comparator.UNKNOWN, confidence="low",
                     notes=["The retrieved article text did not match a known deterministic pattern; a human "
                            "should compare it with the contract clause directly."])


# --------------------------------------------------------------------------- contract fact
def _field_value_unit(value) -> tuple[float, str] | None:
    name = type(value).__name__
    if name == "DurationValue":
        return float(value.count), value.unit
    if name == "WorkingHoursValue":
        if value.hours_per_week is not None:
            return float(value.hours_per_week), "hour_per_week"
        if value.hours_per_day is not None:
            return float(value.hours_per_day), "hour_per_day"
        if value.hours is not None:
            return float(value.hours), "hour"
        return None
    if name in ("MoneyValue", "AllowanceValue") and getattr(value, "amount", None) is not None:
        currency = (getattr(value, "currency", None) or "unknown").lower()
        period = getattr(value, "period", None) or "unspecified"
        return float(value.amount), f"{currency}_{period}"
    return None


def _candidate_readings(field: ExtractedField) -> list[tuple[object, str, object]]:
    """(value, raw_value, source) this field could offer a clause, most authoritative reading first.

    Reads AMBIGUOUS candidates too: a document-wide conflict between two clauses does not make
    either clause's own number unclear (Stage 2 keeps both clauses; Stage 3 keeps both facts).
    """
    if field.status is FieldStatus.FOUND and field.sources:
        return [(field.value, field.raw_value, field.sources[0])]
    if field.status is FieldStatus.AMBIGUOUS:
        return [(c.value, c.raw_value, c.source) for c in field.candidates if c.value is not None]
    return []


# Stage 4.1 linkage hardening: fields Stage 1 extracts but extraction.clauses does not (yet) give
# their own narrative clause type - most often because the sentence that states them is a short,
# table-derived label/value pair rather than a narrative sentence a NARRATIVE_RULE recognises (e.g.
# "9.1.1.2 Housing Allowance" and its amount are often two separate rows). This dict is used only to
# decide which already-extracted field a clause's fact might be, never to reclassify the clause
# itself - extraction.clauses.FIELD_CLAUSE_TYPES (and the clause's own clause_type) are untouched.
_LINKING_FIELD_TYPES: dict[str, ClauseType] = {
    **FIELD_CLAUSE_TYPES,
    "housing_allowance": ClauseType.HOUSING,
    "transportation_allowance": ClauseType.TRANSPORTATION,
    "other_allowances": ClauseType.ALLOWANCES,
    "net_salary": ClauseType.SALARY,
    "notice_period": ClauseType.NOTICE,
    "termination_terms": ClauseType.TERMINATION,
    "benefits": ClauseType.BENEFITS,
}


def _text_matches(source_text: str | None, clause_text: str) -> bool:
    """Whether a field's source text and a clause's text are the same underlying words.

    `matching_key` (not just whitespace-flattening) absorbs punctuation/diacritic differences
    between the two independent text walks that produced them; both directions of containment are
    checked because a table-row clause's text is sometimes shorter than the field's own source
    sentence, and sometimes longer (a row with more than one label in it).
    """
    if not source_text or not clause_text:
        return False
    source_key, clause_key = matching_key(source_text), matching_key(clause_text)
    return bool(source_key) and (source_key in clause_key or clause_key in source_key)


def _source_within_clause(source, clause: ClauseValue) -> bool:
    """Whether a field's source location is verifiably the same place as this clause - by the
    provenance the parser already stamped on both, never by proximity, page or a guess.

    A table's `table_id` identifies the whole table, not one row, so a table_id match alone is not
    enough to pick the right row-clause among several sharing it; text overlap (see _text_matches)
    still disambiguates which row. A section_id, by contrast, already identifies one specific
    paragraph/section 1:1 with (at most) one clause, so it needs no further check.
    """
    span = clause.source_span
    if span is None:
        return False
    if source.table_id is not None:
        return source.table_id == span.table_id and _text_matches(source.text, clause.text)
    if source.section_id is not None:
        return source.section_id == span.section_id
    return False


def _structural_candidates(clause: ClauseValue, contract: ContractExtraction) -> list[tuple[str, object, str, object]]:
    """Fields linked to this clause by provenance alone, when the clause's own text did not (yet)
    contain a recognisable match for any field of its declared type.

    Only tried for a clause the segmenter could not confidently classify (OTHER/UNKNOWN - the common
    shape for a table row or short bilingual fragment whose own words never matched a narrative
    rule), or for a field whose expected type already agrees with this clause's own declared type
    (the clause IS the right kind, but its text was too garbled for a plain containment check to
    see it - a known symptom on the two dense bilingual PDF fixtures). A confidently-classified
    clause of a DIFFERENT type is never touched: this only recovers linkage, it never overrides a
    classification the segmenter already made with confidence.
    """
    found: list[tuple[str, object, str, object]] = []
    for name, expected_type in _LINKING_FIELD_TYPES.items():
        if clause.clause_type not in (ClauseType.OTHER, ClauseType.UNKNOWN) and expected_type is not clause.clause_type:
            continue
        field = getattr(contract, name, None)
        if not isinstance(field, ExtractedField):
            continue
        for value, raw_value, source in _candidate_readings(field):
            if _source_within_clause(source, clause):
                found.append((name, value, raw_value, source))
    return found


def _readings_within_clause(clause: ClauseValue, contract: ContractExtraction) -> list[tuple[str, object, str, object]]:
    """Every (field_name, value, raw_value, source) this clause's own words support - not just the first.

    Almost always zero or one: Stage 2 never merges two classified sentences into one clause (see
    extraction/clauses.py), so a single clause offering two DIFFERENT numeric readings for its own
    field is a genuine internal contradiction, not a cross-clause disagreement - see
    _distinct_numeric_readings(), which is what AMBIGUOUS (section 11) is reserved for.

    Two tiers, both deterministic and provenance-only (never a re-parse, never a guess): first, the
    original narrative match (this clause's declared type -> the fields that type can state -> text
    containment); when that finds nothing, a structural fallback keyed on the same section_id/
    table_id the parser already stamped on both the field and the clause (see _structural_candidates).
    """
    field_names = [name for name, clause_type in _LINKING_FIELD_TYPES.items() if clause_type is clause.clause_type]
    found: list[tuple[str, object, str, object]] = []
    for name in field_names:
        field = getattr(contract, name, None)
        if not isinstance(field, ExtractedField):
            continue
        for value, raw_value, source in _candidate_readings(field):
            if _text_matches(source.text, clause.text):
                found.append((name, value, raw_value, source))
    if found:
        return found
    return _structural_candidates(clause, contract)


def linked_field_for_clause(clause: ClauseValue, contract: ContractExtraction) -> str | None:
    """The one Stage 1 field this clause's own provenance identifies, when the clause itself was too
    weakly classified (OTHER/UNKNOWN) for extraction.clauses to say what it is about.

    Used only to decide whether a clause deserves a regulatory question at all (see
    agents.contract_analysis.ContractAnalysisAgent._clause_checks) - it is the exact same structural
    check derive_contract_fact/_structural_candidates already rely on to attach a ContractFact, never
    a new guess. Returns None (never a guess between two) when more than one distinct field's
    provenance lands on this clause, since that would be a real ambiguity about the clause's subject,
    not a structural fact.
    """
    if clause.clause_type not in (ClauseType.OTHER, ClauseType.UNKNOWN):
        return None
    names = {name for name, *_ in _structural_candidates(clause, contract)}
    return next(iter(names)) if len(names) == 1 else None


def _distinct_numeric_readings(clause: ClauseValue, contract: ContractExtraction) -> set[tuple[float, str]]:
    values = {_field_value_unit(value) for _, value, _, _ in _readings_within_clause(clause, contract)}
    values.discard(None)
    return values


def derive_contract_fact(clause: ClauseValue, contract: ContractExtraction) -> ContractFact | None:
    """The Stage 1 extracted value that this clause's own words produced, if any (the first match;

    callers that need to detect an internal contradiction use _distinct_numeric_readings() first).
    Never re-parses the document: it only reunites provenance chains Stage 1 and Stage 2 already
    built independently, first by text (see _text_matches) and, when that finds nothing, by the
    parser's own section_id/table_id stamped on both sides (see _structural_candidates). Neither
    tier ever invents a value or a location - a clause with no match of either kind returns None.
    """
    for name, value, raw_value, source in _readings_within_clause(clause, contract):
        unit_value = _field_value_unit(value)
        normalized_value, unit = unit_value if unit_value else (None, None)
        field = getattr(contract, name)
        return ContractFact(clause_id=clause.clause_id or "C00", field=name, raw_value=raw_value,
                            normalized_value=normalized_value, unit=unit, source_text=source.text,
                            page_number=source.page_number, confidence=field.confidence)
    return None


# --------------------------------------------------------------------------- unit normalization
# A deliberately small, explicit table (section 13). Two quantities are only ever compared when they
# are the same kind of thing: a day-count and an hour-count are NOT interchangeable here even though
# both are "time", because a legal minimum stated in continuous hours (e.g. the weekly rest day) does
# not become a day-count by dividing by 24 - that would invent a claim the law never made. The same
# reasoning rules out an hour_per_day <-> hour_per_week conversion: turning "40 hours/week" into an
# hours/day figure needs an assumed number of working days the contract does not state, so a rule
# given in one of those units and a fact given in the other are left as incompatible (REQUIRES_REVIEW)
# rather than silently divided or multiplied by an invented schedule.
_UNIT_CONVERSIONS: dict[tuple[str, str], float] = {
    ("day", "week"): 1 / 7,
    ("month", "year"): 1 / 12,
}


def convert(value: float, from_unit: str, to_unit: str) -> float | None:
    if from_unit == to_unit:
        return value
    factor = _UNIT_CONVERSIONS.get((from_unit, to_unit))
    if factor is not None:
        return value * factor
    inverse = _UNIT_CONVERSIONS.get((to_unit, from_unit))
    if inverse is not None:
        return value / inverse
    if from_unit.startswith(("sar_", "riyal_")) or to_unit.startswith(("sar_", "riyal_")):
        if from_unit.endswith("_monthly") and to_unit.endswith("_annual") and from_unit.split("_")[0] == to_unit.split("_")[0]:
            return value * 12
        if from_unit.endswith("_annual") and to_unit.endswith("_monthly") and from_unit.split("_")[0] == to_unit.split("_")[0]:
            return value / 12
    return None


_APPLY = {
    Comparator.GE: lambda v, t: v >= t, Comparator.LE: lambda v, t: v <= t, Comparator.EQ: lambda v, t: v == t,
    Comparator.GT: lambda v, t: v > t, Comparator.LT: lambda v, t: v < t,
}


def _evaluate_condition(fact: ContractFact | None, condition: RuleCondition) -> RuleCondition:
    if condition.value is None or condition.unit is None or condition.comparator is None:
        cue = bool(fact and fact.source_text and _WRITTEN_AGREEMENT_CUES.search(fact.source_text))
        return condition.model_copy(update={"status": ConditionStatus.SATISFIED if cue else ConditionStatus.NOT_STATED})
    if fact is None or fact.normalized_value is None or fact.unit is None:
        return condition.model_copy(update={"status": ConditionStatus.NOT_STATED})
    converted = convert(fact.normalized_value, fact.unit, condition.unit)
    if converted is None:
        return condition.model_copy(update={
            "status": ConditionStatus.NOT_APPLICABLE,
            "notes": [*condition.notes, f"the contract states this in '{fact.unit}', not '{condition.unit}'; not evaluated"]})
    holds = _APPLY[condition.comparator](converted, condition.value)
    return condition.model_copy(update={"status": ConditionStatus.SATISFIED if holds else ConditionStatus.NOT_SATISFIED})


def compare_fact_to_rule(fact: ContractFact | None, rule: LegalRule) -> ComparisonResult:
    """Pure arithmetic: what was compared, with what, and whether it held. No verdict is decided here."""
    if rule.rule_type is RuleType.CONDITIONAL:
        evaluated = [_evaluate_condition(fact, c) for c in rule.conditions]
        applicable = [c for c in evaluated if c.status is not ConditionStatus.NOT_APPLICABLE]
        satisfied = None
        if applicable and all(c.status is ConditionStatus.SATISFIED for c in applicable):
            satisfied = True
        elif applicable and all(c.status is ConditionStatus.NOT_SATISFIED for c in applicable):
            satisfied = False
        return ComparisonResult(
            comparator=rule.comparator, contract_value=fact.normalized_value if fact else None, unit=rule.unit,
            condition_results=evaluated, satisfied=satisfied,
            explanation="; ".join(f"{c.description} -> {c.status.value}" for c in evaluated) or "no conditions evaluated")

    if rule.rule_type in (RuleType.PROHIBITION, RuleType.REQUIREMENT, RuleType.INFORMATIONAL, RuleType.UNKNOWN):
        return ComparisonResult(
            comparator=rule.comparator, contract_value=fact.normalized_value if fact else None, satisfied=None,
            explanation=("The retrieved article did not state a deterministic pattern to check." if rule.rule_type is RuleType.UNKNOWN
                        else "This article states a qualitative rule (a prohibition or a procedural requirement), "
                             "not a numeric threshold, so no amount comparison applies to it."))

    if fact is None or fact.normalized_value is None or fact.unit is None:
        return ComparisonResult(comparator=rule.comparator, unit=rule.unit, satisfied=None,
                                explanation="No numeric contract fact is available for this clause to compare against the rule.")

    converted = convert(fact.normalized_value, fact.unit, rule.unit) if rule.unit else None
    if converted is None:
        return ComparisonResult(comparator=rule.comparator, contract_value=fact.normalized_value, unit=rule.unit,
                                satisfied=None, notes=["incompatible units"],
                                explanation=f"The contract states this in '{fact.unit}' and the law is stated in "
                                            f"'{rule.unit}'; these cannot be safely reconciled, so no automatic "
                                            "comparison was made.")

    if rule.rule_type is RuleType.RANGE:
        holds = rule.value <= converted <= rule.value_max
        explanation = f"{converted:g} {rule.unit} is {'within' if holds else 'outside'} the legal range [{rule.value:g}, {rule.value_max:g}] {rule.unit}."
    elif rule.rule_type is RuleType.MAXIMUM:
        holds = converted <= rule.value
        explanation = f"{converted:g} {rule.unit} {'does not exceed' if holds else 'exceeds'} the legal maximum of {rule.value:g} {rule.unit}."
    elif rule.rule_type is RuleType.MINIMUM:
        holds = converted >= rule.value
        explanation = f"{converted:g} {rule.unit} {'meets or exceeds' if holds else 'falls short of'} the legal minimum of {rule.value:g} {rule.unit}."
    else:  # EXACT
        holds = converted == rule.value
        explanation = f"{converted:g} {rule.unit} {'matches' if holds else 'does not match'} the required value of {rule.value:g} {rule.unit}."

    return ComparisonResult(comparator=rule.comparator, contract_value=converted, legal_value=rule.value,
                            legal_value_max=rule.value_max, unit=rule.unit, satisfied=holds, explanation=explanation)


def _status_from_comparison(fact: ContractFact | None, rule: LegalRule, comparison: ComparisonResult) -> FindingStatus:
    if fact is None:
        # The clause itself exists (Stage 2 kept it), but none of Stage 1's typed fields could be
        # tied to its own words - the relevant CONTRACT FACT is what is absent, not the clause.
        return FindingStatus.NOT_FOUND
    if fact.normalized_value is None:
        # A fact was bound (e.g. a date, a plain string) but it is not a number this engine can put
        # on either side of a comparator; that is a job for a later, qualitative rule stage.
        return FindingStatus.REQUIRES_REVIEW
    if rule.rule_type is RuleType.UNKNOWN:
        return FindingStatus.REQUIRES_REVIEW
    if rule.rule_type in (RuleType.PROHIBITION, RuleType.REQUIREMENT, RuleType.INFORMATIONAL):
        return FindingStatus.NOT_APPLICABLE
    if rule.rule_type is RuleType.CONDITIONAL:
        applicable = [c for c in comparison.condition_results if c.status is not ConditionStatus.NOT_APPLICABLE]
        if not applicable:
            return FindingStatus.REQUIRES_REVIEW
        if any(c.status in (ConditionStatus.NOT_STATED, ConditionStatus.AMBIGUOUS) for c in applicable):
            return FindingStatus.REQUIRES_REVIEW
        if all(c.status is ConditionStatus.SATISFIED for c in applicable):
            return FindingStatus.COMPLIANT
        if all(c.status is ConditionStatus.NOT_SATISFIED for c in applicable):
            return FindingStatus.NON_COMPLIANT
        return FindingStatus.REQUIRES_REVIEW  # mixed: satisfies one sub-threshold, not another
    if comparison.satisfied is None:
        return FindingStatus.REQUIRES_REVIEW
    return FindingStatus.COMPLIANT if comparison.satisfied else FindingStatus.NON_COMPLIANT


_ASSESSMENT_LABELS = {
    FindingStatus.COMPLIANT: "Appears consistent with the cited regulation (deterministic rule comparison)",
    FindingStatus.NON_COMPLIANT: "Appears inconsistent with the cited regulation (deterministic rule comparison)",
    FindingStatus.AMBIGUOUS: "The contract's own wording is unclear or self-conflicting for this clause",
    FindingStatus.REQUIRES_REVIEW: "Requires human review",
    FindingStatus.INSUFFICIENT_EVIDENCE: "Insufficient regulatory evidence retrieved",
    FindingStatus.NOT_APPLICABLE: "No deterministic numeric comparison applies to this clause",
    FindingStatus.NOT_FOUND: "No Stage 1 extracted value could be tied to this clause's own words",
    FindingStatus.ERROR: "The regulatory check could not be performed",
}


def _explain(status: FindingStatus, fact: ContractFact | None, rule: LegalRule | None,
            comparison: ComparisonResult | None, clause: ClauseValue) -> str:
    citation = (rule.article_citation if rule and rule.article_citation else
               (f"article {rule.article_number}" if rule and rule.article_number else "the cited evidence"))
    if status in (FindingStatus.COMPLIANT, FindingStatus.NON_COMPLIANT):
        return f"The contract states {fact.raw_value!r} for this clause. {citation} was read as a {rule.rule_type.value} " \
               f"rule; {comparison.explanation}"
    if status is FindingStatus.NOT_FOUND:
        return ("This clause's own words did not match any Stage 1 extracted field, so no contract fact could be "
                "tied to it. This does not mean the clause is non-compliant; it means no structured value was "
                "available to compare against the retrieved evidence.")
    if rule is not None and rule.rule_type is RuleType.CONDITIONAL:
        return f"{citation} sets more than one applicable threshold or a condition this clause alone cannot settle " \
               f"({comparison.explanation if comparison else 'no comparison was recorded'}). A human reviewer should " \
               "confirm which threshold or condition applies."
    if rule is not None and rule.rule_type is RuleType.UNKNOWN:
        return f"{citation} could not be reduced to a deterministic rule from its retrieved text. A human reviewer " \
               "should compare the contract clause with the article directly."
    if rule is not None and rule.rule_type in (RuleType.PROHIBITION, RuleType.REQUIREMENT, RuleType.INFORMATIONAL):
        return f"{citation} states a qualitative rule, not a numeric threshold, so no automated amount comparison " \
               "applies to this clause."
    if fact is None:
        return "No Stage 1 extracted value could be tied to this clause's own words, so no numeric comparison was made."
    return "This clause's evidence could not be checked deterministically; see the recorded comparison for details."


def build_clause_legal_finding(
    position: int, clause: ClauseValue, check: ClauseCheck, contract: ContractExtraction,
    topics: tuple[RegulatoryTopic, ...] = CONTRACT_TOPICS,
    explain: Callable[["ClauseLegalFinding"], str] | None = None,
) -> ClauseLegalFinding:
    """One clause, compared deterministically against a structured rule derived from its evidence.

    `explain` is called last, after `status` is already fixed by `_status_from_comparison`, and may
    only replace the wording of `explanation` - it is never given a way to change `status`.
    """
    finding_id = f"L{position:02d}"
    base = dict(finding_id=finding_id, clause_id=clause.clause_id or "C00", clause_type=clause.clause_type,
               clause_name=clause.clause_name, clause_text=clause.text, source_span=clause.source_span,
               confidence=clause.confidence, topic=check.regulatory_topic)

    if check.regulatory_topic is None:
        reason = "; ".join(check.check.notes) if check.check and check.check.notes else "no regulatory topic covers this clause"
        finding = ClauseLegalFinding(**base, status=FindingStatus.NOT_APPLICABLE,
                                     assessment=_ASSESSMENT_LABELS[FindingStatus.NOT_APPLICABLE], explanation=reason)
        return _apply_explainer(finding, explain)

    if check.check is not None and check.check.status == "error":
        finding = ClauseLegalFinding(**base, status=FindingStatus.ERROR, assessment=_ASSESSMENT_LABELS[FindingStatus.ERROR],
                                     explanation="The existing regulatory RAG returned an error for this clause; no "
                                                 "comparison was attempted.")
        return _apply_explainer(finding, explain)

    if check.check is None or check.check.status != "evidence_found" or not check.evidence:
        finding = ClauseLegalFinding(**base, status=FindingStatus.INSUFFICIENT_EVIDENCE,
                                     assessment=_ASSESSMENT_LABELS[FindingStatus.INSUFFICIENT_EVIDENCE],
                                     explanation=f"The existing RAG did not return an article relevant to this clause's "
                                                 f"topic ('{check.regulatory_topic}'). No compliance conclusion was made.")
        return _apply_explainer(finding, explain)

    distinct = _distinct_numeric_readings(clause, contract)
    if len(distinct) > 1:
        # The contract's OWN words for this one clause support more than one reading - unlike two
        # separate clauses stating 180 and 90 (each unambiguous on its own, see section 16), here a
        # single clause is internally inconsistent, which is what AMBIGUOUS is for (section 11).
        options = ", ".join(f"{value:g} {unit}" for value, unit in sorted(distinct))
        finding = ClauseLegalFinding(
            **base, status=FindingStatus.AMBIGUOUS, assessment=_ASSESSMENT_LABELS[FindingStatus.AMBIGUOUS],
            explanation=f"This clause's own words support more than one reading ({options}); the contract itself "
                        "is unclear here, so no comparison against the retrieved evidence was made.")
        return _apply_explainer(finding, explain)

    fact = derive_contract_fact(clause, contract)
    rule = derive_legal_rule(f"R{finding_id[1:]}", check.regulatory_topic, check.evidence[0])
    comparison = compare_fact_to_rule(fact, rule)
    status = _status_from_comparison(fact, rule, comparison)
    finding = ClauseLegalFinding(
        **base, status=status, assessment=_ASSESSMENT_LABELS[status],
        explanation=_explain(status, fact, rule, comparison, clause),
        contract_fact=fact, legal_rule=rule, comparison=comparison,
        regulatory_evidence=check.evidence, field=fact.field if fact else None,
    )
    return _apply_explainer(finding, explain)


def _apply_explainer(finding: ClauseLegalFinding, explain: Callable[[ClauseLegalFinding], str] | None) -> ClauseLegalFinding:
    """Reword `explanation` only. `status`/`comparison`/`legal_rule` are already final and untouched."""
    if explain is None:
        return finding
    try:
        text = explain(finding)
    except Exception:
        return finding
    if not isinstance(text, str) or not text.strip():
        return finding
    return finding.model_copy(update={"explanation": text})
