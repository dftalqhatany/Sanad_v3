"""Clause segmentation and classification.

Turns a ParsedDocument into a list of `ClauseValue`s: the smallest spans of the document's own text
that each state one contractual rule, classified into a controlled vocabulary and carrying the
provenance needed to cite them later.

What this layer does NOT do
---------------------------
It does not read values (Stage 1's `extraction.semantic` does that), and it reaches no legal
conclusion of any kind. A clause here answers "what does the contract say, and where", never
"is that allowed".

Segmentation
------------
The unit is the section (paragraph / list item), not the page and not the sentence:

  * within a section, every sentence is classified with a subject-bound rule. Consecutive sentences
    of the same type form one clause; a sentence of a different type starts a new one. That splits
    a single numbered paragraph that states several independent rules - in the Aramco letter, the
    whole of "2. Terms and Conditions" is one paragraph containing the probation period and both
    notice periods.
  * a sentence that classifies as nothing extends the clause it follows, so a rule written over two
    sentences is not torn in half.
  * when no sentence in a section carries a rule but its heading names a topic ("Job Duties &
    Responsibilities", "Confidentiality"), the section becomes one clause of the heading's type.
    That is how list-shaped sections are kept whole.

Classification
--------------
Two sources, both subject-bound, neither of them a bare keyword match:

  * the Stage 1 narrative rules in `extraction.semantic`, mapped field -> clause type. Reusing them
    is what guarantees that the sentence which gave `probation_period` its 180 days is the same
    sentence that becomes the PROBATION clause, and that the notice sentences - which mention
    probation but state a notice period - become NOTICE clauses instead.
  * clause-only patterns for topics that have no extracted field (duties, confidentiality, ...).

A clause whose type cannot be established stays OTHER or UNKNOWN. Nothing is forced into a type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from extraction.semantic import NARRATIVE_RULES
from extraction.text import BULLET_CHARS, flat, iter_sentences
from models.documents import ParsedDocument, SectionType
from models.extraction import ClauseType, ClauseValue, ExtractionMethodName, SourceSpan

_FLAGS = re.IGNORECASE | re.UNICODE
_SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+")

# "15. You will automatically retire ..." opens a new item in the contract's own numbering. Whatever
# it says, it is a different rule from the one before it, so it must never be swallowed as a
# continuation of the clause above.
_NEW_ITEM = re.compile(r"^\s*(?:<b>\s*)?(?:\(?\d{1,2}\)?[.)]|\([a-z]\)|[a-z][.)](?=\s))\s*|^\s*[" +
                       re.escape(BULLET_CHARS) + r"]\s*")


def starts_new_item(text: str) -> bool:
    """Whether a sentence opens a new numbered, lettered or bulleted item."""
    return _NEW_ITEM.match(text) is not None

# Letterheads, seals and signature rules repeat on every page and state no contractual rule.
_NOISE = tuple(re.compile(p, _FLAGS) for p in (
    r"\[(logo|signature|handwritten signature|official seal)",
    r"\bP\.?\s*O\.?\s*Box\b",
    r"\bDhahran,?\s*31311\b",
    r"\bARAMCO\s+Tower\b|\bCyber\s+City\b|\bGurugram\b|\bHaryana\b",
    r"\bwww\.[a-z0-9.-]+|\b[a-z0-9._-]+@[a-z0-9.-]+\.[a-z]{2,}",
    r"^\s*(SAUDI HO|INDIA)\s*[-–]",
    r"^\s*(Landline|Email|Visit)\b",
    r"_{5,}",
    r"^\s*Sincerely\b",
    r"^\s*(Name|Date)\s*$",
    r"^\s*\W*$",
))

# --------------------------------------------------------------------------- field -> clause type
# Stage 1 reads a value; the same bound rule tells us what kind of clause stated it.
FIELD_CLAUSE_TYPES: dict[str, ClauseType] = {
    "job_title": ClauseType.JOB_TITLE,
    "work_location": ClauseType.WORK_LOCATION,
    "start_date": ClauseType.START_DATE,
    "probation_period": ClauseType.PROBATION,
    "notice_period_during_probation": ClauseType.NOTICE,
    "notice_period_after_confirmation": ClauseType.NOTICE,
    "working_hours": ClauseType.WORKING_HOURS,
    "working_days": ClauseType.WORKING_DAYS,
    "weekly_rest": ClauseType.WEEKLY_REST,
    "annual_leave": ClauseType.ANNUAL_LEAVE,
    "salary": ClauseType.SALARY,
    "total_salary": ClauseType.SALARY,
    "in_hand_salary": ClauseType.SALARY,
    "employee_name": ClauseType.EMPLOYMENT,
}


@dataclass(frozen=True)
class ClauseRule:
    """A clause-only pattern, for a topic no extracted field covers.

    `requires` is what keeps a topic word from becoming a regulatory classification on its own.
    "Your compensation and benefits are attached as Annexure A" names benefits but grants none, so
    the BENEFITS rule additionally requires the sentence to state an entitlement. A sentence that
    mentions a topic without ruling on it is better left unclassified than filed under a regulatory
    type it does not belong to.
    """

    clause_type: ClauseType
    pattern: re.Pattern[str]
    excludes: tuple[re.Pattern[str], ...] = ()
    requires: tuple[re.Pattern[str], ...] = ()

    def matches(self, text: str) -> bool:
        if any(bad.search(text) for bad in self.excludes):
            return False
        if not all(need.search(text) for need in self.requires):
            return False
        return self.pattern.search(text) is not None


def _clause_rule(clause_type: ClauseType, pattern: str, *, excludes: tuple[str, ...] = (),
                 requires: tuple[str, ...] = ()) -> ClauseRule:
    return ClauseRule(clause_type, re.compile(pattern, _FLAGS),
                      tuple(re.compile(e, _FLAGS) for e in excludes),
                      tuple(re.compile(r, _FLAGS) for r in requires))


# The sentence grants, owes or pays something - as opposed to merely naming it.
_GRANTS = (r"\b(?:entitled|eligible|shall\s+\w+|will\s+\w+|are\s+provided|is\s+provid\w+|provid\w+|"
           r"offers?|offering|grants?|payable|reimburs\w+|cover(?:s|ed|age)?|receive|get|bring\w*)\b")
# The sentence puts a figure on it. A naming token like "CTC" is not a figure, so
# "CTC component - Basic Pay, House Rent Allowance (HRA)" stays an enumeration of component names,
# not a statement of pay or of a housing entitlement.
_FIGURE = r"(?:\d|%)"
# A grant that governs the word after it, rather than sitting loose in the same sentence. "...the
# terms (including the compensation and benefits, allowances) shall apply" has a modal, but it
# grants nothing: the modal governs "the terms", not the allowances.
_GRANTS_BEFORE = (r"(?:entitled\s+to|eligible\s+for|will\s+(?:get|receive|be\s+provided|be\s+eligible|be\s+paid)|"
                  r"are\s+provided\s+with|is\s+provid\w+|provides?|providing|offers?|offering|grants?|"
                  r"shall\s+be|will\s+be)[^.]{0,45}?")
# The rule is addressed to the worker.
_ADDRESSES_WORKER = r"\b(?:you|your|employee|worker)\b"


# Ordered: the first match wins, so the more specific rule is listed first.
CLAUSE_RULES: tuple[ClauseRule, ...] = (
    _clause_rule(ClauseType.NOTICE,
                 r"\b(?:\d{1,4}|thirty|sixty|ninety)\s*(?:calendar\s+|working\s+)?days?\b[’'`´]?\s*(?:prior\s+|written\s+)?notice"
                 r"|notice\s+(?:period\s+)?of\s+\b(?:\d{1,4}|thirty|sixty|ninety)\s*days?"),
    _clause_rule(ClauseType.TERMINATION,
                 r"\bterminat(?:e|ed|ion)\b|\bresign\w*\b|\bend\s+of\s+(?:the\s+)?(?:contract|employment)\b"
                 r"|\bdismiss\w*\b|إنهاء|فسخ"),
    _clause_rule(ClauseType.SICK_LEAVE, r"\bsick\s+leave\b|\bmedical\s+leave\b|إجازة\s+مرضية"),
    _clause_rule(ClauseType.OVERTIME, r"\bover\s?time\b|ساعات\s+إضافية|العمل\s+الإضافي"),
    _clause_rule(ClauseType.HEALTH_INSURANCE,
                 r"\bhealth\s+insurance\b|\bmedical\s+insurance\b|\bhospitali[sz]ation\b|تأمين\s+طبي",
                 requires=(_GRANTS,)),
    _clause_rule(ClauseType.HOUSING,
                 r"\bhousing\s+allowance\b|\bhouse\s+rent\s+allowance\b|\baccommodation\b|بدل\s+سكن",
                 requires=(rf"(?:{_GRANTS}|{_FIGURE})",)),
    _clause_rule(ClauseType.TRANSPORTATION,
                 r"\btransport(?:ation)?\s+allowance\b|\bconveyance\s+allowance\b|\bfree\s+transportation\b|بدل\s+نقل",
                 requires=(rf"(?:{_GRANTS}|{_FIGURE})",)),
    _clause_rule(ClauseType.NON_COMPETE,
                 r"\bnon[-\s]?compete\b|\bnot\s+(?:be\s+)?(?:allowed\s+to\s+be\s+)?employed\s+by\s+any\s+other\s+company\b"
                 r"|\bcompeting\s+business\b|عدم\s+المنافسة"),
    _clause_rule(ClauseType.CONFIDENTIALITY,
                 r"\bconfidential(?:ity)?\b|\bshall\s+not\s+disclose\b|\bnon[-\s]?disclosure\b|السرية|سري"),
    _clause_rule(ClauseType.DISCIPLINARY,
                 r"\bmisconduct\b|\bdisciplinary\b|\bshow\s+cause\s+notice\b|\bperformance\s+improvement\s+plan\b"),
    _clause_rule(ClauseType.BENEFITS,
                 r"(?:entitled\s+to|eligible\s+for|will\s+(?:get|receive|be\s+provided|be\s+eligible)|"
                 r"are\s+provided\s+with|is\s+provid\w+|provides?|providing|offers?|offering|grants?)"
                 r"[^.]{0,45}?\b(?:benefits?|perks?|bonus(?:es)?|reimbursement)\b|مزايا",
                 requires=(_ADDRESSES_WORKER,)),
    _clause_rule(ClauseType.ALLOWANCES,
                 rf"{_GRANTS_BEFORE}\ballowances?\b|\ballowances?\b[^.]{{0,30}}?{_FIGURE}|بدلات|البدلات"),
    _clause_rule(ClauseType.SALARY,
                 r"\b(?:gross|basic|monthly|annual)\s+(?:salary|pay|wage)\b|\bcompensation\b|\bremuneration\b"
                 r"|\bearning\s+potential\b|\bCTC\b|\bin\s*hand\b|الأجر|الراتب",
                 requires=(rf"(?:{_FIGURE}|\bpaid\b|\bpayable\b|\bpay(?:s|ing)?\s+your?\b)",)),
    _clause_rule(ClauseType.DUTIES,
                 r"\bduties\s+and\s+responsibilities\b|\bjob\s+duties\b|\bresponsible\s+for\b"
                 r"|\bcarry\s+out\s+all\s+duties\b|المهام|الواجبات",
                 requires=(_ADDRESSES_WORKER,)),
    _clause_rule(ClauseType.CONTRACT_DURATION,
                 r"\b(?:fixed[-\s]term|indefinite)\b|\bterm\s+of\s+(?:this\s+)?contract\b|\bcontract\s+duration\b|مدة\s+العقد"),
    _clause_rule(ClauseType.WORK_LOCATION,
                 r"\bplace\s+of\s+work\b|\bwork\s+location\b|\btransferable\s+to\b|\bposted\s+(?:at|in)\b|مكان\s+العمل"),
    _clause_rule(ClauseType.EMPLOYMENT,
                 r"\bappoint(?:ed|ment)\b|\boffer\s+of\s+employment\b|\bselection\s+of\s+employment\b|\bon\s?boarding\b",
                 requires=(_ADDRESSES_WORKER,)),
)

# Headings are titles, not rules, so a keyword there is enough to name the block beneath them.
HEADING_RULES: tuple[tuple[ClauseType, re.Pattern[str]], ...] = tuple(
    (clause_type, re.compile(pattern, _FLAGS)) for clause_type, pattern in (
        (ClauseType.DUTIES, r"\bduties\b|\bresponsibilities\b|المهام"),
        (ClauseType.CONFIDENTIALITY, r"\bconfidential"),
        (ClauseType.ANNUAL_LEAVE, r"\bleave\s+policy\b|\bannual\s+leave\b|\bvacation\b"),
        (ClauseType.HEALTH_INSURANCE, r"\bhealth\s+insurance\b"),
        (ClauseType.BENEFITS, r"\bbenefits?\b|\bperks?\b|\bbonus\b"),
        (ClauseType.SALARY, r"\bcompensation\b|\bsalary\b|\bin\s*hand\b|\bremuneration\b"),
        (ClauseType.ALLOWANCES, r"\ballowances?\b"),
        (ClauseType.TERMINATION, r"\bterminat"),
        (ClauseType.NOTICE, r"\bnotice\b"),
        (ClauseType.PROBATION, r"\bprobation"),
        (ClauseType.WORKING_HOURS, r"\bworking\s+hours\b|\bhours\s+of\s+work\b"),
        (ClauseType.DISCIPLINARY, r"\bdisciplinary\b|\bconduct\b"),
        (ClauseType.EMPLOYMENT, r"\bappointment\b|\bterms\s+and\s+conditions\b|\bemployment\b"),
    )
)

# A NOTICE clause is much more useful to Stage 3 when it says which phase it governs.
_NAME_HINTS: dict[ClauseType, tuple[tuple[str, str], ...]] = {
    ClauseType.NOTICE: (
        (r"during\s+(?:the\s+)?(?:period\s+of\s+)?probation|probation(?:ary)?\s+period", "Notice during probation"),
        (r"after\s+(?:your\s+)?confirmation|once\s+confirmed", "Notice after confirmation"),
    ),
}

_TYPE_LABELS = {
    ClauseType.EMPLOYMENT: "Employment", ClauseType.JOB_TITLE: "Job title",
    ClauseType.WORK_LOCATION: "Work location", ClauseType.START_DATE: "Start date",
    ClauseType.CONTRACT_DURATION: "Contract duration", ClauseType.PROBATION: "Probation",
    ClauseType.NOTICE: "Notice", ClauseType.TERMINATION: "Termination",
    ClauseType.WORKING_HOURS: "Working hours", ClauseType.WEEKLY_REST: "Weekly rest",
    ClauseType.WORKING_DAYS: "Working days", ClauseType.ANNUAL_LEAVE: "Annual leave",
    ClauseType.SICK_LEAVE: "Sick leave", ClauseType.SALARY: "Salary",
    ClauseType.ALLOWANCES: "Allowances", ClauseType.BENEFITS: "Benefits",
    ClauseType.OVERTIME: "Overtime", ClauseType.TRANSPORTATION: "Transportation",
    ClauseType.HOUSING: "Housing", ClauseType.HEALTH_INSURANCE: "Health insurance",
    ClauseType.DUTIES: "Duties", ClauseType.CONFIDENTIALITY: "Confidentiality",
    ClauseType.NON_COMPETE: "Non-compete", ClauseType.DISCIPLINARY: "Disciplinary",
    ClauseType.OTHER: "Other", ClauseType.UNKNOWN: "Unclassified",
}


# --------------------------------------------------------------------------- classification
def classify_sentence(text: str) -> tuple[ClauseType | None, str]:
    """The clause type a single sentence states, and how it was decided.

    The Stage 1 narrative rules are tried first: they are subject-bound, so "During the period of
    probation ... 90 days' notice" matches the notice rule and never the probation one.
    """
    for rule in NARRATIVE_RULES:
        clause_type = FIELD_CLAUSE_TYPES.get(rule.field)
        if clause_type is not None and rule.search(text) is not None:
            return clause_type, f"narrative_rule:{rule.field}"
    for rule in CLAUSE_RULES:
        if rule.matches(text):
            return rule.clause_type, f"clause_rule:{rule.clause_type.value}"
    return None, "unmatched"


def classify_heading(heading: str | None) -> ClauseType | None:
    if not heading:
        return None
    for clause_type, pattern in HEADING_RULES:
        if pattern.search(heading):
            return clause_type
    return None


def clause_name_for(clause_type: ClauseType, text: str, heading: str | None) -> str:
    for pattern, name in _NAME_HINTS.get(clause_type, ()):
        if re.search(pattern, text, _FLAGS):
            return name
    label = _TYPE_LABELS.get(clause_type, clause_type.value.replace("_", " ").capitalize())
    if heading:
        heading_type = classify_heading(heading)
        # Only qualify the name with the heading when the heading is about the same thing. The
        # nearest heading can be stale - the Aramco leave policy sits under "4. Confidentiality" -
        # and "Annual leave (Confidentiality)" would misdescribe the clause.
        if heading_type is not None and heading_type is not clause_type:
            return label
        cleaned = re.sub(r"^\s*\d+[.)]\s*", "", " ".join(heading.split())).strip()
        if cleaned and len(cleaned) <= 60 and cleaned.casefold() != label.casefold():
            return f"{label} ({cleaned})"
    return label


# --------------------------------------------------------------------------- segmentation
@dataclass
class _Group:
    clause_type: ClauseType
    sentences: list[str]
    method: str


def _verbatim_span(section_text: str, sentences: list[str]) -> str:
    """The document's own text from the first sentence to the last, separators included.

    Joining the sentences with a space would drop whatever sits between them in the original - a
    bullet, a numbering, a line break - and the result would no longer be a substring of the
    document, so it could not be shown to the reader as a quotation.
    """
    flat_section = flat(section_text)
    first, last = flat(sentences[0]), flat(sentences[-1])
    start = flat_section.find(first)
    if start < 0:
        return " ".join(sentences)
    end = flat_section.find(last, start)
    end = (end + len(last)) if end >= 0 else (start + len(first))
    return flat_section[start:end]


def _is_noise(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 3:
        return True
    return any(pattern.search(stripped) for pattern in _NOISE)


def _sentences_by_section(document: ParsedDocument) -> dict[str, list[str]]:
    """Sentences per section, via the shared iterator.

    `iter_sentences` rejoins a line that a PDF wrapped mid-sentence ("...90 days\u2019 / notice..."),
    which a naive line split would tear in two - and a torn sentence loses the very binding that
    tells a notice period apart from a probation period.
    """
    by_section: dict[str, list[str]] = {}
    for unit in iter_sentences(document):
        if unit.section_id is None:
            continue
        text = " ".join(unit.text.split())
        if _is_noise(text):
            continue
        by_section.setdefault(unit.section_id, []).append(text)
    return by_section


class ClauseSegmenter:
    """Sections -> classified clauses, each quoting the document verbatim."""

    def __init__(self, *, min_clause_chars: int = 25) -> None:
        self.min_clause_chars = min_clause_chars

    def segment(self, document: ParsedDocument) -> list[ClauseValue]:
        by_section = _sentences_by_section(document)
        first_under_heading = self._first_content_sections(document, by_section)
        clauses: list[ClauseValue] = []
        for section in document.sections:
            if section.section_type is SectionType.HEADING:
                clause = self._heading_clause(section)
                if clause is not None:
                    clauses.append(clause)
                continue
            clauses.extend(self._section_clauses(
                section, by_section.get(section.section_id, []),
                heading_fallback=section.section_id in first_under_heading))
        clauses.extend(self._bullet_list_clauses(document))
        clauses.extend(self._table_row_clauses(document))
        clauses.sort(key=lambda c: (c.source_span.page_number or 0, c.source_span.section_id or ""))
        for position, clause in enumerate(clauses, 1):
            clause.clause_id = f"C{position:02d}"
        return clauses

    # ------------------------------------------------------------------ bookkeeping
    @staticmethod
    def _first_content_sections(document: ParsedDocument, by_section: dict[str, list[str]]) -> set[str]:
        """The first substantive section under each heading.

        A heading only speaks for the block that immediately follows it. Without this, every later
        section in a long run inherits the last heading seen and a signature line ends up filed
        under "Confidentiality".
        """
        first: set[str] = set()
        seen: set[str | None] = set()
        for section in document.sections:
            if section.section_type is SectionType.HEADING or not by_section.get(section.section_id):
                continue
            key = section.heading_id or section.title
            if key not in seen:
                seen.add(key)
                first.add(section.section_id)
        return first

    # ------------------------------------------------------------------ a heading that states a fact
    @staticmethod
    def _heading_clause(section) -> ClauseValue | None:
        """Most headings are titles, but some carry the rule themselves ("In hand monthly- 1800 Riyal").

        Only a subject-bound narrative rule qualifies a heading as a clause; a heading that merely
        contains a topic word stays a title.
        """
        text = " ".join(section.text.split())
        if _is_noise(text) or text[:1] in BULLET_CHARS:
            return None
        for rule in NARRATIVE_RULES:
            clause_type = FIELD_CLAUSE_TYPES.get(rule.field)
            if clause_type is not None and rule.search(text) is not None:
                return ClauseValue(
                    clause_name=clause_name_for(clause_type, text, None),
                    clause_type=clause_type, heading=None, text=text, confidence="high",
                    source_span=SourceSpan(text=text[:1000], page_number=section.page_number,
                                           section_id=section.section_id,
                                           method=ExtractionMethodName.SECTION_CONTENT))
        return None

    # ------------------------------------------------------------------ one section
    def _section_clauses(self, section, sentences: list[str], *, heading_fallback: bool) -> list[ClauseValue]:
        if not sentences:
            return []

        groups: list[_Group] = []
        pending: list[str] = []
        for sentence in sentences:
            clause_type, method = classify_sentence(sentence)
            if clause_type is None:
                # An unclassified sentence continues the rule above it only when it is part of the
                # same item. One that opens a new item is a separate rule Sanad has no category for:
                # it is kept as OTHER rather than being folded into an unrelated clause.
                if starts_new_item(sentence):
                    groups.append(_Group(ClauseType.OTHER, [*pending, sentence], "unclassified_item"))
                    pending = []
                elif groups:
                    groups[-1].sentences.append(sentence)
                else:
                    pending.append(sentence)
                continue
            # A sentence that states a rule always begins its own clause, even when the sentence
            # before it stated a rule of the same kind: "during probation ... 90 days' notice" and
            # "after confirmation ... 90 days' notice" are two NOTICE clauses, not one.
            groups.append(_Group(clause_type, [*pending, sentence], method))
            pending = []

        if not groups:
            heading_type = classify_heading(section.title) if heading_fallback else None
            if heading_type is None:
                return []
            groups = [_Group(heading_type, sentences, "heading")]

        return [self._build(group, section) for group in groups
                if len(flat(" ".join(group.sentences))) >= self.min_clause_chars]

    # ------------------------------------------------------------------ bullet runs
    @staticmethod
    def _bullet_list_clauses(document: ParsedDocument) -> list[ClauseValue]:
        """Some lists are parsed as a run of one-line headings ("• Free Transportation").

        Those lines are the content of the classified heading above them, so the run is collected
        into a single clause rather than discarded with the other headings.
        """
        clauses: list[ClauseValue] = []
        run: list = []
        current: ClauseType | None = None
        sections = list(document.sections)

        def flush() -> None:
            nonlocal run, current
            if run and current is not None:
                # Bullet markers are kept: with them the run is still a verbatim span of the document.
                text = " ".join(" ".join(s.text.split()) for s in run)
                first = run[0]
                clauses.append(ClauseValue(
                    clause_name=clause_name_for(current, text, None),
                    clause_type=current,
                    heading=None,
                    text=text,
                    confidence="medium",
                    source_span=SourceSpan(text=text[:1000], page_number=first.page_number,
                                           section_id=first.section_id,
                                           method=ExtractionMethodName.SECTION_CONTENT),
                ))
            run, current = [], current

        for section in sections:
            if section.section_type is not SectionType.HEADING:
                continue
            text = " ".join(section.text.split())
            if text[:1] in BULLET_CHARS:
                if current is not None:
                    run.append(section)
                continue
            flush()
            run = []
            current = classify_heading(text)
        flush()
        return clauses

    # ------------------------------------------------------------------ table rows (Stage 4.1)
    def _table_row_clauses(self, document: ParsedDocument) -> list[ClauseValue]:
        """One clause per non-trivial table row, classified the same way a sentence would be.

        Stage 1 already reads some facts straight out of table rows (ExtractionMethodName.TABLE_ROW,
        SourceSpan.table_id - see extraction/fields.py); until now, a fact read that way had no
        clause of its own for Stage 3 to attach a legal rule to, not because the value was missing
        but because nothing here ever looked at the table it came from. This never re-reads or
        reinterprets a cell: `row_text` is exactly the text extraction/fields.py itself would see,
        copied verbatim, and classification reuses the same `classify_sentence` a paragraph sentence
        goes through - a row that states no recognisable rule stays OTHER, exactly like an
        unclassified sentence does.
        """
        clauses: list[ClauseValue] = []
        for table in document.tables:
            for row_text in table.row_texts:
                text = " ".join(row_text.split())
                if _is_noise(text) or len(flat(text)) < self.min_clause_chars:
                    continue
                clause_type, method = classify_sentence(text)
                clause_type = clause_type or ClauseType.OTHER
                clauses.append(ClauseValue(
                    clause_name=clause_name_for(clause_type, text, table.title),
                    clause_type=clause_type,
                    heading=table.title,
                    text=text,
                    confidence="high" if method.startswith("narrative_rule") else
                               ("medium" if method.startswith("clause_rule") else "low"),
                    source_span=SourceSpan(text=text[:1000], page_number=table.page_number,
                                           table_id=table.table_id, method=ExtractionMethodName.TABLE_ROW),
                ))
        return clauses

    # ------------------------------------------------------------------ build
    @staticmethod
    def _build(group: _Group, section) -> ClauseValue:
        text = _verbatim_span(section.text, group.sentences)
        heading = section.title
        return ClauseValue(
            clause_name=clause_name_for(group.clause_type, text, heading),
            clause_type=group.clause_type,
            heading=heading,
            text=text,
            confidence="high" if group.method.startswith("narrative_rule") else
                       ("medium" if group.method.startswith("clause_rule") else "low"),
            source_span=SourceSpan(text=text[:1000], page_number=section.page_number,
                                   section_id=section.section_id,
                                   method=ExtractionMethodName.CLAUSE_SENTENCE
                                   if group.method != "heading" else ExtractionMethodName.SECTION_CONTENT),
        )


def segment_clauses(document: ParsedDocument) -> list[ClauseValue]:
    return ClauseSegmenter().segment(document)
