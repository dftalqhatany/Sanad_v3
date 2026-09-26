"""Text units with provenance, and a *matching key* used only to compare labels and keywords.

The matching key (case-folded, diacritics/tatweel removed, alef/hamza/yaa/taa-marbuta unified,
Arabic-Indic digits as ASCII) is never stored as an extracted value: values and source texts are
always sliced from the original parsed text.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from models.documents import DocumentSection, DocumentTable, ParsedDocument, SectionType, row_text
from models.extraction import ExtractionMethodName, SourceSpan

_DIACRITICS = {chr(cp) for cp in [*range(0x0610, 0x061B), *range(0x064B, 0x0660), 0x0670, *range(0x06D6, 0x06EE)]}
_TATWEEL = "ـ"
_LETTER_MAP = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"}
_DIGITS = {**{chr(0x0660 + i): str(i) for i in range(10)}, **{chr(0x06F0 + i): str(i) for i in range(10)},
           "٫": ".", "٬": ","}
_KEEP_PUNCT = set("%()/:.,-–+")
BULLET_CHARS = "•●▪◦‣-–*·"


def to_ascii_digits(text: str) -> str:
    """Same length as the input (1:1 character mapping)."""
    return "".join(_DIGITS.get(char, char) for char in text)


@dataclass
class MatchText:
    """Normalised text plus, for every normalised character, the index of its original character."""

    original: str
    norm: str
    index: list[int]

    def span(self, start: int, end: int) -> str:
        if start >= end:
            return ""
        stop = self.index[end - 1] + 1
        while stop < len(self.original) and (self.original[stop] in _DIACRITICS or self.original[stop] == _TATWEEL):
            stop += 1  # keep trailing tashkeel/tatweel, e.g. the tanween in "يوماً"
        return self.original[self.index[start]: stop]


def match_text(original: str, *, keep_punctuation: bool = True) -> MatchText:
    chars: list[str] = []
    index: list[int] = []
    for position, char in enumerate(original):
        if char in _DIACRITICS or char == _TATWEEL:
            continue
        char = _DIGITS.get(char, char)
        char = _LETTER_MAP.get(char, char)
        for folded in char.casefold():
            if folded.isalnum():
                chars.append(folded)
            elif folded.isspace() or not (keep_punctuation and folded in _KEEP_PUNCT):
                chars.append(" ")
            else:
                chars.append(folded)
            index.append(position)
    return MatchText(original, "".join(chars), index)


def matching_key(text: str) -> str:
    return " ".join(match_text(text, keep_punctuation=False).norm.split())


# A label copied out of a numbered contract ("9.1.1.1 Basic Wage", "6.1 Probationary Period",
# "9.1.1.2 Housing Allowance )" - the stray trailing parenthesis is bidi/RTL text-layer corruption,
# not part of the label) carries its clause number as a prefix; that number is never part of the
# alias, so it is stripped before a label is compared against known aliases. The numeric form
# ("9.1.1.2") must be stripped *before* `matching_key` turns its dots into spaces - "9.1.1.2 " is one
# regex match against the literal dots, but "9 1 1 2 " (dots already gone) is four indistinguishable
# leading numbers, and only the first would ever be stripped. An ordinal *word* ("Second - ...",
# "اولا") carries no such dot, so it is still stripped after normalisation, exactly as before.
_ORDINAL_PREFIX_NUMERIC = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+")
_ORDINAL_PREFIX_WORD = re.compile(
    r"^(?:[a-z]|اولا|ثانيا|ثالثا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا)\s+"
)
_PARENTHETICAL = re.compile(r"\(([^()]*)\)")


def label_keys(label: str) -> set[str]:
    """Keys for a label as written: full form, without parentheses, and each parenthetical on its own."""
    variants = {label, _PARENTHETICAL.sub(" ", label), *_PARENTHETICAL.findall(label)}
    keys = set()
    for variant in variants:
        variant = _ORDINAL_PREFIX_NUMERIC.sub("", variant)
        key = matching_key(variant)
        key = _ORDINAL_PREFIX_WORD.sub("", key)
        if key:
            keys.add(key)
    return keys


def contains_keyword(key_text: str, keyword_key: str) -> bool:
    if not keyword_key:
        return False
    if re.search(r"[؀-ۿ]", keyword_key):  # Arabic: allow attached clitics (ب، ل، و، ال)
        return keyword_key in key_text
    return re.search(rf"(?<![a-z0-9]){re.escape(keyword_key)}", key_text) is not None


def flat(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------- bilingual script runs
# Some PDF exporters lay a bilingual label/value row out as three or four visual columns (an English
# label, an Arabic value, an English value, an Arabic label) that a text-layer extractor reassembles
# into one or two lines with NO separating whitespace at the point where the script changes, e.g.
# "Genderذكر" or "Turki Salem Al-Harbiاسم العامل". There is no punctuation to split on there - the
# script change itself is the only generic signal available - so `script_runs` cuts text at every
# Arabic-letter <-> Latin-letter boundary. Digits, spaces and punctuation never start a boundary on
# their own; they stay attached to whichever run they are already inside, so this never splits a
# number, a date, or an ordinary word, in either language.
_ARABIC_CHAR = re.compile(r"[؀-ۿ]")
_LATIN_CHAR = re.compile(r"[A-Za-z]")
_HAS_LETTER = re.compile(r"[A-Za-z؀-ۿ]")


def script_runs(text: str) -> list[str]:
    runs: list[str] = []
    current: list[str] = []
    current_script: str | None = None
    for char in text:
        script = "ar" if _ARABIC_CHAR.match(char) else ("la" if _LATIN_CHAR.match(char) else None)
        if script and current_script and script != current_script and current:
            runs.append("".join(current))
            current = []
        current.append(char)
        if script:
            current_script = script
    if current:
        runs.append("".join(current))
    return [run.strip() for run in runs if run.strip()]


def dominant_script(text: str) -> str | None:
    """'ar' or 'la', by letter count; None when the text has no Arabic or Latin letters at all."""
    arabic = len(_ARABIC_CHAR.findall(text))
    latin = len(_LATIN_CHAR.findall(text))
    if not arabic and not latin:
        return None
    return "ar" if arabic >= latin else "la"


# --------------------------------------------------------------------------- units
@dataclass
class TextUnit:
    text: str  # verbatim, substring of ParsedDocument.full_text
    page_number: int | None
    section_id: str | None = None
    table_id: str | None = None
    heading: str | None = None
    section_type: SectionType | None = None
    heading_level: int | None = None
    cells: list[str] = field(default_factory=list)

    @property
    def is_heading(self) -> bool:
        return self.section_type is SectionType.HEADING

    @property
    def is_list_item(self) -> bool:
        return self.section_type is SectionType.LIST_ITEM or self.text.lstrip()[:1] in BULLET_CHARS

    def source(self, method: ExtractionMethodName, text: str | None = None) -> SourceSpan:
        return SourceSpan(text=(text if text is not None else self.text)[:1000], page_number=self.page_number,
                          section_id=self.section_id, table_id=self.table_id, method=method)


def iter_lines(document: ParsedDocument) -> Iterator[TextUnit]:
    for block in document.blocks():
        if isinstance(block, DocumentTable):
            for row in block.rows:
                yield TextUnit(text=row_text(row), page_number=block.page_number, table_id=block.table_id,
                               heading=block.title, cells=list(row))
        else:
            yield from _section_lines(block)


def _section_lines(section: DocumentSection) -> Iterator[TextUnit]:
    for line in section.text.split("\n"):
        if line.strip():
            yield TextUnit(text=line, page_number=section.page_number, section_id=section.section_id,
                           heading=None if section.section_type is SectionType.HEADING else section.title,
                           section_type=section.section_type, heading_level=section.heading_level)


_SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+")


def iter_sentences(document: ParsedDocument) -> Iterator[TextUnit]:
    """Sentences inside paragraphs/list items; a line break only continues a sentence in lower-case English."""
    for section in document.sections:
        if section.section_type is SectionType.HEADING:
            continue
        lines = section.text.split("\n")
        merged: list[str] = []
        for line in lines:
            if merged and merged[-1] and merged[-1][-1] not in ".!?؟:" and line[:1].islower():
                merged[-1] = merged[-1] + "\n" + line
            else:
                merged.append(line)
        for chunk in merged:
            for sentence in _SENTENCE_END.split(chunk):
                if sentence.strip():
                    yield TextUnit(text=sentence, page_number=section.page_number, section_id=section.section_id,
                                   heading=section.title, section_type=section.section_type)


_SEGMENT_SPLIT = re.compile(r"\s+[|•·]\s+")
_LABEL_SPLIT = re.compile(r"\s*[:：\t]\s*")
_SENTENCE_FINAL = ".!?؟"


@dataclass
class LabeledValue:
    label: str
    value: str
    unit: TextUnit
    method: ExtractionMethodName
    source_text: str

    @property
    def keys(self) -> set[str]:
        return label_keys(self.label)


def _line_parts(text: str) -> list[str]:
    """`text`, split on '|'-style separators and then on every bilingual script boundary.

    Each returned part is (as far as this mechanical split can tell) written in one script, so a
    colon search or an alias lookup run against one part never sees the other language's glued-on
    text as noise.
    """
    parts: list[str] = []
    for segment in _SEGMENT_SPLIT.split(text):
        parts.extend(script_runs(segment))
    return parts


def _candidate_pool(
    parts: list[str], start: int, units: list[TextUnit], position: int, is_label: Callable[[str], bool] | None,
    *, exclude_sentences: bool = False,
) -> list[str]:
    """Non-label, colon-free parts available within the one-line lookahead budget: the rest of the
    current unit's parts (from `start` on), then - only if that yields nothing - the next unit's
    parts, unless that next unit is a heading or a table row, which never silently donates a value
    to an unrelated label.

    `exclude_sentences` additionally drops a part that ends in terminal punctuation. A short value
    (a name, an ID, a date, an amount) is never itself a full grammatical sentence, so - for the
    bare-label mechanism, which infers its own "is this a label" and "is this its value" boundaries
    without a colon to anchor them - a candidate that reads as one full sentence (e.g. a clause
    heading followed by the clause's own prose, "Either party may terminate ... notice.") is a sign
    the "label" was not a label at all, not a value worth taking. An explicit "Label:" is a strong
    enough signal on its own that this extra guard is not applied there.
    """
    def _usable(part: str) -> bool:
        if _LABEL_SPLIT.search(part):
            return False
        if is_label is not None and is_label(part):
            return False
        if exclude_sentences and part.rstrip()[-1:] in _SENTENCE_FINAL:
            return False
        return True

    pool: list[str] = [part for part in parts[start:] if _usable(part)]
    if pool or position + 1 >= len(units):
        return pool
    following = units[position + 1]
    if following.is_heading or following.table_id is not None:
        return pool
    return [part for part in _line_parts(following.text) if _usable(part)]


def iter_labeled_values(
    document: ParsedDocument, *, max_label_chars: int = 60, is_label: Callable[[str], bool] | None = None,
) -> Iterator[LabeledValue]:
    """Explicit label/value pairs: same-line ("Label: value"), table rows, and label-then-value.

    `is_label` (normally checking every alias `ContractExtractor` recognises, plus a couple of generic
    structural cues such as "ends with 'allowance'") gates the one inferential step here: treating a
    short, colon-less run of text as a *label* whose value sits elsewhere, rather than as ordinary
    prose. Without it, only same-line "Label: value" pairs and table rows are read - always safe,
    since nothing there depends on knowing which words are labels. With it, two further, real layouts
    are read:
      * a label alone on its own line, value on the next line ("Employee Name" / "Ahmed") - common in
        bilingual PDF forms whose label/value pairs do not fit on one line;
      * a label and value glued together with no separator at all because the source PDF mixed left-
        to-right and right-to-left runs on one line ("Genderذكر", ".. Al-Harbiاسم العامل") - `_line_parts`
        has already cut these apart on the script boundary; this function only has to decide which cut
        piece is the label.
    Resolving a bare label prefers a value written in the same script as the label (so an Arabic
    mirror of an English name is never mistaken for it), but falls back to the first available
    candidate when no same-script one exists - a label in one script can still legitimately carry a
    value expressed in another whenever that value is itself language-neutral (an ID, a number, a
    currency amount with Arabic wording). A colon with nothing after it carries no such script
    expectation at all: exactly like the simpler mechanism it replaces, it just takes whatever comes
    next. Either way the value may be resolved within the same line or by looking at exactly the next
    one - never further.
    """
    units = list(iter_lines(document))
    skip_positions: set[int] = set()
    for position, unit in enumerate(units):
        if position in skip_positions:
            continue
        if unit.table_id is not None and len(unit.cells) >= 2:
            cells = unit.cells
            for start in range(0, len(cells) - 1, 2):
                label, value = cells[start].strip(), cells[start + 1].strip()
                if label and value and len(label) <= max_label_chars:
                    yield LabeledValue(label, value, unit, ExtractionMethodName.TABLE_ROW, unit.text)
            continue

        # A heading is never used as an *implicit* value donor for a label pending from elsewhere
        # (see `_candidate_pool`), but a bilingual form-style PDF frequently renders its own label
        # rows (bold, larger font) as HEADING sections - "Establishment Name (Employer) <value>" is
        # one in these fixtures - so a heading line is still read for a same-line label/value pair.
        parts = _line_parts(unit.text)
        index = 0
        while index < len(parts):
            part = parts[index]
            match = _LABEL_SPLIT.search(part)
            if match:
                label, value = part[: match.start()].strip(), part[match.end():].strip()
                index += 1
                if not label or len(label) > max_label_chars:
                    continue
                if value:
                    yield LabeledValue(label, value, unit, ExtractionMethodName.LABELED_LINE, part)
                    continue
                candidate = next(iter(_candidate_pool(parts, index, units, position, is_label)), None)
                if candidate is not None:
                    yield LabeledValue(label, candidate, unit, ExtractionMethodName.LABELED_NEXT_LINE, candidate)
                continue

            # A bare label may share its physical line with its value (a same-line, space- rather
            # than colon-separated pair, or a script-glued bilingual pair `_line_parts` has already
            # cut apart) - it need not be the line's only part. What it must not be is a fragment of
            # a wrapped prose sentence that happens to coincide with a known alias (a paragraph that
            # wraps "... within the\ncompany." leaves "company." alone on its own physical line);
            # a genuine field label is a noun phrase, never a sentence, so one ending in terminal
            # punctuation is never promoted, dictionary match or not.
            if is_label is not None and len(part) <= max_label_chars and part[-1:] not in _SENTENCE_FINAL:
                # A column narrow enough to wrap a label onto two physical lines ("Transportation" /
                # "Allowance") is common in the same bilingual PDF forms this module already handles.
                # There is no punctuation cue for a line break the way there is for a script change,
                # so the only generic signal available is dictionary-driven: if this line, on its
                # own, is not (or not fully) a recognised label, but appending the *entire* next
                # physical line turns it into one, that next line is the rest of the label, not a
                # value - so it is consumed as label text and the real value is looked up one line
                # further than usual, instead of the wrapped fragment ("Allowance") being taken as
                # the value itself.
                continuation = None
                # A bare clause number ("9.1.1.3") is never itself part of the label text - it is
                # numbering, stripped by `label_keys` precisely so it never has to match anything -
                # so it must never be the fragment this merge attaches a following line to (that
                # would grab "Transportation" as 9.1.1.3's own "continuation", leaving the real
                # label fragment "Allowance" one line later to be misread as a value instead).
                if _HAS_LETTER.search(part) and index == len(parts) - 1 and position + 1 < len(units):
                    following = units[position + 1]
                    if following.is_heading or following.table_id is not None:
                        following = None
                    if following is not None:
                        follow_parts = _line_parts(following.text)
                        if (len(follow_parts) == 1 and _LABEL_SPLIT.search(follow_parts[0]) is None
                                and follow_parts[0][-1:] not in _SENTENCE_FINAL):
                            continuation = follow_parts[0]
                combined = f"{part} {continuation}" if continuation is not None else None
                if combined is not None and len(combined) <= max_label_chars and is_label(combined):
                    label_script = dominant_script(combined)
                    pool = _candidate_pool([], 0, units, position + 1, is_label, exclude_sentences=True)
                    candidate = None
                    if label_script is not None:
                        candidate = next((p for p in pool if dominant_script(p) == label_script), None)
                    if candidate is None:
                        candidate = pool[0] if pool else None
                    skip_positions.add(position + 1)
                    index += 1
                    if candidate is not None:
                        yield LabeledValue(combined, candidate, unit, ExtractionMethodName.LABELED_NEXT_LINE, candidate)
                    continue

                if is_label(part):
                    label_script = dominant_script(part)
                    pool = _candidate_pool(parts, index + 1, units, position, is_label, exclude_sentences=True)
                    candidate = None
                    if label_script is not None:
                        candidate = next((p for p in pool if dominant_script(p) == label_script), None)
                    if candidate is None:
                        candidate = pool[0] if pool else None
                    index += 1
                    if candidate is not None:
                        yield LabeledValue(part, candidate, unit, ExtractionMethodName.LABELED_NEXT_LINE, candidate)
                    continue

            index += 1
