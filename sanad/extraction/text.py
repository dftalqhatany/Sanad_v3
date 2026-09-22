"""Text units with provenance, and a *matching key* used only to compare labels and keywords.

The matching key (case-folded, diacritics/tatweel removed, alef/hamza/yaa/taa-marbuta unified,
Arabic-Indic digits as ASCII) is never stored as an extracted value: values and source texts are
always sliced from the original parsed text.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
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


_ORDINAL_PREFIX = re.compile(
    r"^(?:\d+|[a-z]|اولا|ثانيا|ثالثا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا)\s+"
)
_PARENTHETICAL = re.compile(r"\(([^()]*)\)")


def label_keys(label: str) -> set[str]:
    """Keys for a label as written: full form, without parentheses, and each parenthetical on its own."""
    variants = {label, _PARENTHETICAL.sub(" ", label), *_PARENTHETICAL.findall(label)}
    keys = set()
    for variant in variants:
        key = matching_key(variant)
        key = _ORDINAL_PREFIX.sub("", key)
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


def iter_labeled_values(document: ParsedDocument, *, max_label_chars: int = 60) -> Iterator[LabeledValue]:
    units = list(iter_lines(document))
    for position, unit in enumerate(units):
        if unit.table_id is not None and len(unit.cells) >= 2:
            cells = unit.cells
            for start in range(0, len(cells) - 1, 2):
                label, value = cells[start].strip(), cells[start + 1].strip()
                if label and value and len(label) <= max_label_chars:
                    yield LabeledValue(label, value, unit, ExtractionMethodName.TABLE_ROW, unit.text)
            continue
        for segment in _SEGMENT_SPLIT.split(unit.text):
            match = _LABEL_SPLIT.search(segment)
            if not match:
                continue
            label, value = segment[: match.start()].strip(), segment[match.end():].strip()
            if not label or len(label) > max_label_chars:
                continue
            if value:
                yield LabeledValue(label, value, unit, ExtractionMethodName.LABELED_LINE, segment)
            elif position + 1 < len(units):
                following = units[position + 1]
                if not following.is_heading and not _LABEL_SPLIT.search(following.text) and following.table_id is None:
                    yield LabeledValue(label, following.text.strip(), following,
                                       ExtractionMethodName.LABELED_NEXT_LINE, following.text)
