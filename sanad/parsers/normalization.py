"""Conservative normalisation of text extracted from NEWLY UPLOADED documents.

What is changed (extraction artifacts only):
  * line endings -> "\\n"; NUL and other control characters removed (tabs and newlines kept)
  * zero-width spaces, BOM, soft hyphens and bidi control marks removed
  * non-breaking and other Unicode spaces -> regular space; runs of spaces collapsed
  * trailing spaces removed; more than one blank line collapsed to one
  * PDF only: Arabic *presentation forms* (glyph codes some PDFs emit instead of letters) mapped to
    the standard Arabic letters they represent

What is deliberately NOT changed: letters, diacritics (tashkeel), tatweel, hamza/alef forms, taa
marbuta, digits (Arabic-Indic digits stay as written), punctuation, article/section numbers.
The existing RAG's documents are never touched by this module.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_REMOVE = {
    "​": "zero_width_space",
    "﻿": "byte_order_mark",
    "­": "soft_hyphen",
    "‎": "bidi_mark",
    "‏": "bidi_mark",
    "؜": "bidi_mark",
    **{chr(cp): "bidi_mark" for cp in range(0x202A, 0x202F)},
    **{chr(cp): "bidi_mark" for cp in range(0x2066, 0x206A)},
}
_SPACES = {" ", " ", "　", *(chr(cp) for cp in range(0x2000, 0x200B))}
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE = re.compile(r" {2,}")
_TAB_RUN = re.compile(r"[ \t]*\t[ \t]*")
_BLANK_LINES = re.compile(r"\n{3,}")
_ARABIC_LETTER = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-ﻼ]")
_LATIN_LETTER = re.compile(r"[A-Za-zÀ-ɏ]")


def _is_arabic_presentation_form(char: str) -> bool:
    code = ord(char)
    return 0xFB50 <= code <= 0xFDFF or 0xFE70 <= code <= 0xFEFC


@dataclass
class NormalizationReport:
    removed: dict[str, int] = field(default_factory=dict)
    presentation_forms_mapped: int = 0

    def merge(self, other: "NormalizationReport") -> None:
        for key, count in other.removed.items():
            self.removed[key] = self.removed.get(key, 0) + count
        self.presentation_forms_mapped += other.presentation_forms_mapped

    def warnings(self) -> list[str]:
        messages = []
        if self.presentation_forms_mapped:
            messages.append(
                f"{self.presentation_forms_mapped} Arabic presentation-form characters from the PDF were mapped "
                "to standard Arabic letters."
            )
        return messages


def normalize_text(text: str, *, map_arabic_presentation_forms: bool = False,
                   report: NormalizationReport | None = None) -> str:
    report = report if report is not None else NormalizationReport()
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    for char in text:
        reason = _REMOVE.get(char)
        if reason is not None:
            report.removed[reason] = report.removed.get(reason, 0) + 1
            continue
        if char in _SPACES:
            out.append(" ")
            continue
        if map_arabic_presentation_forms and _is_arabic_presentation_form(char):
            mapped = unicodedata.normalize("NFKC", char)
            if mapped != char:
                report.presentation_forms_mapped += 1
                out.append(mapped)
                continue
        out.append(char)
    text = _CONTROL.sub("", "".join(out))
    text = _TAB_RUN.sub("\t", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = "\n".join(line.strip(" ") for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def count_text_chars(text: str) -> int:
    """Letters and digits (any script); used to decide whether a page has a usable text layer."""
    return sum(1 for char in text if char.isalnum())


def detect_language(text: str) -> str:
    arabic = len(_ARABIC_LETTER.findall(text))
    latin = len(_LATIN_LETTER.findall(text))
    total = arabic + latin
    if total == 0:
        return "unknown"
    if arabic / total >= 0.8:
        return "ar"
    if latin / total >= 0.8:
        return "en"
    return "mixed"
