"""Readers for explicitly written values (numbers, money, durations, dates, hours, days).

Every reader returns what is literally written. Nothing is converted between units (90 days is
not turned into 3 months), no currency is assumed, and a day/month order that cannot be told
apart is not resolved. When a phrase contains conflicting numbers (e.g. "ninety (60) days" or
"8 or 9 hours"), the reading is returned with value None and a note, so it becomes AMBIGUOUS.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from extraction.text import MatchText, match_text, to_ascii_digits

# --------------------------------------------------------------------------- number words
_AR_UNITS = {
    "واحد": 1, "واحده": 1, "احد": 1, "احدي": 1, "اثنان": 2, "اثنين": 2, "اثنتان": 2, "اثنتين": 2, "اثنا": 2, "اثني": 2,
    "ثلاثه": 3, "ثلاث": 3, "اربعه": 4, "اربع": 4, "خمسه": 5, "خمس": 5, "سته": 6, "ست": 6, "سبعه": 7, "سبع": 7,
    "ثمانيه": 8, "ثماني": 8, "ثمان": 8, "تسعه": 9, "تسع": 9, "عشره": 10, "عشر": 10,
}
_AR_TENS = {
    "عشرون": 20, "عشرين": 20, "ثلاثون": 30, "ثلاثين": 30, "اربعون": 40, "اربعين": 40, "خمسون": 50, "خمسين": 50,
    "ستون": 60, "ستين": 60, "سبعون": 70, "سبعين": 70, "ثمانون": 80, "ثمانين": 80, "تسعون": 90, "تسعين": 90,
}
_AR_HUNDREDS = {"مائه": 100, "مئه": 100, "مائتان": 200, "مائتين": 200, "مئتان": 200, "مئتين": 200}
for _prefix, _value in (("ثلاث", 3), ("اربع", 4), ("خمس", 5), ("ست", 6), ("سبع", 7), ("ثمان", 8), ("تسع", 9)):
    _AR_HUNDREDS[_prefix + "مائه"] = _value * 100
    _AR_HUNDREDS[_prefix + "مئه"] = _value * 100
_EN_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_RANGE_WORDS = {"or", "to", "-", "–", "او", "الي", "حتي", "and"}


def _word_value(token: str) -> tuple[str, int] | None:
    for lexicon, kind in ((_AR_UNITS, "unit"), (_AR_TENS, "tens"), (_AR_HUNDREDS, "hundreds"), (_EN_NUMBERS, "en")):
        if token in lexicon:
            return kind, lexicon[token]
    if token.startswith("و") and len(token) > 2:
        return _word_value(token[1:])
    if token in ("hundred",):
        return "hundred_mult", 100
    return None


def parse_number_words(tokens: list[str]) -> int | None:
    """Value of a contiguous run of number words (Arabic or English), or None."""
    total, previous_unit = 0, None
    for token in tokens:
        if token in ("and", "و"):
            continue
        reading = _word_value(token)
        if reading is None:
            return None
        kind, value = reading
        if kind == "hundred_mult":
            total = (total or 1) * 100 if previous_unit is None else total - previous_unit + previous_unit * 100
            previous_unit = None
            continue
        if kind == "unit" and value == 10 and previous_unit is not None and previous_unit < 10:  # ثلاثة عشر
            total += 10
            previous_unit = None
            continue
        total += value
        previous_unit = value if value < 10 or kind == "en" and value < 20 else None
    return total or None


# --------------------------------------------------------------------------- tokens
@dataclass
class Token:
    text: str
    start: int
    end: int

    @property
    def number(self) -> float | None:
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", self.text):
            return float(self.text.replace(",", ""))
        return None

    @property
    def is_number_word(self) -> bool:
        return _word_value(self.text) is not None or self.text in ("hundred",)


_TOKEN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|[^\W\d_]+|[%()\-–/:+]")


def tokenize(text: str) -> tuple[MatchText, list[Token]]:
    matched = match_text(text)
    return matched, [Token(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(matched.norm)]


@dataclass
class Reading:
    value: object | None
    raw: str
    notes: list[str] = field(default_factory=list)


def _number_before(tokens: list[Token], end: int) -> tuple[float | None, int, list[str]]:
    """Read the number written immediately before tokens[end] (exclusive). Returns (value, start_index, notes)."""
    j = end - 1
    found: list[float] = []
    start = end
    if j >= 2 and tokens[j].text == ")" and tokens[j - 2].text == "(" and tokens[j - 1].number is not None:
        found.append(tokens[j - 1].number)
        start = j - 2
        j -= 3
    if j >= 0 and tokens[j].number is not None:
        found.append(tokens[j].number)
        start = j
        j -= 1
    words_end = j + 1
    while j >= 0 and (tokens[j].is_number_word or (tokens[j].text in ("and", "-") and j > 0 and tokens[j - 1].is_number_word)):
        j -= 1
    if words_end - (j + 1) > 0:
        value = parse_number_words([t.text for t in tokens[j + 1: words_end] if t.text != "-"])
        if value is not None:
            found.append(float(value))
            start = j + 1
    notes = []
    distinct = sorted(set(found))
    if len(distinct) > 1:
        notes.append(f"conflicting numbers written: {', '.join(f'{n:g}' for n in distinct)}")
        return None, start, notes
    if distinct and start >= 2 and tokens[start - 1].text in _RANGE_WORDS and (
        tokens[start - 2].number is not None or tokens[start - 2].is_number_word
    ):
        notes.append("a range or alternative of numbers is written")
        return None, start - 2, notes
    return (distinct[0] if distinct else None), start, notes


# --------------------------------------------------------------------------- durations
_DURATION_UNITS = {
    "day": "day", "days": "day", "يوم": "day", "يوما": "day", "ايام": "day",
    "week": "week", "weeks": "week", "اسبوع": "week", "اسابيع": "week",
    "month": "month", "months": "month", "شهر": "month", "شهرا": "month", "اشهر": "month", "شهور": "month",
    "year": "year", "years": "year", "سنه": "year", "سنوات": "year", "سنين": "year", "عام": "year", "اعوام": "year",
}
_DUAL_UNITS = {"يومان": "day", "يومين": "day", "اسبوعان": "week", "اسبوعين": "week", "شهران": "month",
               "شهرين": "month", "سنتان": "year", "سنتين": "year", "عامان": "year", "عامين": "year"}
_SINGULAR = {"day", "week", "month", "year", "يوم", "اسبوع", "شهر", "سنه", "عام"}
_QUALIFIERS = {"working": "working", "business": "business", "calendar": "calendar"}
_PER_WORDS = {"per", "a", "an", "each", "every", "كل"}


def find_durations(text: str, *, allow_bare_singular: bool = False) -> list[Reading]:
    matched, tokens = tokenize(text)
    readings = []
    for i, token in enumerate(tokens):
        unit = _DURATION_UNITS.get(token.text)
        dual = _DUAL_UNITS.get(token.text)
        if unit is None and dual is None:
            continue
        if i > 0 and tokens[i - 1].text in _PER_WORDS:
            continue
        end = token.end
        qualifier = None
        cursor = i
        if cursor > 0 and tokens[cursor - 1].text in _QUALIFIERS:
            qualifier = _QUALIFIERS[tokens[cursor - 1].text]
            cursor -= 1
        if i + 1 < len(tokens) and tokens[i + 1].text == "عمل":
            qualifier, end = "working", tokens[i + 1].end
        count, start, notes = _number_before(tokens, cursor)
        if dual is not None:
            unit = dual
            if count is None and not notes:
                count, start = 2.0, i
        elif count is None and not notes and i + 1 < len(tokens) and tokens[i + 1].text in ("واحد", "واحده"):
            count, start, end = 1.0, i, tokens[i + 1].end
        elif count is None and not notes:
            if allow_bare_singular and token.text in _SINGULAR:
                count, start = 1.0, i
                notes = ["singular unit written without a number"]
            else:
                continue
        raw = matched.span(tokens[start].start, end)
        value = None if count is None else {"count": count, "unit": unit, "qualifier": qualifier}
        readings.append(Reading(value, raw, notes))
    return readings


# --------------------------------------------------------------------------- hours
_HOUR_UNITS = {"hour", "hours", "hrs", "hr", "ساعه", "ساعات"}
_DAY_MARKERS = {"daily", "يوميا", "يومي", "اليوم"}
_WEEK_MARKERS = {"weekly", "اسبوعيا", "اسبوعي", "الاسبوع"}


def find_hours(text: str) -> list[Reading]:
    matched, tokens = tokenize(text)
    readings = []
    for i, token in enumerate(tokens):
        if token.text not in _HOUR_UNITS:
            continue
        count, start, notes = _number_before(tokens, i)
        if count is None and not notes:
            continue
        period, end = None, token.end
        for k in range(i + 1, min(i + 4, len(tokens))):
            word = tokens[k].text
            if word in _DAY_MARKERS or (word == "day" and tokens[k - 1].text in _PER_WORDS | {"في"}):
                period, end = "day", tokens[k].end
                break
            if word in _WEEK_MARKERS or (word == "week" and tokens[k - 1].text in _PER_WORDS | {"في"}):
                period, end = "week", tokens[k].end
                break
        value = None if count is None else {"hours": count, "period": period}
        readings.append(Reading(value, matched.span(tokens[start].start, end), notes))
    return readings


# --------------------------------------------------------------------------- days
WEEK = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
_DAY_NAMES = {
    **{name: name for name in WEEK},
    "الاحد": "sunday", "الاثنين": "monday", "الثلاثاء": "tuesday", "الاربعاء": "wednesday",
    "الخميس": "thursday", "الجمعه": "friday", "السبت": "saturday",
}
_RANGE_SEPARATORS = {"to", "through", "till", "until", "-", "–", "الي", "حتي"}


def _day(token: str) -> str | None:
    if token in _DAY_NAMES:
        return _DAY_NAMES[token]
    if token.startswith("و") and token[1:] in _DAY_NAMES:
        return _DAY_NAMES[token[1:]]
    return None


def find_working_days(text: str) -> list[Reading]:
    matched, tokens = tokenize(text)
    readings: list[Reading] = []
    for i, token in enumerate(tokens):  # "5 days per week" / "خمسة أيام في الأسبوع"
        if _DURATION_UNITS.get(token.text) != "day":
            continue
        following = [t.text for t in tokens[i + 1: i + 4]]
        if not ({"week", "الاسبوع", "اسبوعيا"} & set(following)):
            continue
        count, start, notes = _number_before(tokens, i)
        if count is None and not notes:
            continue
        end = next(t.end for t in tokens[i + 1: i + 4] if t.text in {"week", "الاسبوع", "اسبوعيا"})
        value = None if count is None else {"days_per_week": int(count), "days": []}
        readings.append(Reading(value, matched.span(tokens[start].start, end), notes))

    i = 0
    while i < len(tokens):  # named days: ranges and lists
        first = _day(tokens[i].text)
        if first is None:
            i += 1
            continue
        if i + 2 < len(tokens) and tokens[i + 1].text in _RANGE_SEPARATORS and _day(tokens[i + 2].text):
            last = _day(tokens[i + 2].text)
            a, b = WEEK.index(first), WEEK.index(last)
            days = [WEEK[(a + k) % 7] for k in range(((b - a) % 7) + 1)]
            readings.append(Reading({"days_per_week": None, "days": days},
                                    matched.span(tokens[i].start, tokens[i + 2].end)))
            i += 3
            continue
        names, j = [first], i + 1
        while j < len(tokens) and (tokens[j].text in {"and", "و"} or _day(tokens[j].text)):
            if _day(tokens[j].text):
                names.append(_day(tokens[j].text))
            j += 1
        if len(names) >= 2:
            readings.append(Reading({"days_per_week": None, "days": names},
                                    matched.span(tokens[i].start, tokens[j - 1].end)))
        i = j
    return readings


# --------------------------------------------------------------------------- money
_CURRENCY_TOKENS = {"sar": "SAR", "sr": "SAR", "riyal": "SAR", "riyals": "SAR", "ريال": "SAR", "ريالا": "SAR",
                    "ريالات": "SAR", "usd": "USD", "دولار": "USD"}
_MONTHLY = {"monthly", "month", "شهريا", "شهري", "الشهر"}
_ANNUAL = {"annual", "annually", "yearly", "annum", "سنويا", "سنوي"}


def _currency(tokens: list[Token], index: int, window: int = 2) -> tuple[str, int, int] | None:
    """Currency written next to tokens[index]: (code, first token index, last token index)."""
    order = sorted(range(max(0, index - window), min(len(tokens), index + window + 1)), key=lambda k: abs(k - index))
    for k in order:
        text = tokens[k].text
        if text in _CURRENCY_TOKENS:
            return _CURRENCY_TOKENS[text], min(k, index), max(k, index)
        if text == "ر" and k + 1 < len(tokens) and tokens[k + 1].text == "س":
            return "SAR", min(k, index), max(k + 1, index)
    return None


def _period(tokens: list[Token]) -> str | None:
    words = {t.text for t in tokens}
    monthly, annual = bool(words & _MONTHLY), bool(words & _ANNUAL)
    if monthly and not annual:
        return "monthly"
    if annual and not monthly:
        return "annual"
    return None


def find_money(text: str, *, require_currency: bool) -> tuple[list[Reading], list[Reading]]:
    """Returns (amount readings, percentage readings)."""
    matched, tokens = tokenize(text)
    amounts, percentages = [], []
    period = _period(tokens)
    for i, token in enumerate(tokens):
        number = token.number
        if number is None:
            continue
        nxt = tokens[i + 1].text if i + 1 < len(tokens) else ""
        prv = tokens[i - 1].text if i > 0 else ""
        if nxt == "/" or prv == "/" or (nxt == "-" and i + 2 < len(tokens) and tokens[i + 2].number is not None):
            continue  # date or numeric range
        if nxt in ("%", "percent") or (nxt in ("في", "بال") and i + 2 < len(tokens) and tokens[i + 2].text in ("المائه", "المئه")):
            basis_start = tokens[i + 2].start if nxt == "%" and i + 2 < len(tokens) else None
            basis = matched.span(basis_start, len(matched.norm)).strip(" .") if basis_start is not None else None
            percentages.append(Reading({"percentage": number, "percentage_basis": basis or None},
                                       matched.span(token.start, tokens[i + 1].end)))
            continue
        currency = _currency(tokens, i)
        if require_currency and currency is None:
            continue
        first, last = (currency[1], currency[2]) if currency else (i, i)
        amounts.append(Reading({"amount": number, "currency": currency[0] if currency else None, "period": period},
                               matched.span(tokens[first].start, tokens[last].end)))
    return amounts, percentages


# --------------------------------------------------------------------------- dates
_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
    "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
    "يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4, "مايو": 5, "يونيو": 6, "يونيه": 6, "يوليو": 7, "يوليه": 7,
    "اغسطس": 8, "سبتمبر": 9, "اكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12,
}
_NUMERIC_DMY = re.compile(r"(?<!\d)(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})(?!\d)")
_NUMERIC_YMD = re.compile(r"(?<!\d)(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})(?!\d)")
_HIJRI_MARK = re.compile(r"^\s*(?:هـ|ه(?![\u0621-\u064A])|h\b|ah\b)", re.IGNORECASE)


def _iso(year: int, month: int, day: int) -> str | None:
    try:
        return dt.date(year, month, day).isoformat()
    except ValueError:
        return None


def find_dates(text: str) -> list[Reading]:
    ascii_text = to_ascii_digits(text)
    readings: list[Reading] = []
    taken: list[tuple[int, int]] = []

    def add(match_start: int, match_end: int, iso: str | None, notes: list[str], year: int) -> None:
        mark = _HIJRI_MARK.match(ascii_text[match_end: match_end + 4])
        calendar = "hijri" if mark or 1300 <= year <= 1500 else "gregorian"
        if mark:
            match_end += len(mark.group().rstrip())
        if calendar == "hijri":
            iso, notes = None, [*notes, "Hijri date; not converted"]
        taken.append((match_start, match_end))
        readings.append(Reading({"raw": text[match_start:match_end], "iso_date": iso, "calendar": calendar},
                                text[match_start:match_end], notes))

    for match in _NUMERIC_YMD.finditer(ascii_text):
        year, month, day = (int(g) for g in match.groups())
        add(match.start(), match.end(), _iso(year, month, day), [], year)
    for match in _NUMERIC_DMY.finditer(ascii_text):
        if any(s <= match.start() < e for s, e in taken):
            continue
        first, second, year = (int(g) for g in match.groups())
        if first > 12 >= second:
            iso, notes = _iso(year, second, first), []
        elif second > 12 >= first:
            iso, notes = _iso(year, first, second), ["read as month/day/year"]
        elif first == second:
            iso, notes = _iso(year, first, second), []
        else:
            iso, notes = None, ["day/month order cannot be determined"]
        add(match.start(), match.end(), iso, notes, year)

    matched, tokens = tokenize(text)
    for i, token in enumerate(tokens):
        month = _MONTHS.get(token.text)
        if month is None:
            continue
        day_token = tokens[i - 1] if i > 0 and tokens[i - 1].number is not None else None
        year_token = tokens[i + 1] if i + 1 < len(tokens) and re.fullmatch(r"\d{4}", tokens[i + 1].text) else None
        if day_token is None and i + 2 < len(tokens) and tokens[i + 1].number is not None and re.fullmatch(r"\d{4}", tokens[i + 2].text):
            day_token, year_token = tokens[i + 1], tokens[i + 2]  # "October 15 2026"
            start_token, end_token = token, tokens[i + 2]
        else:
            start_token, end_token = day_token or token, year_token
        if day_token is None or year_token is None:
            continue
        raw = matched.span(start_token.start, end_token.end)
        readings.append(Reading({"raw": raw, "iso_date": _iso(int(year_token.text), month, int(day_token.number)),
                                 "calendar": "gregorian"}, raw))
    return readings


# --------------------------------------------------------------------------- misc
def years_in(text: str) -> list[int]:
    return [int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", to_ascii_digits(text))]


def contract_type(text: str) -> str | None:
    key = match_text(text, keep_punctuation=False).norm
    key = " ".join(key.split())
    if "غير محدد" in key or re.search(r"\b(indefinite|unlimited|open ended|permanent)\b", key):
        return "indefinite"
    if "محدد المده" in key or "محدد" in key or re.search(r"\b(fixed term|fixed|limited)\b", key):
        return "fixed_term"
    return None
