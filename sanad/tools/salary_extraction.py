"""Reads salary figures out of retrieved pages, and never guesses one.

Rules kept deliberately strict, because the output becomes cited evidence:
  * a figure is kept only with a currency it states and a period (monthly / annual) stated on the
    line or in the page's own heading; otherwise it is reported but may not drive a range;
  * annual figures are normalised by / 12; other currencies only by an explicitly configured rate;
  * base salary and total compensation (allowances, bonuses, overtime) are labelled, never merged;
  * figures outside a plausible monthly band are dropped rather than "corrected";
  * every observation keeps the exact sentence it was read from, the URL and the retrieval date.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from models.analysis import SalaryBasis, SalaryObservation
from tools.salary_sources import SourceProfile, profile_for
from tools.web_search import WebSearchResult

MIN_MONTHLY_SAR = 500.0        # below this a "salary" is a typo, a grade number or a daily rate
MAX_MONTHLY_SAR = 500_000.0
MAX_OBSERVATIONS_PER_PAGE = 6

CURRENCY_TOKENS = {
    "SAR": ("sar", "s.r", "sr", "ريال", "ر.س", "riyal", "riyals"),
    "USD": ("usd", "us$", "$", "dollar", "dollars"),
    "EUR": ("eur", "€", "euro", "euros"),
    "GBP": ("gbp", "£", "pound", "pounds"),
    "AED": ("aed", "dirham", "درهم"),
    "EGP": ("egp", "جنيه"),
}
MONTHLY_WORDS = ("per month", "a month", "monthly", "/month", "/mo", "month)", "month:", "شهري", "شهريا", "شهرياً",
                 "الشهري", "بالشهر", "في الشهر")
ANNUAL_WORDS = ("per year", "a year", "per annum", "annual", "annually", "yearly", "/year", "/yr", "سنوي", "سنويا",
                "سنوياً", "السنوي", "في السنة")
TOTAL_WORDS = ("total compensation", "total pay", "including bonus", "incl. bonus", "with bonuses", "gross total",
               "total salary", "total monthly salary including", "allowances, bonuses", "bonuses, overtimes",
               "package", "الإجمالي", "شامل", "البدلات", "مكافآت", "إجمالي")
BASE_WORDS = ("base salary", "basic salary", "base pay", "basic pay", "excluding bonus", "without bonuses",
              "الراتب الأساسي", "الأساسي")
SALARY_WORDS = ("salary", "salaries", "wage", "wages", "pay", "compensation", "راتب", "الراتب", "رواتب", "أجر", "الأجر",
                "أجور")
RANGE_SEPARATORS = r"(?:-|–|—|to|إلى|حتى|until)"
_NUMBER = r"\d{1,3}(?:[,٬ ]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_CURRENCY_BETWEEN = r"(?:\s*(?:sar|usd|eur|gbp|aed|egp|ر\.س|ريال|riyals?|\$|€|£))?"
RANGE_RE = re.compile(rf"({_NUMBER}){_CURRENCY_BETWEEN}\s*{RANGE_SEPARATORS}\s*({_NUMBER})", re.IGNORECASE)
NUMBER_RE = re.compile(_NUMBER)
_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.DOTALL | re.IGNORECASE)
_WS_RE = re.compile(r"[ \t ]+")


@dataclass(frozen=True)
class ExtractionSettings:
    fx_rates_to_sar: dict[str, float]
    min_monthly: float = MIN_MONTHLY_SAR
    max_monthly: float = MAX_MONTHLY_SAR
    max_per_page: int = MAX_OBSERVATIONS_PER_PAGE


# --------------------------------------------------------------------------- text preparation
def plain_text(result: WebSearchResult) -> str:
    """HTML -> text, JSON -> one 'key: value' line per field, anything else unchanged."""
    body = result.content or result.snippet
    if "json" in (result.content_type or "") or result.url.lower().endswith(".json"):
        return _json_lines(body)
    text = _TAG_RE.sub(" ", body)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return _WS_RE.sub(" ", text)


def _json_lines(body: str) -> str:
    try:
        payload = json.loads(body)
    except ValueError:
        return body
    lines: list[str] = []

    def walk(node, prefix=""):
        if isinstance(node, dict):
            parts = [f"{key}: {value}" for key, value in node.items() if not isinstance(value, (dict, list))]
            if parts:
                lines.append((prefix + " " if prefix else "") + ", ".join(parts))
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    walk(value, f"{prefix} {key}".strip())
        elif isinstance(node, list):
            for item in node:
                walk(item, prefix)
        else:
            lines.append(f"{prefix}: {node}".strip())

    walk(payload)
    return "\n".join(lines)


def sentences(text: str) -> list[str]:
    parts: list[str] = []
    for line in text.splitlines():
        for piece in re.split(r"(?<=[.!?؟۔])\s+|\s{2,}|\|", line):
            piece = piece.strip(" \t‏‎")
            if piece:
                parts.append(piece)
    return parts


# --------------------------------------------------------------------------- readers
def read_number(raw: str) -> float | None:
    """'12,000' -> 12000; '10.238' -> 10238 (Arabic-style thousands dot); '4 080' -> 4080."""
    cleaned = raw.replace("٬", ",").replace(" ", "").strip()
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", cleaned):
        cleaned = cleaned.replace(",", "")
    elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", cleaned):        # 10.238 / 1.234.567
        cleaned = cleaned.replace(".", "")
    elif re.fullmatch(r"\d{1,2}\.\d{3}", cleaned):             # 10.238 written once, as GASTAT does
        cleaned = cleaned.replace(".", "")
    try:
        value = float(cleaned.replace(",", ""))
    except ValueError:
        return None
    return value if value > 0 else None


def find_currency(text: str) -> str | None:
    lowered = text.lower()
    for code, tokens in CURRENCY_TOKENS.items():
        for token in tokens:
            if token in ("sr", "$", "€", "£"):
                if token in lowered:
                    return code
            elif re.search(rf"(?<![a-z]){re.escape(token)}(?![a-z])", lowered):
                return code
    return None


def find_period(text: str) -> str | None:
    lowered = text.lower()
    if any(word in lowered for word in MONTHLY_WORDS):
        return "monthly"
    if any(word in lowered for word in ANNUAL_WORDS):
        return "annual"
    return None


def find_basis(text: str) -> SalaryBasis:
    lowered = text.lower()
    if any(word in lowered for word in TOTAL_WORDS):
        return "total_compensation"
    if any(word in lowered for word in BASE_WORDS):
        return "base_salary"
    return "unspecified"


def mentions_salary(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in SALARY_WORDS)


# --------------------------------------------------------------------------- extraction
def extract_observations(result: WebSearchResult, settings: ExtractionSettings,
                         profile: SourceProfile | None = None) -> list[SalaryObservation]:
    profile = profile or profile_for(result.url)
    text = plain_text(result)
    page_period = find_period(f"{result.title} {result.snippet} {text[:400]}")
    page_basis = find_basis(f"{result.title} {result.snippet} {text[:400]}")
    retrieved_at = result.retrieved_at or ""

    observations: list[SalaryObservation] = []
    seen: set[tuple[float, float, str]] = set()
    for sentence in sentences(text):
        if len(observations) >= settings.max_per_page:
            break
        if not mentions_salary(sentence) and not find_currency(sentence):
            continue
        currency = find_currency(sentence) or (find_currency(result.title) if profile.tier == "official" else None)
        if currency is None:
            continue
        period = find_period(sentence) or page_period
        basis = find_basis(sentence)
        if basis == "unspecified":
            basis = page_basis
        for low, high in _amounts(sentence):
            key = (low, high, currency)
            if key in seen:
                continue
            observation = _observation(result, profile, sentence, low, high, currency, period, basis, retrieved_at,
                                       settings)
            if observation is not None:
                seen.add(key)
                observations.append(observation)
            if len(observations) >= settings.max_per_page:
                break
    return observations


def _amounts(sentence: str) -> list[tuple[float, float]]:
    found: list[tuple[float, float]] = []
    covered: list[tuple[int, int]] = []
    for match in RANGE_RE.finditer(sentence):
        low, high = read_number(match.group(1)), read_number(match.group(2))
        if low and high and high >= low:
            found.append((low, high))
            covered.append(match.span())
    for match in NUMBER_RE.finditer(sentence):
        if any(start <= match.start() < end for start, end in covered):
            continue
        value = read_number(match.group(0))
        if value is not None:
            found.append((value, value))
    return found


def _observation(result: WebSearchResult, profile: SourceProfile, sentence: str, low: float, high: float,
                 currency: str, period: str | None, basis: SalaryBasis, retrieved_at: str,
                 settings: ExtractionSettings) -> SalaryObservation | None:
    notes: list[str] = []
    usable = True
    monthly_low, monthly_high = low, high
    converted_from = None

    if currency != "SAR":
        rate = settings.fx_rates_to_sar.get(currency)
        if rate is None:
            return None if period is None else _unusable(result, profile, sentence, low, high, currency, period, basis,
                                                         retrieved_at,
                                                         f"{currency} was not converted: no exchange rate is configured.")
        monthly_low, monthly_high, converted_from = monthly_low * rate, monthly_high * rate, currency
        notes.append(f"Converted from {currency} at the configured rate {rate:g} SAR.")

    if period == "annual":
        monthly_low, monthly_high = monthly_low / 12, monthly_high / 12
        notes.append("Annual figure normalised to a month (/ 12).")
    elif period is None:
        return _unusable(result, profile, sentence, low, high, currency, "monthly", basis, retrieved_at,
                         "The page does not say whether this figure is monthly or annual, so it is not used for the "
                         "range.")

    if not (settings.min_monthly <= monthly_low <= settings.max_monthly
            and settings.min_monthly <= monthly_high <= settings.max_monthly):
        return None

    if profile.tier == "lead_only":
        usable = False
        notes.append("LinkedIn is treated as a discovery lead: this figure is never used as evidence on its own.")

    return SalaryObservation(
        source_name=profile.name, url=result.url, tier=profile.tier, retrieved_at=retrieved_at,
        quote=sentence[:400], title=result.title or None, minimum=low, maximum=high, currency=currency,
        period=period or "monthly", monthly_min=round(monthly_low, 2), monthly_max=round(monthly_high, 2),
        basis=basis, converted_from=converted_from, usable=usable, notes=notes,
    )


def _unusable(result: WebSearchResult, profile: SourceProfile, sentence: str, low: float, high: float, currency: str,
              period: str, basis: SalaryBasis, retrieved_at: str, reason: str) -> SalaryObservation:
    return SalaryObservation(
        source_name=profile.name, url=result.url, tier=profile.tier, retrieved_at=retrieved_at, quote=sentence[:400],
        title=result.title or None, minimum=low, maximum=high, currency=currency, period=period,
        monthly_min=low, monthly_max=high, basis=basis, usable=False, notes=[reason],
    )
