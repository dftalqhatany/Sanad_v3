"""Turn candidate readings into ExtractedField objects (FOUND / NOT_FOUND / AMBIGUOUS / NOT_EXTRACTED)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from extraction.text import flat
from models.extraction import ExtractedField, ExtractionMethodName, FieldCandidate, FieldStatus, SourceSpan

HIGH_CONFIDENCE_METHODS = {ExtractionMethodName.LABELED_LINE, ExtractionMethodName.TABLE_ROW,
                           ExtractionMethodName.SECTION_CONTENT, ExtractionMethodName.PATTERN}


def normalized(value: Any) -> str | None:
    """Canonical one-line form of an extracted value, for comparison and display.

    Derived only from `value`; it never introduces a fact the document does not state. A field whose
    value is None (not found / ambiguous) has no normalized form.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return " ".join(value.split()) or None
    if isinstance(value, list):
        parts = [n for n in (normalized(item) for item in value) if n]
        return "; ".join(parts) or None
    name = type(value).__name__
    if name == "DurationValue":
        count = int(value.count) if float(value.count).is_integer() else value.count
        qualifier = f" {value.qualifier}" if value.qualifier else ""
        return f"{count}{qualifier} {value.unit}"
    if name == "DateValue":
        return value.iso_date or " ".join(value.raw.split())
    if name in ("MoneyValue", "AllowanceValue"):
        bits = []
        if getattr(value, "name", None):
            bits.append(f"{value.name}:")
        if getattr(value, "amount", None) is not None:
            amount = int(value.amount) if float(value.amount).is_integer() else value.amount
            bits.append(f"{value.currency} {amount}" if value.currency else str(amount))
        if getattr(value, "percentage", None) is not None:
            bits.append(f"{value.percentage:g}%")
        if getattr(value, "period", None):
            bits.append(value.period)
        return " ".join(bits) or None
    if name == "WorkingHoursValue":
        bits = [f"{h:g} h/{unit}" for h, unit in
                ((value.hours_per_day, "day"), (value.hours_per_week, "week")) if h is not None]
        if not bits and value.hours is not None:
            bits.append(f"{value.hours:g} h")
        return ", ".join(bits) or None
    if name == "WorkingDaysValue":
        bits = []
        if value.days_per_week is not None:
            bits.append(f"{value.days_per_week} days/week")
        if value.days:
            bits.append(", ".join(value.days))
        return " (".join(bits) + ")" if len(bits) == 2 else (bits[0] if bits else None)
    if name == "ContractTypeValue":
        return value.normalized or " ".join(value.raw.split())
    if name == "ClauseValue":
        return " ".join((value.heading or value.text).split())[:120]
    if name == "ListItem":
        return " ".join(value.text.split())
    return None


def candidate(value: Any, raw_value: str, source: SourceSpan, notes: Sequence[str] = ()) -> FieldCandidate | None:
    """A reading whose raw value is not literally present in its source text is rejected (no fabrication)."""
    raw_value = raw_value.strip()
    if not raw_value or flat(raw_value) not in flat(source.text):
        return None
    return FieldCandidate(value=value, raw_value=raw_value, source=source, notes=list(notes))


def not_extracted(name: str, reason: str) -> ExtractedField:
    return ExtractedField(name=name, status=FieldStatus.NOT_EXTRACTED, notes=[reason])


def not_applicable(name: str, source: SourceSpan, reason: str) -> ExtractedField:
    """The document explicitly states this topic does not apply - a real, positive fact read from the
    document, not silence. Unlike not_found/not_extracted, this always carries the source (the
    statement itself) so the determination stays traceable; it never carries a value."""
    return ExtractedField(name=name, status=FieldStatus.NOT_APPLICABLE, source_text=source.text,
                          page_number=source.page_number, sources=[source], notes=[reason])


def resolve(
    name: str,
    candidates: Sequence[FieldCandidate | None],
    *,
    key: Callable[[Any], Any] = lambda value: value,
    merge: Callable[[list[Any]], tuple[Any | None, list[str]]] | None = None,
) -> ExtractedField:
    items = _dedupe([c for c in candidates if c is not None])
    if not items:
        return ExtractedField(name=name, status=FieldStatus.NOT_FOUND)
    unreadable = [c for c in items if c.value is None]
    readable = [c for c in items if c.value is not None]
    notes: list[str] = []
    if unreadable:
        notes.append("the document states this field in a form that could not be read reliably")
        for c in unreadable:
            notes.extend(c.notes)
    value = None
    if readable:
        if merge is not None:
            value, merge_notes = merge([c.value for c in readable])
            notes.extend(merge_notes)
        else:
            distinct = {_hashable(key(c.value)) for c in readable}
            if len(distinct) == 1:
                value = readable[0].value
            else:
                notes.append(f"{len(distinct)} different values are written in the document")
    if unreadable or value is None:
        return ExtractedField(name=name, status=FieldStatus.AMBIGUOUS, candidates=items, notes=_unique(notes))
    first = readable[0]
    confidence = "high" if any(c.source.method in HIGH_CONFIDENCE_METHODS for c in readable) else "medium"
    for c in readable:
        notes.extend(c.notes)
    return ExtractedField(
        name=name,
        status=FieldStatus.FOUND,
        value=value,
        raw_value=first.raw_value,
        normalized_value=normalized(value),
        source_text=first.source.text,
        page_number=first.source.page_number,
        confidence=confidence,
        sources=_unique_sources([c.source for c in readable]),
        candidates=items if len(items) > 1 else [],
        notes=_unique(notes),
    )


def resolve_list(name: str, items: Sequence[Any], sources: Sequence[SourceSpan], notes: Sequence[str] = ()) -> ExtractedField:
    """List-valued fields (skills, clauses, entries): several explicit items are not a conflict."""
    if not items:
        return ExtractedField(name=name, status=FieldStatus.NOT_FOUND)
    sources = _unique_sources(list(sources))
    return ExtractedField(name=name, status=FieldStatus.FOUND, value=list(items), raw_value=sources[0].text,
                          normalized_value=normalized(list(items)),
                          source_text=sources[0].text, page_number=sources[0].page_number, confidence="high",
                          sources=sources, notes=list(notes))


def _dedupe(items: list[FieldCandidate]) -> list[FieldCandidate]:
    seen, result = set(), []
    for item in items:
        marker = (item.raw_value, item.source.text, item.source.section_id, item.source.table_id, item.source.method)
        if marker not in seen:
            seen.add(marker)
            result.append(item)
    return result


def _unique_sources(sources: list[SourceSpan]) -> list[SourceSpan]:
    seen, result = set(), []
    for source in sources:
        marker = (source.text, source.section_id, source.table_id)
        if marker not in seen:
            seen.add(marker)
            result.append(source)
    return result


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if hasattr(value, "model_dump"):
        return _hashable(value.model_dump())
    return value
