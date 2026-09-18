"""Helpers shared by the contract and CV analysis agents."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from enum import Enum
from typing import Any

from sanad.models.analysis import DocumentFact
from sanad.models.extraction import ExtractedField


def json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return value


def fact_from_field(name: str, field: ExtractedField) -> DocumentFact:
    """Copy a Phase 3 field (value, raw text, provenance) without re-reading the document."""
    primary = field.sources[0] if field.sources else None
    return DocumentFact(
        field=name,
        extraction_status=field.status,
        value=json_value(field.value),
        raw_value=field.raw_value,
        source_text=field.source_text,
        page_number=field.page_number,
        section_id=primary.section_id if primary else None,
        table_id=primary.table_id if primary else None,
        sources=list(field.sources),
        candidates=list(field.candidates),
        notes=list(field.notes),
    )


def status_counts(statuses: Iterable[Enum]) -> dict[str, int]:
    counts = Counter(status.value for status in statuses)
    return dict(sorted(counts.items()))


def label(field_name: str) -> str:
    return field_name.replace("_", " ")
