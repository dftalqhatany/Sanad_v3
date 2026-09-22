"""Pure conversion of the existing RAG's output into Sanad models.

Two legacy output shapes are supported, both read-only:
  * HybridRetriever.retrieve()  -> [{"index", "score", "content", "metadata": <full article dict>}]
  * answer_policy_question()    -> references [{"similarity", "part", "chapter", "article_name",
                                               "article_number", "arabic_content", "english_content"}]
Missing fields become None; nothing is invented.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from models.regulatory import ArticleReference, RegulatoryEvidence, SourceDocument
from rag.errors import InvalidLegacyOutputError

UNKNOWN_ARTICLE_NAME_AR = "غير معروفة"  # placeholder the legacy backend uses when a name is missing
IDENTITY_KEYS = ("index", "arabic_name", "arabic_content")


def to_json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    item = getattr(value, "item", None)  # numpy scalar
    if callable(item):
        try:
            return to_json_safe(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _opt_int(value: Any) -> int | None:
    value = to_json_safe(value)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _opt_text(value: Any) -> str | None:
    """Evidence text is kept byte-for-byte; only blank values become None."""
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


def _score(value: Any) -> float:
    value = to_json_safe(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(float(value)):
        raise InvalidLegacyOutputError(f"missing or non-numeric retrieval score: {value!r}")
    return float(value)


def article_reference_from_metadata(metadata: Mapping[str, Any]) -> ArticleReference:
    name = _opt_str(metadata.get("arabic_name"))
    if name == UNKNOWN_ARTICLE_NAME_AR:
        name = None
    return ArticleReference(
        kb_index=_opt_int(metadata.get("index")),
        article_number=_opt_int(metadata.get("article_number")),
        article_number_ar=_opt_str(metadata.get("number_ar")),
        article_name_ar=name,
        english_number=_opt_str(metadata.get("english_number")),
        part_number=_opt_int(metadata.get("part_number")),
        part_number_ar=_opt_str(metadata.get("part_number_ar")),
        part_title_ar=_opt_str(metadata.get("part_title_ar")),
        chapter_number=_opt_int(metadata.get("chapter_number")),
        chapter_number_ar=_opt_str(metadata.get("chapter_number_ar")),
        chapter_title_ar=_opt_str(metadata.get("chapter_title_ar")),
    )


def is_metadata_complete(metadata: Mapping[str, Any]) -> bool:
    if any(_opt_str(metadata.get(key)) is None for key in IDENTITY_KEYS):
        return False
    return _opt_str(metadata.get("arabic_name")) != UNKNOWN_ARTICLE_NAME_AR


def build_citation(reference: ArticleReference, source: SourceDocument) -> str:
    segments = [source.title_ar]
    if reference.article_name_ar:
        suffix = f" ({reference.english_number})" if reference.english_number else ""
        segments.append(reference.article_name_ar + suffix)
    elif reference.english_number:
        segments.append(reference.english_number)
    else:
        segments.append("مادة غير محددة / unidentified article")
    if reference.part_title_ar:
        label = f"الباب {reference.part_number_ar}" if reference.part_number_ar else "الباب"
        segments.append(f"{label}: {reference.part_title_ar}")
    if reference.chapter_title_ar:
        label = f"الفصل {reference.chapter_number_ar}" if reference.chapter_number_ar else "الفصل"
        segments.append(f"{label}: {reference.chapter_title_ar}")
    return " — ".join(segments)


def _evidence(
    raw_metadata: dict[str, Any],
    *,
    rank: int,
    score: float,
    score_is_rounded: bool,
    source: SourceDocument,
    metadata_complete: bool,
    fallback_content: Any = None,
    legacy_reference: dict[str, Any] | None = None,
) -> RegulatoryEvidence:
    reference = article_reference_from_metadata(raw_metadata)
    return RegulatoryEvidence(
        reference=reference,
        citation=build_citation(reference, source),
        arabic_content=_opt_text(raw_metadata.get("arabic_content")) or _opt_text(fallback_content),
        english_content=_opt_text(raw_metadata.get("english_content")),
        rank=rank,
        score=score,
        score_is_rounded=score_is_rounded,
        source=source,
        metadata_complete=metadata_complete,
        raw_metadata=raw_metadata,
        legacy_reference=legacy_reference,
    )


def evidence_from_retriever_result(item: Any, *, rank: int, source: SourceDocument) -> RegulatoryEvidence:
    if not isinstance(item, Mapping):
        raise InvalidLegacyOutputError(f"HybridRetriever.retrieve returned {type(item).__name__}, expected dict")
    metadata = item.get("metadata")
    if not isinstance(metadata, Mapping):
        raise InvalidLegacyOutputError("HybridRetriever.retrieve result has no 'metadata' dict")
    raw = to_json_safe(metadata)
    return _evidence(
        raw,
        rank=rank,
        score=_score(item.get("score")),
        score_is_rounded=False,
        source=source,
        metadata_complete=is_metadata_complete(raw),
        fallback_content=item.get("content"),
    )


def evidence_from_legacy_reference(
    reference: Any,
    *,
    rank: int,
    source: SourceDocument,
    documents: Sequence[Mapping[str, Any]] | None,
) -> tuple[RegulatoryEvidence, str | None]:
    """Map one answer_policy_question() reference; enrich it from the legacy KB by exact match only."""
    if not isinstance(reference, Mapping):
        raise InvalidLegacyOutputError(f"answer_policy_question reference is {type(reference).__name__}, expected dict")
    legacy_reference = to_json_safe(reference)
    score = _score(reference.get("similarity"))
    name = reference.get("article_name")
    content = reference.get("arabic_content")

    match = None
    if documents is not None and name and name != UNKNOWN_ARTICLE_NAME_AR and content:
        matches = [
            doc
            for doc in documents
            if isinstance(doc, Mapping) and doc.get("arabic_name") == name and doc.get("arabic_content") == content
        ]
        if len(matches) == 1:
            match = matches[0]

    if match is not None:
        raw = to_json_safe(match)
        return (
            _evidence(raw, rank=rank, score=score, score_is_rounded=True, source=source,
                      metadata_complete=is_metadata_complete(raw), legacy_reference=legacy_reference),
            None,
        )

    raw = to_json_safe(
        {
            "part_title_ar": reference.get("part"),
            "chapter_title_ar": reference.get("chapter"),
            "arabic_name": name,
            "number_ar": reference.get("article_number"),
            "arabic_content": content,
            "english_content": reference.get("english_content"),
        }
    )
    warning = (
        f"Reference #{rank} ({_opt_str(name) or 'unnamed'}) could not be matched to exactly one knowledge-base "
        "article; only the fields returned by answer_policy_question are available."
    )
    return (
        _evidence(raw, rank=rank, score=score, score_is_rounded=True, source=source,
                  metadata_complete=False, legacy_reference=legacy_reference),
        warning,
    )
