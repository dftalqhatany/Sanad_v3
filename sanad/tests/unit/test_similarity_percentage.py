"""Similarity is reported as a percentage for display, and stays a SIMILARITY score.

The number says how closely a retrieved article matches the question. It says nothing about whether the
answer is correct, so nothing here may present it as accuracy or confidence.
"""

from __future__ import annotations

import pytest

from models.regulatory import ArticleReference, RegulatoryEvidence, SourceDocument

SOURCE = SourceDocument(title_ar="نظام العمل", title_en="Saudi Labor Law", publisher="MHRSD",
                        knowledge_base_file="kb.json")


def evidence(score: float) -> RegulatoryEvidence:
    return RegulatoryEvidence(
        reference=ArticleReference(kb_index=55, article_number="54"), citation="نظام العمل — المادة 54",
        arabic_content="…", rank=1, score=score, score_is_rounded=False, source=SOURCE,
        metadata_complete=True, raw_metadata={},
    )


@pytest.mark.parametrize("score, percent", [(0.91, 91), (0.87, 87), (0.84, 84), (0.74, 74),
                                            (1.0, 100), (0.0, 0), (0.006, 1), (0.6051, 61)])
def test_the_percentage_is_the_score_times_one_hundred(score, percent):
    assert evidence(score).similarity_percentage == percent


def test_the_percentage_is_derived_not_stored():
    """One source of truth: `score`. The percentage is computed, so the two can never disagree."""
    item = evidence(0.91)
    assert "similarity_percentage" not in type(item).model_fields   # not a second stored field
    assert item.model_dump()["similarity_percentage"] == 91         # but always present in the payload
    assert item.model_dump()["score"] == 0.91                       # and the raw score is still there
