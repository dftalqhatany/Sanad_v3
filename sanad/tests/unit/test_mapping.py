"""Retriever/backend output -> structured evidence, without inventing or dropping information.

"legacy" in the names below mirrors the model field `RegulatoryEvidence.legacy_reference` and the
function `evidence_from_legacy_reference`, which keep their names because they appear in API output."""

import numpy as np
import pytest

from models.regulatory import SourceDocument
from rag.errors import InvalidLegacyOutputError
from rag.mapping import evidence_from_legacy_reference, evidence_from_retriever_result

SOURCE = SourceDocument(
    title_ar="نظام العمل", title_en="Saudi Labor Law", publisher="MHRSD",
    knowledge_base_file="labor_law_parsed.json", vector_collection="saudi_labor_law",
)


def test_full_metadata_is_preserved(knowledge_base):
    article = knowledge_base[110]  # Article 109 (annual leave)
    item = {"index": np.int64(110), "score": np.float64(0.73), "content": article["arabic_content"], "metadata": article}
    evidence = evidence_from_retriever_result(item, rank=1, source=SOURCE)
    assert evidence.raw_metadata == article
    assert evidence.reference.kb_index == 111
    assert evidence.reference.article_number == 109
    assert evidence.reference.english_number == "Article 109"
    assert evidence.reference.article_name_ar == article["arabic_name"]
    assert evidence.reference.part_title_ar == article["part_title_ar"]
    assert evidence.reference.chapter_title_ar == article["chapter_title_ar"]
    assert evidence.arabic_content == article["arabic_content"]  # byte-for-byte
    assert evidence.english_content == article["english_content"]
    assert isinstance(evidence.score, float) and evidence.score == pytest.approx(0.73)
    assert evidence.metadata_complete
    assert article["arabic_name"] in evidence.citation and "Article 109" in evidence.citation


def test_missing_metadata_is_handled_without_fabrication():
    evidence = evidence_from_retriever_result({"score": 0.4, "content": "نص", "metadata": {}}, rank=2, source=SOURCE)
    assert evidence.metadata_complete is False
    assert evidence.reference.article_name_ar is None and evidence.reference.kb_index is None
    assert evidence.arabic_content == "نص"
    assert "unidentified article" in evidence.citation


@pytest.mark.parametrize("bad", [None, "text", {"score": 0.5}, {"metadata": {}}, {"metadata": {}, "score": "high"}])
def test_malformed_legacy_results_raise(bad):
    with pytest.raises(InvalidLegacyOutputError):
        evidence_from_retriever_result(bad, rank=1, source=SOURCE)


def _legacy_reference(article, similarity=0.812):
    """Same keys/values rag.backend.answer_policy_question builds."""
    return {
        "similarity": similarity,
        "part": article["part_title_ar"],
        "chapter": article["chapter_title_ar"],
        "article_name": article["arabic_name"],
        "article_number": article["number_ar"],
        "arabic_content": article["arabic_content"],
        "english_content": article["english_content"],
    }


def test_legacy_reference_is_enriched_by_exact_knowledge_base_match(knowledge_base):
    article = knowledge_base[110]
    reference = _legacy_reference(article)
    evidence, warning = evidence_from_legacy_reference(reference, rank=1, source=SOURCE, documents=knowledge_base)
    assert warning is None
    assert evidence.metadata_complete and evidence.score_is_rounded
    assert evidence.raw_metadata == article
    assert evidence.legacy_reference == reference


def test_unmatched_legacy_reference_is_flagged_not_guessed(knowledge_base):
    reference = _legacy_reference(knowledge_base[110])
    reference["arabic_content"] = "نص معدل لا يطابق أي مادة"
    evidence, warning = evidence_from_legacy_reference(reference, rank=3, source=SOURCE, documents=knowledge_base)
    assert warning and "could not be matched" in warning
    assert evidence.metadata_complete is False
    assert evidence.reference.kb_index is None and evidence.reference.english_number is None
    assert evidence.reference.article_name_ar == knowledge_base[110]["arabic_name"]


def test_legacy_unknown_name_placeholder_is_not_treated_as_a_name(knowledge_base):
    reference = _legacy_reference(knowledge_base[0])
    reference["article_name"] = "غير معروفة"
    evidence, warning = evidence_from_legacy_reference(reference, rank=1, source=SOURCE, documents=knowledge_base)
    assert warning is not None
    assert evidence.reference.article_name_ar is None
