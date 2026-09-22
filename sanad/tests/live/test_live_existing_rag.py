"""Live tests against the REAL infrastructure: running Qdrant (saudi_labor_law), e5 model, OpenAI.

Run on the machine where the existing RAG runs, from the sanad/ folder:
    PYTHONDONTWRITEBYTECODE=1 OPENAI_API_KEY=sk-... python -m pytest tests/live -v -rs

Tests skip (with the reason) when Qdrant is not reachable or the legacy dependencies are not
installed, and the LLM tests skip without OPENAI_API_KEY. If Qdrant IS reachable, a missing or
changed collection is a failure, not a skip.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

from config import SanadSettings
from models import ResultStatus
from rag import RegulatoryRAGAdapter

pytestmark = pytest.mark.live

LEGACY_DEPENDENCIES = (
    "llama_index.core", "llama_index.embeddings.huggingface", "llama_index.vector_stores.qdrant",
    "qdrant_client", "rank_bm25", "sklearn", "openai",
)
ANNUAL_LEAVE_QUESTION_AR = "كم مدة الإجازة السنوية التي يستحقها العامل؟"
TATWEEL = "ـ"


def _missing_dependencies() -> list[str]:
    missing = []
    for name in LEGACY_DEPENDENCIES:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except ModuleNotFoundError:
            missing.append(name)
    return missing


@pytest.fixture(scope="module")
def live_adapter():
    missing = _missing_dependencies()
    if missing:
        pytest.skip(f"existing RAG dependencies not installed: {missing}")
    adapter = RegulatoryRAGAdapter(SanadSettings.from_env())
    health = adapter.health()
    if health.vector_db_reachable is not True:
        pytest.skip(f"Qdrant not reachable at {health.vector_db_url}: {[e.message for e in health.errors]}")
    return adapter, health


@pytest.fixture(scope="module")
def api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


def test_live_collection_is_intact(live_adapter, knowledge_base):
    _, health = live_adapter
    assert not health.errors, health.errors
    assert health.collection_exists is True
    assert (health.collection_vector_size, health.collection_distance) == (768, "Cosine")
    assert health.collection_points_count >= len(knowledge_base) == 249
    from qdrant_client import QdrantClient

    client = QdrantClient(url=health.vector_db_url)
    try:
        names = {c.name for c in client.get_collections().collections}
        indices, offset = set(), None
        while True:  # read-only scroll over the payload field written by process_data_vectors.ipynb
            points, offset = client.scroll(
                "saudi_labor_law", limit=256, offset=offset, with_payload=["index"], with_vectors=False
            )
            indices.update(p.payload.get("index") for p in points)
            if offset is None:
                break
    finally:
        client.close()
    assert {"saudi_labor_law", "labor_law_ar"} <= names
    assert indices == set(range(1, 250)), "every knowledge-base article must still be indexed"


def test_live_adapter_returns_regulatory_evidence_from_existing_rag(live_adapter, knowledge_base):
    adapter, _ = live_adapter
    result = adapter.retrieve_evidence(ANNUAL_LEAVE_QUESTION_AR)
    assert result.status is ResultStatus.SUCCESS, result.errors
    assert len(result.evidence) == 5
    for item in result.evidence:
        assert item.metadata_complete
        assert item.raw_metadata == knowledge_base[item.reference.kb_index - 1]
        assert item.reference.article_name_ar and item.citation.startswith("نظام العمل")
    normalised = [item.arabic_content.replace(TATWEEL, "") for item in result.evidence]
    assert any("إجازة" in text or "اجازة" in text for text in normalised), "expected leave-related evidence"


def test_live_adapter_matches_direct_legacy_retrieval(live_adapter):
    adapter, _ = live_adapter
    result = adapter.retrieve_evidence(ANNUAL_LEAVE_QUESTION_AR)
    direct = sys.modules["rag.backend"].get_retriever().retrieve(ANNUAL_LEAVE_QUESTION_AR)
    assert [(e.raw_metadata["index"], round(e.score, 6)) for e in result.evidence] == [
        (int(r["metadata"]["index"]), round(float(r["score"]), 6)) for r in direct
    ]


def test_live_top_semantic_hit_is_credited_to_the_article_qdrant_returned(live_adapter):
    """Regression for the approved retriever.py index fix, on the real collection.

    After min-max normalisation the top Qdrant hit contributes alpha * 1.0 = 0.6, while articles without a
    semantic hit can reach at most 0.4, so only the other two Qdrant hits can outrank it.
    """
    adapter, _ = live_adapter
    result = adapter.retrieve_evidence(ANNUAL_LEAVE_QUESTION_AR)
    dense_hits = sys.modules["rag.backend"].get_retriever().dense.retrieve(ANNUAL_LEAVE_QUESTION_AR)
    assert dense_hits, "Qdrant returned no semantic hits"
    top_index = int(max(dense_hits, key=lambda hit: hit.score).node.metadata["index"])
    assert top_index in [item.reference.kb_index for item in result.evidence[:3]]


@pytest.mark.live_llm
def test_live_existing_rag_still_answers_a_normal_question_like_streamlit(live_adapter, api_key):
    live_adapter[0].retrieve_evidence("warm up")  # ensure the legacy module is loaded via the adapter
    backend = sys.modules["rag.backend"]
    answer, references = backend.answer_policy_question(ANNUAL_LEAVE_QUESTION_AR, None, api_key=api_key)
    assert isinstance(answer, str) and answer.strip()
    assert references
    for key in ("similarity", "part", "chapter", "article_name", "article_number", "arabic_content", "english_content"):
        assert key in references[0]  # keys app.py renders


@pytest.mark.live_llm
def test_live_adapter_ask_returns_answer_with_evidence(live_adapter, api_key):
    adapter, _ = live_adapter
    result = adapter.ask(ANNUAL_LEAVE_QUESTION_AR, api_key=api_key)
    assert result.status is ResultStatus.SUCCESS, (result.errors, result.warnings)
    assert result.answer and result.evidence
    assert all(item.metadata_complete for item in result.evidence)
    direct = sys.modules["rag.backend"].get_retriever().retrieve(ANNUAL_LEAVE_QUESTION_AR)
    assert [e.raw_metadata["index"] for e in result.evidence] == [int(r["metadata"]["index"]) for r in direct]
