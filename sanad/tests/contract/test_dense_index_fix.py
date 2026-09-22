"""Regression tests for the approved one-line fix in rag/retriever.py.

Bug: the Qdrant payload field "index" is 1-based (labor_law_parsed.json) but was used directly as a
0-based position in dense_scores, so every semantic hit was credited to the NEXT article and a hit
on the last article was dropped.

Real legacy code, real knowledge base, real BM25 / scikit-learn / numpy; only Qdrant, the e5 model
and OpenAI are simulated (tests/fakes).
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from models import ResultStatus

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CHANGE = json.loads((FIXTURES / "approved_legacy_changes.json").read_text(encoding="utf-8"))["changes"][
    "rag/retriever.py"
]
ARTICLE_109 = 111  # kb index of "المادة التاسعة بعد المائة"
LAST_ARTICLE = 249
NO_LEXICAL_OVERLAP = "zzqx no lexical overlap"
ALPHA = 0.6


class FakeDense:
    """Stands in for the Qdrant VectorIndexRetriever: returns nodes carrying the given payloads."""

    def __init__(self, hits):
        self.hits = hits  # [(payload dict, cosine score)]

    def retrieve(self, query):
        return [SimpleNamespace(node=SimpleNamespace(metadata=dict(payload)), score=score) for payload, score in self.hits]


def dense(pairs):
    return FakeDense([({"index": index}, score) for index, score in pairs])


def signature(results):
    return [(int(r["index"]), float(r["score"]), int(r["metadata"]["index"])) for r in results]


def queries(knowledge_base):
    lexical = [" ".join(knowledge_base[i - 1]["arabic_content"].split()[:8]) for i in (1, 50, ARTICLE_109, 200, LAST_ARTICLE)]
    return lexical + ["annual leave", "إجازة سنوية", "فترة التجربة", NO_LEXICAL_OVERLAP]


def hit_configurations():
    rng = random.Random(20260917)
    configs = [
        [],
        [(ARTICLE_109, 0.88)],
        [(1, 0.90), (2, 0.85), (3, 0.80)],
        [(LAST_ARTICLE, 0.91), (248, 0.90), (100, 0.70)],
        [(ARTICLE_109, 0.86), (112, 0.86), (40, 0.50)],  # tie
    ]
    configs += [[(i, round(rng.uniform(0.6, 0.95), 4)) for i in rng.sample(range(1, 250), 3)] for _ in range(20)]
    return configs


@pytest.fixture
def fixed_module(adapter, rag_dir):
    adapter.retrieve_evidence("warm up")  # loads the real legacy modules through the adapter
    module = sys.modules["rag.retriever"]
    assert Path(module.__file__).resolve() == (rag_dir / "retriever.py").resolve()
    return module


@pytest.fixture
def original_module(infra, rag_dir, baseline):
    """Pre-fix retriever.py rebuilt in memory by reverting ONLY the fixed line (hash-checked, nothing written)."""
    current = (rag_dir / "retriever.py").read_bytes()
    fixed_line = (CHANGE["fixed_line"] + "\n").encode()
    original_line = (CHANGE["original_line"] + "\n").encode()
    assert current.count(fixed_line) == 1, "the approved fix is not applied to retriever.py"
    source = current.replace(fixed_line, original_line)
    # the historical baseline still holds the pre-fix hash under the file's original name
    assert hashlib.sha256(source).hexdigest() == baseline["protected_files_sha256"][CHANGE["migrated_from"]]
    module = types.ModuleType("retriever_original")
    exec(compile(source, "retriever_original.py", "exec"), module.__dict__)
    return module


# --------------------------------------------------------------------------- the bug is fixed
def test_article_109_semantic_hit_is_credited_to_article_109(adapter, infra, knowledge_base):
    infra.dense_hits = [(ARTICLE_109, 0.9)]
    result = adapter.retrieve_evidence(NO_LEXICAL_OVERLAP)
    assert result.status is ResultStatus.SUCCESS
    top = result.evidence[0]
    assert top.reference.kb_index == ARTICLE_109, f"credited to {top.reference.english_number}"
    assert top.reference.english_number == "Article 109"
    assert top.reference.article_name_ar == knowledge_base[ARTICLE_109 - 1]["arabic_name"] == "المادة التاسعة بعد المائة"
    assert top.score == pytest.approx(ALPHA)  # alpha * minmax(dense)=1.0 + (1 - alpha) * BM25=0
    assert all(item.score == 0 for item in result.evidence[1:])


def test_last_article_semantic_hit_is_not_dropped(adapter, infra, knowledge_base):
    infra.dense_hits = [(LAST_ARTICLE, 0.9)]
    result = adapter.retrieve_evidence(NO_LEXICAL_OVERLAP)
    assert result.status is ResultStatus.SUCCESS, "the hit on the last article was dropped"
    top = result.evidence[0]
    assert top.reference.kb_index == LAST_ARTICLE
    assert top.raw_metadata == knowledge_base[-1]
    assert top.score == pytest.approx(ALPHA)


def test_first_article_semantic_hit_is_credited_to_the_first_article(adapter, infra):
    infra.dense_hits = [(1, 0.9)]
    result = adapter.retrieve_evidence(NO_LEXICAL_OVERLAP)
    assert result.evidence[0].reference.kb_index == 1
    assert result.evidence[0].score == pytest.approx(ALPHA)


def test_several_semantic_hits_keep_their_articles_and_qdrant_order(adapter, infra):
    infra.dense_hits = [(ARTICLE_109, 0.90), (20, 0.85), (LAST_ARTICLE, 0.80)]
    result = adapter.retrieve_evidence(NO_LEXICAL_OVERLAP)
    assert [item.reference.kb_index for item in result.evidence[:3]] == [ARTICLE_109, 20, LAST_ARTICLE]
    assert [item.score for item in result.evidence[:3]] == pytest.approx([ALPHA, ALPHA * 0.85 / 0.9, ALPHA * 0.80 / 0.9])


# --------------------------------------------------------------------------- nothing else changed
def test_only_the_index_mapping_changed(fixed_module, original_module, knowledge_base):
    """fixed(hit on article m) == original(hit on m - 1) for all m in 1..249.

    Same BM25 scores, min-max normalisation, alpha fusion, top-k, tie-breaking and result shape;
    configuration without semantic hits (pure BM25) must be identical outright.
    """
    for name in ("QDRANT_URL", "COLLECTION", "EMBED_MODEL", "TOP_K", "ALPHA"):
        assert getattr(fixed_module, name) == getattr(original_module, name)
    compared = 0
    for config in hit_configurations():
        fixed = fixed_module.HybridRetriever(knowledge_base, dense_retriever=dense(config))
        original = original_module.HybridRetriever(knowledge_base, dense_retriever=dense([(m - 1, s) for m, s in config]))
        for query in queries(knowledge_base):
            assert signature(fixed.retrieve(query)) == signature(original.retrieve(query)), (config, query)
            assert signature(fixed.retrieve(query, top_k=10)) == signature(original.retrieve(query, top_k=10)), (config, query)
            compared += 2
    assert compared == 25 * 9 * 2


def test_hits_without_a_valid_index_are_still_ignored(fixed_module, original_module, knowledge_base):
    invalid = FakeDense([({}, 0.9), ({"index": None}, 0.9), ({"index": 250}, 0.9)])
    for query in queries(knowledge_base):
        fixed = fixed_module.HybridRetriever(knowledge_base, dense_retriever=invalid).retrieve(query)
        no_hits = original_module.HybridRetriever(knowledge_base, dense_retriever=FakeDense([])).retrieve(query)
        assert signature(fixed) == signature(no_hits)
