"""The adapter reads retrieval settings from rag/retriever.py instead of redefining them."""

import pytest

from rag.retrieval_config import RetrievalConfigError, read_answer_model, read_retrieval_config


def test_reads_active_collection_and_retrieval_settings_from_existing_rag(rag_dir):
    config = read_retrieval_config(rag_dir / "retriever.py")
    assert config.collection == "saudi_labor_law"
    assert config.qdrant_url == "http://localhost:6333"
    assert config.embedding_model == "intfloat/multilingual-e5-base"
    assert config.top_k == 5
    assert config.alpha == 0.6
    assert config.dense_similarity_top_k == 3


def test_reads_answer_model_from_existing_backend(rag_dir):
    assert read_answer_model(rag_dir / "backend.py") == "gpt-4o-mini"


def test_unreadable_or_changed_config_raises_explicit_error(tmp_path):
    with pytest.raises(RetrievalConfigError):
        read_retrieval_config(tmp_path / "does_not_exist.py")
    changed = tmp_path / "retriever.py"
    changed.write_text('QDRANT_URL = "http://localhost:6333"\n', encoding="utf-8")
    with pytest.raises(RetrievalConfigError, match="COLLECTION"):
        read_retrieval_config(changed)
