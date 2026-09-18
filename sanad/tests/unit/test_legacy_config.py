"""The adapter reads retrieval settings from the existing hybird_search.py instead of redefining them."""

import pytest

from sanad.rag.legacy_config import LegacyConfigError, read_legacy_answer_model, read_legacy_retrieval_config


def test_reads_active_collection_and_retrieval_settings_from_existing_rag(legacy_dir):
    config = read_legacy_retrieval_config(legacy_dir / "hybird_search.py")
    assert config.collection == "saudi_labor_law"
    assert config.qdrant_url == "http://localhost:6333"
    assert config.embedding_model == "intfloat/multilingual-e5-base"
    assert config.top_k == 5
    assert config.alpha == 0.6
    assert config.dense_similarity_top_k == 3


def test_reads_answer_model_from_existing_backend(legacy_dir):
    assert read_legacy_answer_model(legacy_dir / "chatbot_backend.py") == "gpt-4o-mini"


def test_unreadable_or_changed_config_raises_explicit_error(tmp_path):
    with pytest.raises(LegacyConfigError):
        read_legacy_retrieval_config(tmp_path / "does_not_exist.py")
    changed = tmp_path / "hybird_search.py"
    changed.write_text('QDRANT_URL = "http://localhost:6333"\n', encoding="utf-8")
    with pytest.raises(LegacyConfigError, match="COLLECTION"):
        read_legacy_retrieval_config(changed)
