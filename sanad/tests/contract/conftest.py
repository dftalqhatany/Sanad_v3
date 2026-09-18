from __future__ import annotations

import sys

import pytest

from sanad.config import SanadSettings
from sanad.rag import RegulatoryRAGAdapter
from tests.conftest import LEGACY_MODULE_NAMES
from tests.fakes.legacy_stubs import StubState, build_stub_modules

TEST_KEY = "sk-test-not-a-real-key"


@pytest.fixture
def infra(monkeypatch, knowledge_base) -> StubState:
    """Real legacy RAG code + real KB, with Qdrant/embedding model/OpenAI replaced by doubles."""
    state = StubState(documents=knowledge_base)
    for name in LEGACY_MODULE_NAMES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name, module in build_stub_modules(state).items():
        monkeypatch.setitem(sys.modules, name, module)
    yield state
    for name in LEGACY_MODULE_NAMES:
        sys.modules.pop(name, None)


@pytest.fixture
def adapter(legacy_dir, infra) -> RegulatoryRAGAdapter:
    return RegulatoryRAGAdapter(SanadSettings(legacy_rag_dir=legacy_dir, openai_api_key=TEST_KEY))


def legacy_backend():
    return sys.modules["chatbot_backend"]
