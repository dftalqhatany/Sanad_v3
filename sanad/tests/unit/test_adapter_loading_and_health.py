"""Import-time behaviour, loader safety and the read-only health check."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from config import SanadSettings
from models import ResultStatus
from rag import RegulatoryRAGAdapter
from rag.errors import LegacyInterfaceError, LegacyRagNotFoundError
from rag.loader import RagLoader
from tests.conftest import RAG_MODULE_NAMES, PROJECT_ROOT


def test_importing_sanad_and_building_adapter_has_no_rag_side_effects():
    code = (
        "import sys\n"
        "from rag import RegulatoryRAGAdapter\n"
        "adapter = RegulatoryRAGAdapter()\n"
        "heavy = [m for m in sys.modules if m.split('.')[0] in "
        "('rag.backend', 'rag.retriever', 'llama_index', 'openai', 'torch', 'transformers')]\n"
        "assert not heavy, heavy\n"
        "assert not adapter.is_backend_loaded\n"
        "print('clean')\n"
    )
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "clean"


def test_settings_read_secrets_from_environment_and_never_print_them():
    settings = SanadSettings.from_env({"OPENAI_API_KEY": "sk-secret-value"})
    assert settings.openai_api_key == "sk-secret-value"
    assert "sk-secret-value" not in repr(settings)
    assert SanadSettings.from_env({}).openai_api_key is None


def test_health_reports_unreachable_qdrant_explicitly(rag_dir, closed_qdrant_url):
    adapter = RegulatoryRAGAdapter(SanadSettings(rag_dir=rag_dir))
    health = adapter.health(qdrant_url=closed_qdrant_url)
    assert health.status is ResultStatus.ERROR
    assert health.vector_db_reachable is False
    assert [e.code for e in health.errors] == ["vector_db_unreachable"]
    assert health.knowledge_base_article_count == 249
    assert health.legacy_files_present is True
    assert health.legacy_backend_loaded is False
    assert health.retrieval_config.collection == "saudi_labor_law"


def test_missing_rag_directory_is_explicit(tmp_path):
    adapter = RegulatoryRAGAdapter(SanadSettings(rag_dir=tmp_path / "missing", openai_api_key="sk-test"))
    result = adapter.retrieve_evidence("What is the maximum probation period?")
    assert result.status is ResultStatus.ERROR
    assert result.errors[0].code == "legacy_rag_not_found"
    assert any("retrieval_config_unreadable" in w for w in result.warnings)
    health = adapter.health(check_vector_db=False)
    assert health.status is ResultStatus.ERROR
    assert {e.code for e in health.errors} >= {"legacy_rag_not_found", "knowledge_base_missing"}


@pytest.fixture
def isolated_rag_modules(monkeypatch):
    for name in RAG_MODULE_NAMES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield
    for name in RAG_MODULE_NAMES:
        sys.modules.pop(name, None)


def test_loader_detects_changed_backend_interface(rag_dir, isolated_rag_modules, monkeypatch):
    """A rag.backend that no longer defines the callables the adapter needs is reported, not used."""
    module = type(sys)("rag.backend")
    module.__file__ = str(rag_dir / "backend.py")
    module.get_retriever = lambda: None  # answer_policy_question and detect_language are gone
    monkeypatch.setitem(sys.modules, "rag.backend", module)
    cwd, path = os.getcwd(), list(sys.path)
    with pytest.raises(LegacyInterfaceError, match="answer_policy_question"):
        RagLoader(rag_dir).load()
    assert os.getcwd() == cwd and sys.path == path  # the loader no longer touches cwd or sys.path


def test_loader_refuses_a_same_named_module_from_elsewhere(tmp_path, rag_dir, isolated_rag_modules, monkeypatch):
    impostor = tmp_path / "backend.py"
    impostor.write_text("def answer_policy_question(*a, **k): pass\n", encoding="utf-8")
    module = type(sys)("rag.backend")
    module.__file__ = str(impostor)
    monkeypatch.setitem(sys.modules, "rag.backend", module)
    with pytest.raises(LegacyRagNotFoundError, match="already loaded"):
        RagLoader(rag_dir).load()
