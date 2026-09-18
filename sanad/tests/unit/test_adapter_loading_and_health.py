"""Import-time behaviour, loader safety and the read-only health check."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from sanad.config import SanadSettings
from sanad.models import ResultStatus
from sanad.rag import RegulatoryRAGAdapter
from sanad.rag.errors import LegacyInterfaceError, LegacyRagNotFoundError
from sanad.rag.legacy_loader import LegacyRagLoader
from tests.conftest import LEGACY_MODULE_NAMES, PROJECT_ROOT


def test_importing_sanad_and_building_adapter_has_no_legacy_side_effects():
    code = (
        "import sys\n"
        "from sanad.rag import RegulatoryRAGAdapter\n"
        "adapter = RegulatoryRAGAdapter()\n"
        "heavy = [m for m in sys.modules if m.split('.')[0] in "
        "('chatbot_backend', 'hybird_search', 'llama_index', 'openai', 'torch', 'transformers')]\n"
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


def test_health_reports_unreachable_qdrant_explicitly(legacy_dir, closed_qdrant_url):
    adapter = RegulatoryRAGAdapter(SanadSettings(legacy_rag_dir=legacy_dir))
    health = adapter.health(qdrant_url=closed_qdrant_url)
    assert health.status is ResultStatus.ERROR
    assert health.vector_db_reachable is False
    assert [e.code for e in health.errors] == ["vector_db_unreachable"]
    assert health.knowledge_base_article_count == 249
    assert health.legacy_files_present is True
    assert health.legacy_backend_loaded is False
    assert health.retrieval_config.collection == "saudi_labor_law"


def test_missing_legacy_directory_is_explicit(tmp_path):
    adapter = RegulatoryRAGAdapter(SanadSettings(legacy_rag_dir=tmp_path / "missing", openai_api_key="sk-test"))
    result = adapter.retrieve_evidence("What is the maximum probation period?")
    assert result.status is ResultStatus.ERROR
    assert result.errors[0].code == "legacy_rag_not_found"
    assert any("retrieval_config_unreadable" in w for w in result.warnings)
    health = adapter.health(check_vector_db=False)
    assert health.status is ResultStatus.ERROR
    assert {e.code for e in health.errors} >= {"legacy_rag_not_found", "knowledge_base_missing"}


@pytest.fixture
def isolated_legacy_modules(monkeypatch):
    for name in LEGACY_MODULE_NAMES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield
    for name in LEGACY_MODULE_NAMES:
        sys.modules.pop(name, None)


def test_loader_detects_changed_legacy_interface(tmp_path, isolated_legacy_modules):
    (tmp_path / "hybird_search.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "chatbot_backend.py").write_text("def get_retriever():\n    return None\n", encoding="utf-8")
    cwd, path = os.getcwd(), list(sys.path)
    with pytest.raises(LegacyInterfaceError, match="answer_policy_question"):
        LegacyRagLoader(tmp_path).load()
    assert os.getcwd() == cwd and sys.path == path


def test_loader_refuses_a_same_named_module_from_elsewhere(tmp_path, legacy_dir, isolated_legacy_modules, monkeypatch):
    impostor = tmp_path / "chatbot_backend.py"
    impostor.write_text("def answer_policy_question(*a, **k): pass\n", encoding="utf-8")
    module = type(sys)("chatbot_backend")
    module.__file__ = str(impostor)
    monkeypatch.setitem(sys.modules, "chatbot_backend", module)
    with pytest.raises(LegacyRagNotFoundError, match="already loaded"):
        LegacyRagLoader(legacy_dir).load()
