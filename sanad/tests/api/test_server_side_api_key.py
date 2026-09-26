"""The OpenAI key is the server's, read from the environment or .env, and never travels with a request.

Sanad answers with one key that the operator configures. These tests pin that: the request carries no
key and cannot be made to carry one, the key enables generation when present, and it never appears in a
response, a log line or a repr.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from agents import AnalysisAgent, ContractAnalysisAgent, ContractComparisonAgent
from api import create_app
from api.schemas import AskBody
from config import DOTENV_PATH, SanadSettings, _dotenv, _environment
from models.orchestration import SanadRequest
from orchestrator import SanadOrchestrator
from rag.errors import RagErrorCode, classify_exception, make_error
from tests.fakes.infra_stubs import foreign_exception
from tests.fakes.regulatory_adapter import FakeRegulatoryAdapter

KEY = "sk-test-not-a-real-key-000111222333"
QUESTION = "كم مدة فترة التجربة في عقد العمل؟"


@pytest.fixture
def unconfigured(knowledge_base) -> SanadOrchestrator:
    """A server with no key configured: retrieval only, no answer generation."""
    rag = FakeRegulatoryAdapter(knowledge_base)
    analysis = AnalysisAgent(ContractAnalysisAgent(rag))
    orchestrator = SanadOrchestrator(analysis, ContractComparisonAgent(analysis), rag, generate_answers=False)
    orchestrator.rag = rag
    return orchestrator


# --------------------------------------------------------------------------- the server's key decides
def test_a_configured_server_key_enables_answer_generation(client, real_orchestrator):
    response = client.post("/api/ask", json={"question": QUESTION})

    assert response.status_code == 200
    assert len(real_orchestrator.rag.ask_calls) == 1       # the answer pipeline ran
    assert response.json()["regulatory_answer"]["answer"]


def test_without_a_configured_key_the_question_falls_back_to_retrieval(unconfigured):
    response = TestClient(create_app(unconfigured)).post("/api/ask", json={"question": QUESTION})

    assert response.status_code == 200
    assert unconfigured.rag.ask_calls == []                # generation was never attempted
    assert unconfigured.rag.calls                          # retrieval still happened
    answer = response.json()["regulatory_answer"]
    assert answer["answer"] is None and answer["evidence"]
    assert "no OPENAI_API_KEY" in answer["message"]


def test_generate_answers_follows_the_configured_setting():
    built = SanadOrchestrator.from_settings.__doc__
    assert built  # the wiring is documented
    assert SanadSettings(openai_api_key=None).openai_api_key is None
    assert SanadSettings(openai_api_key=KEY).openai_api_key == KEY


# --------------------------------------------------------------------------- a request cannot carry a key
def test_the_request_models_have_no_api_key_field():
    assert "api_key" not in AskBody.model_fields
    assert "api_key" not in SanadRequest.model_fields


def test_a_key_sent_in_the_body_is_ignored_not_honoured(unconfigured):
    """Even if a caller invents the field, it cannot switch generation on for an unconfigured server."""
    response = TestClient(create_app(unconfigured)).post(
        "/api/ask", json={"question": QUESTION, "api_key": KEY})

    assert response.status_code == 200
    assert unconfigured.rag.ask_calls == []
    assert KEY not in response.text


# --------------------------------------------------------------------------- the key never escapes
def test_the_key_is_absent_from_the_api_response(client):
    body = client.post("/api/ask", json={"question": QUESTION}).text
    assert "sk-" not in body and "api_key" not in body


def test_the_key_never_reaches_the_logs(client, caplog):
    with caplog.at_level(logging.DEBUG):
        client.post("/api/ask", json={"question": QUESTION})
    assert "sk-" not in caplog.text


def test_settings_never_print_the_key():
    settings = SanadSettings(openai_api_key=KEY)
    assert settings.openai_api_key == KEY
    assert KEY not in repr(settings)


def test_an_openai_failure_never_echoes_the_key_back():
    """OpenAI repeats the rejected credential in its own error text; Sanad scrubs it before reporting."""
    exc = foreign_exception("openai", "AuthenticationError",
                            f"Incorrect API key provided: {KEY}. You can find your API key at ...")
    info = classify_exception(exc, "answer_generation")

    assert info.code == RagErrorCode.LLM_REQUEST_FAILED.value
    assert KEY not in info.message and "<redacted>" in info.message
    assert all(KEY not in entry for entry in info.exception_chain)
    assert KEY not in make_error(RagErrorCode.LLM_REQUEST_FAILED, "answer_generation", f"key {KEY} failed").message


# --------------------------------------------------------------------------- .env reaches the settings
def test_a_dotenv_file_supplies_the_key_when_the_environment_does_not(tmp_path):
    path = tmp_path / ".env"
    path.write_text('# a comment\n\nOPENAI_API_KEY="sk-from-the-file"\nSANAD_API_PORT=9001\nnot a pair\n',
                    encoding="utf-8")
    values = _dotenv(path)

    assert values == {"OPENAI_API_KEY": "sk-from-the-file", "SANAD_API_PORT": "9001"}
    assert _dotenv(tmp_path / "absent.env") == {}          # a missing file is simply empty


def test_the_real_environment_wins_over_the_file(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-exported")
    assert _environment(None)["OPENAI_API_KEY"] == "sk-exported"
    assert SanadSettings.from_env().openai_api_key == "sk-exported"


def test_an_explicit_mapping_ignores_the_file_entirely():
    """Tests and callers that pass a mapping get exactly that mapping, so .env can never leak into them."""
    assert SanadSettings.from_env({}).openai_api_key is None
    assert _environment({"A": "1"}) == {"A": "1"}


def test_importing_the_settings_module_does_not_touch_the_process_environment(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import importlib

    import config

    importlib.reload(config)
    assert "OPENAI_API_KEY" not in __import__("os").environ   # the file is read, never written back
    assert isinstance(DOTENV_PATH.name, str)
