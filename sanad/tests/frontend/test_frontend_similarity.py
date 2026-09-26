"""The interface shows how closely each source matches the question, and never calls that accuracy."""

from __future__ import annotations

import pytest

from frontend import i18n, view
from frontend.client import SanadApiClient

pytest.importorskip("fastapi", reason="the frontend tests exercise the UI against the real API app")


@pytest.fixture
def ui(client) -> SanadApiClient:
    return SanadApiClient(base_url="http://testserver", http_client=client)


def test_the_view_passes_the_similarity_percentage_through(ui):
    response = ui.ask("ما هي مدة فترة التجربة؟")
    answer = view.regulatory_answer(response.result)

    assert answer["evidence"]
    for item in answer["evidence"]:
        assert item["similarity_percentage"] == round(item["score"] * 100)


def test_the_score_is_labelled_similarity_and_never_accuracy():
    assert i18n.t("evidence.similarity", "ar") == "درجة التشابه"
    for lang in ("en", "ar"):
        label = i18n.t("evidence.similarity", lang).lower()
        assert "accuracy" not in label and "confiden" not in label
        assert "دقة" not in label and "صحة" not in label


def test_the_interface_asks_for_no_api_key():
    """The key is the server's: no sidebar input, no session storage, no key wording in the UI."""
    from pathlib import Path

    from frontend import app, pages

    for module in (app, pages):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "openai_key" not in source
        assert "session_state[\"openai_key\"]" not in source
        assert "type=\"password\"" not in source
    assert not [key for key in i18n.STRINGS if key.startswith("key.")]
