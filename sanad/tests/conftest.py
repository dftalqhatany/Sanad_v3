"""Shared fixtures. Tests never write into hr_assistant/ (run with PYTHONDONTWRITEBYTECODE=1)."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
LEGACY_DIR = REPO_ROOT / "hr_assistant"
BASELINE_PATH = Path(__file__).parent / "fixtures" / "legacy_rag_baseline.json"
LEGACY_MODULE_NAMES = ("chatbot_backend", "hybird_search")


@pytest.fixture(scope="session")
def legacy_dir() -> Path:
    return LEGACY_DIR


@pytest.fixture(scope="session")
def baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def knowledge_base() -> list[dict]:
    return json.loads((LEGACY_DIR / "data/labor_law/labor_law_parsed.json").read_text(encoding="utf-8"))


def _closed_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def closed_qdrant_url() -> str:
    return f"http://127.0.0.1:{_closed_local_port()}"


@pytest.fixture(scope="session")
def real_qdrant_connection_error(closed_qdrant_url):
    """The genuine exception qdrant-client raises when no Qdrant server is listening."""
    qdrant_client = pytest.importorskip("qdrant_client")
    try:
        client = qdrant_client.QdrantClient(url=closed_qdrant_url, timeout=2, check_compatibility=False)
    except TypeError:
        client = qdrant_client.QdrantClient(url=closed_qdrant_url, timeout=2)
    try:
        client.get_collections()
    except Exception as exc:  # noqa: BLE001 - we want the real exception object
        return exc
    finally:
        client.close()
    pytest.fail(f"expected a connection failure against {closed_qdrant_url}")
