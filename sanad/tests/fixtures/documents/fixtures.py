"""Pytest fixtures for Phase 3 documents (imported by the parsers/extraction conftest files)."""

from __future__ import annotations

import pytest

from sanad.config import DocumentProcessingSettings
from sanad.parsers import DocumentProcessor
from tests.fixtures.documents.builders import build_all, write_all


@pytest.fixture(scope="session")
def sample_bytes() -> dict[str, bytes]:
    return build_all()


@pytest.fixture(scope="session")
def sample_paths(tmp_path_factory) -> dict:
    """The synthetic fixtures written to a temporary directory under their documented names."""
    return write_all(tmp_path_factory.mktemp("documents"))


@pytest.fixture(scope="session")
def processor() -> DocumentProcessor:
    return DocumentProcessor(DocumentProcessingSettings())


@pytest.fixture(scope="session")
def parsed(processor, sample_paths) -> dict:
    return {name: processor.parse_path(path) for name, path in sample_paths.items()}
