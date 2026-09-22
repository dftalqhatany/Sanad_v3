from __future__ import annotations

import io

import pytest

from extraction import extract_contract, extract_cv
from tests.fakes.regulatory_adapter import FakeRegulatoryAdapter
from tests.fixtures.documents.builders import docx_from_paragraphs
from tests.fixtures.documents.fixtures import parsed, processor, sample_bytes, sample_paths  # noqa: F401


@pytest.fixture
def fake_rag(knowledge_base) -> FakeRegulatoryAdapter:
    return FakeRegulatoryAdapter(knowledge_base)


@pytest.fixture(scope="session")
def contract_en(parsed):
    return extract_contract(parsed["sample_contract_en.docx"])


@pytest.fixture(scope="session")
def contract_ar(parsed):
    return extract_contract(parsed["sample_contract_ar.docx"])


@pytest.fixture(scope="session")
def cv_en(parsed):
    return extract_cv(parsed["sample_cv_en.docx"])


@pytest.fixture(scope="session")
def cv_ar(parsed):
    return extract_cv(parsed["sample_cv_ar.docx"])


@pytest.fixture
def make_contract(processor):
    def build(lines: list[str]):
        document = processor.parse_bytes(docx_from_paragraphs(lines), "contract.docx")
        return document, extract_contract(document)
    return build


@pytest.fixture
def make_cv(processor):
    """build(docx.Document) adds headings/paragraphs; returns (ParsedDocument, CvExtraction)."""
    def make(build):
        from docx import Document

        document = Document()
        build(document)
        buffer = io.BytesIO()
        document.save(buffer)
        parsed_document = processor.parse_bytes(buffer.getvalue(), "cv.docx")
        return parsed_document, extract_cv(parsed_document)
    return make
