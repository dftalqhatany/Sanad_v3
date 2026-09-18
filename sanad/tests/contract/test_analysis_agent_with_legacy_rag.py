"""Contract Analysis Agent -> RegulatoryRAGAdapter -> REAL, unmodified hr_assistant RAG code and knowledge base.

Only Qdrant, the e5 embedding model and OpenAI are simulated (tests/fakes/legacy_stubs.py). The agent's
questions go through the real HybridRetriever (real BM25 over the real knowledge base).
"""

from __future__ import annotations

import sys

from sanad.agents import ContractAnalysisAgent, LLMEvidenceInterpreter
from sanad.extraction import extract_contract
from sanad.models.analysis import AnalysisStatus, FindingStatus
from tests.fakes.llm import FakeLLMClient
from tests.fixtures.documents.fixtures import parsed, processor, sample_bytes, sample_paths  # noqa: F401

PROBATION_ARTICLE_53 = 54  # kb index
ANNUAL_LEAVE_ARTICLE_109 = 111


def test_contract_analysis_runs_through_the_existing_rag(adapter, infra, parsed, knowledge_base):
    infra.dense_hits = [(PROBATION_ARTICLE_53, 0.91), (ANNUAL_LEAVE_ARTICLE_109, 0.9)]
    contract = extract_contract(parsed["sample_contract_en.docx"])
    llm = FakeLLMClient()

    result = ContractAnalysisAgent(adapter, LLMEvidenceInterpreter(llm)).analyze(contract)

    assert result.status is AnalysisStatus.SUCCESS
    assert adapter.is_backend_loaded and "hybird_search" in sys.modules
    assert set(infra.collections) == {"saudi_labor_law"}  # the existing collection, read only
    assert infra.dense_queries  # every agent question reached the existing dense retriever
    assert len(infra.dense_queries) == sum(len(check.questions) for check in result.regulatory_checks)
    assert infra.llm_calls == []  # retrieve_evidence never calls the legacy answer LLM

    probation = result.finding("probation_period")
    assert probation.status is FindingStatus.REQUIRES_REVIEW
    [article] = [e for e in probation.regulatory_evidence if e.article_number == 53]
    assert article.arabic_text == knowledge_base[PROBATION_ARTICLE_53 - 1]["arabic_content"]
    assert article.evidence.raw_metadata == knowledge_base[PROBATION_ARTICLE_53 - 1]
    assert article.evidence.source.vector_collection == "saudi_labor_law"
    assert article.score > 0 and article.rank >= 1

    leave = result.finding("annual_leave")
    assert 109 in [e.article_number for e in leave.regulatory_evidence]
    assert "probation_period" in llm.fields_called() and "annual_leave" in llm.fields_called()
    assert all(e.evidence.legacy_reference is None and not e.evidence.score_is_rounded for e in result.evidence)


def test_qdrant_unreachable_makes_the_contract_analysis_a_rag_error(adapter, infra, parsed,
                                                                    real_qdrant_connection_error):
    infra.vector_store_init_error = real_qdrant_connection_error
    contract = extract_contract(parsed["sample_contract_en.docx"])

    result = ContractAnalysisAgent(adapter).analyze(contract)

    assert result.status is AnalysisStatus.RAG_ERROR
    assert [(e.code, e.stage) for e in result.errors] == [("vector_db_unreachable", "initialization")]
    assert result.evidence == []
    assert not {f.status for f in result.findings} & {FindingStatus.COMPLIANT, FindingStatus.NON_COMPLIANT}
    assert all(f.status in (FindingStatus.ERROR, FindingStatus.NOT_FOUND) for f in result.findings if f.topic)
