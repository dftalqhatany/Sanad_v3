"""Comparison Agent -> Analysis Agent -> RegulatoryRAGAdapter -> REAL, unmodified hr_assistant RAG (infra simulated).

Tested with 2 and 3 contracts: every contract gets its own regulatory analysis through the existing retriever.
"""

from __future__ import annotations

import pytest

from sanad.agents import AnalysisAgent, ContractAnalysisAgent, ContractComparisonAgent
from sanad.extraction import extract_contract, extract_cv
from sanad.models.analysis import AnalysisStatus
from tests.fixtures.documents.builders import docx_from_paragraphs
from tests.fixtures.documents.fixtures import parsed, processor, sample_bytes, sample_paths  # noqa: F401

PROBATION_ARTICLE_53 = 54
ANNUAL_LEAVE_ARTICLE_109 = 111
SHORTER_OFFER = ["Job Title: Data Analyst", "Basic Salary: 10,000 SAR per month", "Working Hours: 8 hours per day, 48 hours per week",
                 "Annual Leave: 21 days", "Probation Period: 180 days", "Notice Period: 30 days"]


@pytest.mark.parametrize("extra", [0, 1])
def test_two_and_three_contracts_run_through_the_existing_rag(adapter, infra, parsed, processor, extra):
    infra.dense_hits = [(PROBATION_ARTICLE_53, 0.91), (ANNUAL_LEAVE_ARTICLE_109, 0.9)]
    contracts = [extract_contract(parsed["sample_contract_en.docx"]),
                 extract_contract(processor.parse_bytes(docx_from_paragraphs(SHORTER_OFFER), "offer_b.docx"))]
    if extra:
        contracts.append(extract_contract(parsed["sample_contract_ar.docx"]))
    cv = extract_cv(parsed["sample_cv_en.docx"])

    result = ContractComparisonAgent(AnalysisAgent(ContractAnalysisAgent(adapter))).compare(contracts, cv)

    assert result.status is AnalysisStatus.SUCCESS
    assert len(result.contracts) == len(contracts)
    questions = [sum(len(c.questions) for c in entry.contract_analysis.regulatory_checks) for entry in result.contracts]
    assert len(infra.dense_queries) == sum(questions)  # each contract analysed individually through the existing RAG
    assert set(infra.collections) == {"saudi_labor_law"} and infra.llm_calls == []
    for entry in result.contracts:
        articles = {e.article_number for e in entry.contract_analysis.finding("probation_period").regulatory_evidence}
        assert 53 in articles
    assert result.dimension("basic_salary").best_contract_ids == ["contract_1"]
    assert result.dimension("probation_period").best_contract_ids[0] == "contract_1"
    assert result.recommendation.status in ("preferred_contract", "no_clear_preference")
    for factor in result.recommendation.decision_factors:
        assert all(v.source_texts or v.fields_used for v in factor.evidence)


def test_qdrant_down_during_comparison_is_explicit(adapter, infra, parsed, processor, real_qdrant_connection_error):
    infra.vector_store_init_error = real_qdrant_connection_error
    contracts = [extract_contract(parsed["sample_contract_en.docx"]),
                 extract_contract(processor.parse_bytes(docx_from_paragraphs(SHORTER_OFFER), "offer_b.docx"))]
    result = ContractComparisonAgent(AnalysisAgent(ContractAnalysisAgent(adapter))).compare(contracts)
    assert result.status is AnalysisStatus.RAG_ERROR
    assert all(c.contract_analysis.errors[0].code == "vector_db_unreachable" for c in result.contracts)
    assert not result.dimension("compliance").comparable
