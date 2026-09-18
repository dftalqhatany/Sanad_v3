"""End-to-end through the Orchestrator onto the REAL, unmodified hr_assistant RAG (infrastructure simulated).

Uploaded bytes -> existing parsers -> existing extraction -> agents -> RegulatoryRAGAdapter -> legacy retriever.
"""

from __future__ import annotations

from sanad.agents import AnalysisAgent, ContractAnalysisAgent, ContractComparisonAgent
from sanad.models.analysis import AnalysisStatus
from sanad.models.common import ResultStatus
from sanad.models.orchestration import DocumentRole, Route, SanadRequest, TaskHint, UploadedDocument
from sanad.orchestrator import SanadOrchestrator
from tests.fixtures.documents.builders import build_all, docx_from_paragraphs

PROBATION_ARTICLE_53 = 54
ANNUAL_LEAVE_ARTICLE_109 = 111
SHORTER_OFFER = ["Job Title: Data Analyst", "Basic Salary: 10,000 SAR per month",
                 "Working Hours: 8 hours per day, 48 hours per week", "Annual Leave: 21 days",
                 "Probation Period: 180 days", "Notice Period: 30 days"]


def build(adapter, api_key: str | None = None) -> SanadOrchestrator:
    analysis = AnalysisAgent(ContractAnalysisAgent(adapter))
    return SanadOrchestrator(analysis, ContractComparisonAgent(analysis), adapter,
                             generate_answers=api_key is not None)


def upload(name: str, data: bytes, role: DocumentRole = DocumentRole.AUTO) -> UploadedDocument:
    return UploadedDocument(filename=name, content=data, role=role)


def test_question_is_answered_by_the_existing_answer_pipeline(adapter, infra):
    infra.dense_hits = [(ANNUAL_LEAVE_ARTICLE_109, 0.9)]
    infra.llm_answer = "الإجازة السنوية لا تقل عن واحد وعشرين يوماً."

    result = build(adapter, api_key="configured").handle(SanadRequest(question="ما مدة الإجازة السنوية؟"))

    assert result.routing.route is Route.REGULATORY_QUESTION
    assert result.status is AnalysisStatus.SUCCESS
    assert result.regulatory_answer.answer == infra.llm_answer
    assert 109 in [item.reference.article_number for item in result.regulatory_answer.evidence]
    assert len(infra.llm_calls) == 1 and set(infra.collections) == {"saudi_labor_law"}


def test_question_without_an_api_key_returns_retrieved_evidence_only(adapter, infra):
    infra.dense_hits = [(ANNUAL_LEAVE_ARTICLE_109, 0.9)]
    result = build(adapter).handle(SanadRequest(question="ما مدة الإجازة السنوية؟"))

    assert result.status is AnalysisStatus.SUCCESS and result.regulatory_answer.answer is None
    assert result.regulatory_answer.evidence and infra.llm_calls == []


def test_uploaded_contract_and_cv_run_end_to_end(adapter, infra):
    infra.dense_hits = [(PROBATION_ARTICLE_53, 0.91), (ANNUAL_LEAVE_ARTICLE_109, 0.9)]
    documents = build_all()
    request = SanadRequest(documents=[upload("contract.docx", documents["sample_contract_en.docx"]),
                                      upload("cv.docx", documents["sample_cv_en.docx"])])

    result = build(adapter).handle(request)

    assert result.routing.route is Route.CONTRACT_ANALYSIS and result.status is AnalysisStatus.SUCCESS
    assert [d.role for d in result.documents] == [DocumentRole.CONTRACT, DocumentRole.CV]
    probation = result.analysis.contract_analysis.finding("probation_period")
    assert 53 in [e.article_number for e in probation.regulatory_evidence]
    assert probation.contract_fact.source_text and probation.contract_fact.raw_value == "ninety (90) days"
    assert result.analysis.cv_analysis.job_compatibility.target_job.title == "Data Analyst"
    assert infra.dense_queries and infra.llm_calls == []


def test_uploaded_contracts_are_compared_end_to_end(adapter, infra):
    infra.dense_hits = [(PROBATION_ARTICLE_53, 0.91), (ANNUAL_LEAVE_ARTICLE_109, 0.9)]
    documents = build_all()
    request = SanadRequest(documents=[upload("offer_a.docx", documents["sample_contract_en.docx"]),
                                      upload("offer_b.docx", docx_from_paragraphs(SHORTER_OFFER)),
                                      upload("cv.docx", documents["sample_cv_en.docx"])])

    result = build(adapter).handle(request)

    assert result.routing.route is Route.CONTRACT_COMPARISON and result.status is AnalysisStatus.SUCCESS
    assert len(result.comparison.contracts) == 2
    per_contract = [sum(len(c.questions) for c in entry.contract_analysis.regulatory_checks)
                    for entry in result.comparison.contracts]
    assert len(infra.dense_queries) == sum(per_contract)  # one analysis per contract, no repeats
    assert result.comparison.recommendation.preferred_contract_id == "contract_1"
    assert result.comparison.contracts[0].cv_analysis.job_compatibility is not None


def test_salary_benchmarking_never_touches_the_rag(adapter, infra):
    documents = build_all()
    result = build(adapter).handle(SanadRequest(documents=[upload("contract.docx", documents["sample_contract_en.docx"])],
                                                task=TaskHint.SALARY_BENCHMARK))
    assert result.routing.route is Route.SALARY_BENCHMARK
    assert result.salary_benchmark.status == "not_configured" and result.salary_benchmark.sources == []
    assert infra.dense_queries == [] and infra.llm_calls == []
    assert not adapter.is_backend_loaded


def test_qdrant_down_is_reported_through_the_orchestrator(adapter, infra, real_qdrant_connection_error):
    infra.vector_store_init_error = real_qdrant_connection_error
    documents = build_all()
    result = build(adapter).handle(SanadRequest(documents=[upload("contract.docx", documents["sample_contract_en.docx"])]))

    assert result.status is AnalysisStatus.RAG_ERROR
    assert result.analysis.contract_analysis.errors[0].code == "vector_db_unreachable"

    answer = build(adapter).handle(SanadRequest(question="ما مدة الإجازة السنوية؟"))
    assert answer.status is AnalysisStatus.RAG_ERROR
    assert answer.regulatory_answer.status is ResultStatus.ERROR
