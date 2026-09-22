"""Orchestrator: one request in, exactly one existing component called, structured result out."""

from __future__ import annotations

import logging

import pytest

from models.analysis import AnalysisStatus, TargetJob
from models.common import ResultStatus
from models.orchestration import (
    DocumentRole,
    OrchestratorResult,
    Route,
    SanadRequest,
    TaskHint,
    UploadedDocument,
)
from orchestrator import SanadOrchestrator
from orchestrator.routing import route_request
from tests.orchestrator.conftest import CONTRACT_LINES, SpyAnalysisAgent

OFFER_B = [line.replace("12,000", "10,000").replace("30 days", "21 days").replace("Riyadh", "Jeddah")
           for line in CONTRACT_LINES]
OFFER_C = [line.replace("12,000", "15,000").replace("40 hours per week", "48 hours per week") for line in CONTRACT_LINES]


def request_for(*documents, **kwargs) -> SanadRequest:
    return SanadRequest(documents=list(documents), **kwargs)


# --------------------------------------------------------------------------- route 5: regulatory question
def test_a_question_without_documents_goes_to_the_rag_adapter(orchestrator, analysis, comparison, rag):
    result = orchestrator.handle(SanadRequest(question="ما هي مدة الإجازة السنوية؟"))

    assert result.routing.route is Route.REGULATORY_QUESTION and result.routing.agent == "RegulatoryRAGAdapter"
    assert result.routing.rule == "a question and no usable document"
    assert [q.question for q in rag.ask_calls] == ["ما هي مدة الإجازة السنوية؟"]
    assert result.regulatory_answer.answer == "Scripted answer from the existing RAG."
    assert result.regulatory_answer.evidence and result.regulatory_answer.evidence[0].citation
    assert result.status is AnalysisStatus.SUCCESS
    assert analysis.calls == [] and comparison.calls == []  # no unnecessary agent call
    assert result.documents == []
    assert OrchestratorResult.model_validate_json(result.model_dump_json()).model_dump() == result.model_dump()


def test_without_an_api_key_the_question_falls_back_to_retrieval_only(analysis, comparison, rag):
    rag.answer = None
    orchestrator = SanadOrchestrator(analysis, comparison, rag, generate_answers=False)
    result = orchestrator.handle(SanadRequest(question="What is the maximum probation period?"))

    assert result.routing.route is Route.REGULATORY_QUESTION
    assert rag.ask_calls == [] and rag.calls  # retrieval only, no answer generation attempted
    assert result.regulatory_answer.answer is None and "no OPENAI_API_KEY" in result.regulatory_answer.message
    assert result.regulatory_answer.evidence
    assert result.status is AnalysisStatus.SUCCESS


def test_a_failing_rag_is_reported_as_rag_error(orchestrator, rag):
    rag.error_code, rag.answer = "vector_db_unreachable", None
    result = orchestrator.handle(SanadRequest(question="What is the maximum probation period?"))
    assert result.status is AnalysisStatus.RAG_ERROR
    assert result.regulatory_answer.status is ResultStatus.ERROR
    assert [e.code for e in result.errors] == ["routed_agent_failed"]
    assert "could not answer" in result.summary


# --------------------------------------------------------------------------- routes 1 and 2: one contract
def test_one_contract_goes_to_the_analysis_agent(orchestrator, analysis, comparison, upload):
    result = orchestrator.handle(request_for(upload("contract", lines=CONTRACT_LINES)))

    assert result.routing.route is Route.CONTRACT_ANALYSIS and result.routing.agent == "AnalysisAgent"
    assert len(analysis.calls) == 1 and analysis.calls[0]["cv"] is None
    assert comparison.calls == []
    assert result.comparison is None and result.analysis.cv_analysis is None
    assert result.analysis.contract_analysis.finding("probation_period").regulatory_evidence[0].article_number == 53
    assert result.documents[0].role is DocumentRole.CONTRACT and result.documents[0].role_source == "detected"
    assert result.documents[0].extraction_status is ResultStatus.SUCCESS and result.documents[0].usable
    assert result.status is AnalysisStatus.SUCCESS


def test_one_contract_and_a_cv_are_analysed_together_with_job_fit(orchestrator, analysis, upload):
    result = orchestrator.handle(request_for(upload("contract", lines=CONTRACT_LINES), upload("sample_cv_en.docx")))

    assert result.routing.route is Route.CONTRACT_ANALYSIS
    assert len(analysis.calls) == 1 and analysis.calls[0]["use_contract_job_title"] is True
    compatibility = result.analysis.cv_analysis.job_compatibility
    assert compatibility.target_job.title == "Data Analyst" and compatibility.target_job.source == "contract_job_title"
    assert compatibility.requirements[0].cv_evidence[0].source.text
    assert [d.role for d in result.documents] == [DocumentRole.CONTRACT, DocumentRole.CV]
    assert result.status is AnalysisStatus.SUCCESS


def test_an_explicit_target_job_is_used_instead_of_the_contract_title(orchestrator, analysis, upload):
    job = TargetJob(title="Data Analyst", required_skills=["Python"])
    result = orchestrator.handle(request_for(upload("contract", lines=CONTRACT_LINES), upload("sample_cv_en.docx"),
                                             target_job=job))
    assert analysis.calls[0]["target_job"] == job
    assert result.analysis.cv_analysis.job_compatibility.target_job.source == "user_provided"


def test_a_cv_on_its_own_is_analysed_without_a_contract(orchestrator, analysis, comparison, upload, rag):
    result = orchestrator.handle(request_for(upload("sample_cv_en.docx"),
                                             target_job=TargetJob(title="Data Analyst", required_skills=["SQL"])))
    assert result.routing.route is Route.CV_ANALYSIS
    assert result.analysis.contract_analysis is None and result.analysis.cv_analysis.profile.skills
    assert rag.calls == [] and comparison.calls == []  # no regulatory retrieval for a CV
    assert result.status is AnalysisStatus.SUCCESS


# --------------------------------------------------------------------------- routes 3 and 4: several contracts
def test_two_contracts_go_to_the_comparison_agent(orchestrator, analysis, comparison, upload):
    result = orchestrator.handle(request_for(upload("offer_a", lines=CONTRACT_LINES, label="Offer A"),
                                             upload("offer_b", lines=OFFER_B, label="Offer B")))

    assert result.routing.route is Route.CONTRACT_COMPARISON and result.routing.agent == "ContractComparisonAgent"
    assert len(comparison.calls) == 1 and comparison.calls[0]["cv"] is None
    assert comparison.calls[0]["labels"] == ["Offer A", "Offer B"]
    assert len(analysis.calls) == 2  # Phase 5 behaviour preserved: one analysis per contract
    assert [c.label for c in result.comparison.contracts] == ["Offer A", "Offer B"]
    assert result.comparison.recommendation.preferred_contract_id == "contract_1"
    assert result.analysis is None and result.status is AnalysisStatus.SUCCESS


def test_three_contracts_and_a_cv_go_to_the_comparison_agent(orchestrator, analysis, comparison, upload):
    result = orchestrator.handle(request_for(upload("offer_a", lines=CONTRACT_LINES), upload("offer_b", lines=OFFER_B),
                                             upload("offer_c", lines=OFFER_C), upload("sample_cv_en.docx"),
                                             priorities=["basic_salary"]))

    assert result.routing.route is Route.CONTRACT_COMPARISON and result.routing.contract_count == 3
    assert len(comparison.calls) == 1 and comparison.calls[0]["priorities"] == ["basic_salary"]
    assert len(analysis.calls) == 3 and all(call["use_contract_job_title"] for call in analysis.calls)
    assert len(result.comparison.contracts) == 3
    assert result.comparison.recommendation.preferred_contract_id == "contract_3"  # highest basic salary
    assert result.comparison.dimension("cv_compatibility").values
    assert result.comparison.cv_document_id == result.documents[3].document_id
    assert result.status is AnalysisStatus.SUCCESS


def test_comparison_evidence_survives_the_orchestrator(orchestrator, upload):
    result = orchestrator.handle(request_for(upload("offer_a", lines=CONTRACT_LINES), upload("offer_b", lines=OFFER_B)))
    factor = next(f for f in result.comparison.recommendation.decision_factors if f.dimension == "basic_salary")
    value = factor.evidence[0]
    assert value.finding_ids and value.raw_values == ["12,000 SAR"] and value.source_texts
    assert result.comparison.risks and result.comparison.recommendation.caveats
    assert result.comparison.contracts[0].contract_analysis.evidence[0].arabic_text


# --------------------------------------------------------------------------- route 6: salary benchmarking
def test_salary_benchmark_request_uses_the_salary_interface_only(orchestrator, analysis, comparison, rag, upload):
    result = orchestrator.handle(request_for(upload("contract", lines=CONTRACT_LINES), task=TaskHint.SALARY_BENCHMARK))

    assert result.routing.route is Route.SALARY_BENCHMARK and result.routing.agent == "SalaryBenchmarkProvider"
    assert len(analysis.salary_calls) == 1 and analysis.calls == []  # no regulatory analysis was run
    assert comparison.calls == [] and rag.calls == []
    assert result.salary_benchmark.status == "not_configured"
    assert result.salary_benchmark.contract_salary.amount == 12000 and result.salary_benchmark.market_min is None
    assert result.salary_contract_id == result.documents[0].document_id
    assert result.status is AnalysisStatus.INSUFFICIENT_EVIDENCE


def test_salary_benchmark_without_a_contract_is_refused(orchestrator, analysis, upload):
    result = orchestrator.handle(request_for(upload("sample_cv_en.docx"), task=TaskHint.SALARY_BENCHMARK))
    assert result.status is AnalysisStatus.INVALID_INPUT and result.routing.route is Route.NONE
    assert [e.code for e in result.errors] == ["no_contract_uploaded"]
    assert analysis.salary_calls == [] and analysis.calls == []


# --------------------------------------------------------------------------- task hints
@pytest.mark.parametrize("task, route", [
    (TaskHint.CONTRACT_ANALYSIS, Route.CONTRACT_ANALYSIS),
    (TaskHint.AUTO, Route.CONTRACT_ANALYSIS),
])
def test_task_hint_is_respected(orchestrator, upload, task, route):
    result = orchestrator.handle(request_for(upload("contract", lines=CONTRACT_LINES), task=task))
    assert result.routing.route is route and result.routing.task_hint is task


def test_a_question_asked_together_with_documents_is_not_silently_dropped(orchestrator, upload, rag):
    result = orchestrator.handle(request_for(upload("contract", lines=CONTRACT_LINES), question="Is this legal?"))
    assert result.routing.route is Route.CONTRACT_ANALYSIS
    assert rag.ask_calls == []
    assert any("question was not answered here" in w for w in result.warnings)


def test_a_regulatory_question_hint_ignores_uploads(orchestrator, analysis, upload, rag):
    result = orchestrator.handle(request_for(upload("contract", lines=CONTRACT_LINES), question="Probation limit?",
                                             task=TaskHint.REGULATORY_QUESTION))
    assert result.routing.route is Route.REGULATORY_QUESTION and analysis.calls == []
    assert len(rag.ask_calls) == 1
    assert any("were not used" in w for w in result.warnings)


def test_a_comparison_hint_with_one_contract_is_refused(orchestrator, comparison, upload):
    result = orchestrator.handle(request_for(upload("contract", lines=CONTRACT_LINES),
                                             task=TaskHint.CONTRACT_COMPARISON))
    assert result.status is AnalysisStatus.INVALID_INPUT and [e.code for e in result.errors] == ["task_not_possible"]
    assert comparison.calls == []


def test_a_contract_analysis_hint_with_two_contracts_is_refused(orchestrator, analysis, upload):
    result = orchestrator.handle(request_for(upload("offer_a", lines=CONTRACT_LINES), upload("offer_b", lines=OFFER_B),
                                             task=TaskHint.CONTRACT_ANALYSIS))
    assert result.status is AnalysisStatus.INVALID_INPUT and analysis.calls == []
    assert "ask for a comparison" in result.errors[0].message


# --------------------------------------------------------------------------- invalid input and failures
def test_an_unsupported_file_type_is_explicit(orchestrator, analysis, upload):
    image = UploadedDocument(filename="scan.png", content=b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    result = orchestrator.handle(request_for(image))
    assert result.status is AnalysisStatus.INVALID_INPUT and result.routing.route is Route.NONE
    assert result.documents[0].document_status.value == "unsupported_file_type"
    assert "PDF and DOCX" in result.documents[0].notes[0]
    assert {e.code for e in result.errors} >= {"nothing_to_do", "unusable_document"}
    assert analysis.calls == []


def test_a_missing_file_is_explicit(orchestrator, tmp_path, analysis):
    result = orchestrator.handle(SanadRequest(documents=[UploadedDocument(path=str(tmp_path / "missing.pdf"))]))
    assert result.status is AnalysisStatus.INVALID_INPUT
    assert result.documents[0].document_status.value == "file_not_found"
    assert analysis.calls == []


def test_a_scanned_pdf_is_reported_as_needing_ocr(orchestrator, upload):
    result = orchestrator.handle(request_for(upload("sample_scanned.pdf")))
    assert result.status is AnalysisStatus.INVALID_INPUT
    assert result.documents[0].document_status.value == "ocr_required"
    assert "does not perform OCR" in result.documents[0].notes[0]


def test_too_many_contracts_are_refused_before_anything_is_parsed(orchestrator, analysis, upload):
    uploads = [upload(f"offer_{i}", lines=CONTRACT_LINES, role=DocumentRole.CONTRACT) for i in range(6)]
    result = orchestrator.handle(request_for(*uploads))
    assert result.status is AnalysisStatus.INVALID_INPUT and [e.code for e in result.errors] == ["too_many_contracts"]
    assert result.documents == [] and analysis.calls == []


def test_an_empty_request_is_refused(orchestrator):
    result = orchestrator.handle(SanadRequest())
    assert result.status is AnalysisStatus.INVALID_INPUT and result.routing.route is Route.NONE
    assert [e.code for e in result.errors] == ["nothing_to_do"]


def test_an_invalid_request_object_is_refused(orchestrator):
    result = orchestrator.handle({"documents": [{"filename": "a.pdf"}]})  # neither content nor path
    assert result.status is AnalysisStatus.INVALID_INPUT and result.routing.route is Route.NONE


def test_a_request_can_be_given_as_a_dictionary(orchestrator, sample_bytes):
    result = orchestrator.handle({"documents": [{"filename": "contract.docx",
                                                 "content": sample_bytes["sample_contract_en.docx"],
                                                 "role": "contract"}]})
    assert result.routing.route is Route.CONTRACT_ANALYSIS and result.status is AnalysisStatus.SUCCESS


def test_partial_failure_still_compares_the_readable_contracts(orchestrator, analysis, comparison, upload):
    result = orchestrator.handle(request_for(upload("offer_a", lines=CONTRACT_LINES), upload("offer_b", lines=OFFER_B),
                                             upload("sample_scanned.pdf")))
    assert result.routing.route is Route.CONTRACT_COMPARISON
    assert len(comparison.calls[0]["contracts"]) == 2 and len(analysis.calls) == 2
    assert result.status is AnalysisStatus.PARTIAL
    assert [e.code for e in result.errors] == ["unusable_document"]
    assert any("could not be used" in w for w in result.warnings)
    assert result.comparison.recommendation.status == "preferred_contract"


def test_one_unusable_contract_falls_back_to_a_single_analysis(orchestrator, analysis, comparison, upload):
    result = orchestrator.handle(request_for(upload("offer_a", lines=CONTRACT_LINES), upload("sample_empty.pdf")))
    assert result.routing.route is Route.CONTRACT_ANALYSIS
    assert result.routing.fallback_from is Route.CONTRACT_COMPARISON
    assert comparison.calls == [] and len(analysis.calls) == 1
    assert result.status is AnalysisStatus.PARTIAL
    assert any("analysed on its own instead of compared" in w for w in result.warnings)


def test_a_failing_agent_is_reported_not_hidden(orchestrator, analysis, upload, monkeypatch):
    monkeypatch.setattr(SpyAnalysisAgent, "analyze", lambda self, **kwargs: 1 / 0)
    result = orchestrator.handle(request_for(upload("contract", lines=CONTRACT_LINES)))
    assert result.status is AnalysisStatus.ANALYSIS_ERROR
    assert [e.code for e in result.errors] == ["orchestration_failed"]
    assert result.errors[0].exception_type == "ZeroDivisionError"
    assert result.analysis is None and "could not be completed" in result.summary


def test_an_agent_returning_a_failure_status_is_propagated(orchestrator, rag, upload):
    rag.error_code = "vector_db_unreachable"
    result = orchestrator.handle(request_for(upload("contract", lines=CONTRACT_LINES)))
    assert result.status is AnalysisStatus.RAG_ERROR
    assert result.analysis.contract_analysis.status is AnalysisStatus.RAG_ERROR
    assert [e.code for e in result.errors] == ["routed_agent_failed"]


def test_an_undetectable_document_asks_for_an_explicit_role(orchestrator, upload):
    result = orchestrator.handle(request_for(upload("notes", lines=["Meeting notes", "Nothing useful here."])))
    assert result.status is AnalysisStatus.INVALID_INPUT
    assert result.documents[0].role is None and result.documents[0].role_source == "undetermined"
    assert any("explicit role" in e.message for e in result.errors)


def test_declared_roles_are_not_second_guessed(orchestrator, analysis, upload):
    result = orchestrator.handle(request_for(upload("contract", lines=CONTRACT_LINES, role=DocumentRole.CONTRACT),
                                             upload("sample_cv_en.docx", role=DocumentRole.CV)))
    assert [d.role_source for d in result.documents] == ["declared", "declared"]
    assert result.routing.route is Route.CONTRACT_ANALYSIS and len(analysis.calls) == 1


def test_only_the_first_cv_is_used(orchestrator, analysis, upload):
    result = orchestrator.handle(request_for(upload("contract", lines=CONTRACT_LINES),
                                             upload("sample_cv_en.docx"), upload("sample_cv_ar.docx")))
    assert any("only the first one is used" in w for w in result.warnings)
    assert len(analysis.calls) == 1


def test_logs_contain_no_document_content(orchestrator, upload, caplog):
    with caplog.at_level(logging.DEBUG, logger="sanad"):
        orchestrator.handle(request_for(upload("contract", lines=CONTRACT_LINES)))
    assert "12,000" not in caplog.text and "Data Analyst" not in caplog.text


# --------------------------------------------------------------------------- the routing table on its own
@pytest.mark.parametrize("task, counts, expected", [
    (TaskHint.AUTO, dict(has_question=True, contracts=0, cvs=0), Route.REGULATORY_QUESTION),
    (TaskHint.AUTO, dict(has_question=False, contracts=1, cvs=0), Route.CONTRACT_ANALYSIS),
    (TaskHint.AUTO, dict(has_question=False, contracts=1, cvs=1), Route.CONTRACT_ANALYSIS),
    (TaskHint.AUTO, dict(has_question=False, contracts=2, cvs=0), Route.CONTRACT_COMPARISON),
    (TaskHint.AUTO, dict(has_question=False, contracts=5, cvs=1), Route.CONTRACT_COMPARISON),
    (TaskHint.AUTO, dict(has_question=False, contracts=0, cvs=1), Route.CV_ANALYSIS),
    (TaskHint.AUTO, dict(has_question=True, contracts=1, cvs=0), Route.CONTRACT_ANALYSIS),
    (TaskHint.AUTO, dict(has_question=False, contracts=0, cvs=0), Route.NONE),
    (TaskHint.AUTO, dict(has_question=False, contracts=6, cvs=0), Route.NONE),
    (TaskHint.SALARY_BENCHMARK, dict(has_question=False, contracts=1, cvs=0), Route.SALARY_BENCHMARK),
    (TaskHint.REGULATORY_QUESTION, dict(has_question=True, contracts=2, cvs=0), Route.REGULATORY_QUESTION),
    (TaskHint.REGULATORY_QUESTION, dict(has_question=False, contracts=0, cvs=0), Route.NONE),
    (TaskHint.CV_ANALYSIS, dict(has_question=False, contracts=1, cvs=1), Route.CV_ANALYSIS),
])
def test_routing_table(task, counts, expected):
    outcome = route_request(task, **counts)
    assert outcome.decision.route is expected
    assert (outcome.errors == []) is (expected is not Route.NONE)
    assert outcome.decision.rule and outcome.decision.reason


def test_routing_is_a_pure_function_of_the_counts():
    outcome = route_request(TaskHint.CONTRACT_COMPARISON, has_question=False, contracts=1, cvs=0, unusable=1)
    assert outcome.decision.route is Route.CONTRACT_ANALYSIS and outcome.decision.fallback_from is Route.CONTRACT_COMPARISON
    assert route_request(TaskHint.CONTRACT_COMPARISON, has_question=False, contracts=1, cvs=0,
                         unusable=0).decision.route is Route.NONE
