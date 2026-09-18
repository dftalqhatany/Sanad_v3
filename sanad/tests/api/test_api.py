"""The HTTP API: edge validation, one call to SanadOrchestrator.handle(), structured JSON back."""

from __future__ import annotations

import json

import pytest

from sanad.api.schemas import HTTP_STATUS
from sanad.models.analysis import AnalysisStatus
from sanad.models.orchestration import OrchestratorResult, SanadRequest
from tests.api.conftest import DOCX_TYPE, StubOrchestrator, contract_bytes

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sanad.api import create_app  # noqa: E402


def files_for(*names: str, data: dict[str, bytes] | None = None) -> list:
    data = data or {}
    return [("files", (name, data.get(name, contract_bytes()), DOCX_TYPE)) for name in names]


def analyze(client: TestClient, files=None, **form):
    payload = {key: value for key, value in form.items() if value is not None}
    return client.post("/api/analyze", files=files, data=payload)


# --------------------------------------------------------------------------- service endpoints
def test_health_and_config_do_no_work(client, real_orchestrator):
    health = client.get("/api/health")
    assert health.status_code == 200 and health.json()["entry_point"] == "SanadOrchestrator.handle"

    config = client.get("/api/config").json()
    assert config["allowed_file_types"] == [".pdf", ".docx"] and config["max_contracts"] == 5
    assert config["max_file_size_mb"] == 20.0 and config["max_files"] == 6
    assert "contract_comparison" in config["tasks"] and "basic_salary" in config["comparison_priorities"]
    assert config["answer_generation_enabled"] is False
    assert real_orchestrator.rag.calls == []  # nothing was analysed to answer these


# --------------------------------------------------------------------------- routed requests
def test_one_contract_is_analysed(client):
    response = analyze(client, files_for("contract.docx"))
    body = response.json()
    assert response.status_code == 200 and body["status"] == "success"
    assert body["routing"]["route"] == "contract_analysis" and body["routing"]["agent"] == "AnalysisAgent"
    finding = next(f for f in body["analysis"]["contract_analysis"]["findings"] if f["field"] == "probation_period")
    assert finding["regulatory_evidence"][0]["article_number"] == 53
    assert finding["contract_fact"]["source_text"] and finding["finding_id"]
    assert OrchestratorResult.model_validate(body).status is AnalysisStatus.SUCCESS  # the documented schema


def test_one_contract_and_a_cv_are_analysed_together(client, sample_bytes):
    files = files_for("contract.docx") + [("files", ("cv.docx", sample_bytes["sample_cv_en.docx"], DOCX_TYPE))]
    body = analyze(client, files, roles=["contract", "cv"]).json()
    assert body["routing"]["route"] == "contract_analysis"
    compatibility = body["analysis"]["cv_analysis"]["job_compatibility"]
    assert compatibility["target_job"]["title"] == "Data Analyst"
    assert [d["role"] for d in body["documents"]] == ["contract", "cv"]


def test_two_contracts_are_compared(client):
    files = files_for("offer_a.docx") + [("files", ("offer_b.docx", contract_bytes(("12,000", "9,000")), DOCX_TYPE))]
    body = analyze(client, files, labels=["Offer A", "Offer B"]).json()
    assert body["routing"]["route"] == "contract_comparison" and body["comparison"] is not None
    assert [c["label"] for c in body["comparison"]["contracts"]] == ["Offer A", "Offer B"]
    assert body["comparison"]["recommendation"]["preferred_contract_id"] == "contract_1"
    salary = next(d for d in body["comparison"]["comparison"]["salary"] if d["dimension"] == "basic_salary")
    assert [v["display"] for v in salary["values"]] == ["12,000 SAR/month", "9,000 SAR/month"]
    assert body["comparison"]["risks"] and body["comparison"]["recommendation"]["caveats"]


def test_three_contracts_and_a_cv_are_compared(client, sample_bytes):
    files = (files_for("offer_a.docx")
             + [("files", ("offer_b.docx", contract_bytes(("12,000", "9,000")), DOCX_TYPE)),
                ("files", ("offer_c.docx", contract_bytes(("30 days", "21 days")), DOCX_TYPE)),
                ("files", ("cv.docx", sample_bytes["sample_cv_en.docx"], DOCX_TYPE))])
    body = analyze(client, files, priorities=["basic_salary"]).json()
    assert body["routing"]["route"] == "contract_comparison" and body["routing"]["contract_count"] == 3
    assert len(body["comparison"]["contracts"]) == 3
    assert body["comparison"]["recommendation"]["method"] == "priorities"
    assert body["comparison"]["cv_document_id"] == body["documents"][3]["document_id"]


def test_a_regulatory_question_reaches_the_rag_adapter(client, real_orchestrator):
    body = client.post("/api/ask", json={"question": "ما هي مدة الإجازة السنوية؟"}).json()
    assert body["routing"]["route"] == "regulatory_question" and body["routing"]["agent"] == "RegulatoryRAGAdapter"
    assert body["regulatory_answer"]["answer"] == "Scripted answer from the existing RAG."
    assert body["regulatory_answer"]["evidence"][0]["citation"]
    assert [q.question for q in real_orchestrator.rag.ask_calls] == ["ما هي مدة الإجازة السنوية؟"]

    form_body = analyze(client, None, question="What is the maximum probation period?").json()
    assert form_body["routing"]["route"] == "regulatory_question"


def test_salary_benchmarking_is_routed_without_regulatory_work(client, real_orchestrator):
    body = analyze(client, files_for("contract.docx"), task="salary_benchmark").json()
    assert body["routing"]["route"] == "salary_benchmark"
    assert body["salary_benchmark"]["status"] == "not_configured" and body["salary_benchmark"]["sources"] == []
    assert real_orchestrator.rag.calls == []


def test_target_job_and_priorities_reach_the_orchestrator(stub, stub_client):
    job = {"title": "Data Analyst", "required_skills": ["Python"]}
    analyze(stub_client, files_for("contract.docx"), target_job=json.dumps(job), priorities=["annual_leave"],
            task="contract_analysis")
    [request] = stub.calls
    assert isinstance(request, SanadRequest)
    assert request.target_job.title == "Data Analyst" and request.target_job.required_skills == ["Python"]
    assert request.priorities == ["annual_leave"] and request.task.value == "contract_analysis"


# --------------------------------------------------------------------------- edge validation
@pytest.mark.parametrize("filename, content_type, code", [
    ("scan.png", "image/png", "unsupported_file_type"),
    ("notes.txt", "text/plain", "unsupported_file_type"),
    ("contract", "application/octet-stream", "unsupported_file_type"),
])
def test_unsupported_file_types_are_rejected(stub, stub_client, filename, content_type, code):
    response = stub_client.post("/api/analyze", files=[("files", (filename, b"data" * 10, content_type))])
    assert response.status_code == 415 and response.json()["error"]["code"] == code
    assert stub.calls == []  # nothing reached the Orchestrator


def test_oversized_files_are_rejected(stub, stub_client):
    oversized = b"%PDF-1.4" + b"0" * (21 * 1024 * 1024)
    response = stub_client.post("/api/analyze", files=[("files", ("big.pdf", oversized, "application/pdf"))])
    assert response.status_code == 413 and response.json()["error"]["code"] == "file_too_large"
    assert "20 MB" in response.json()["error"]["message"] and stub.calls == []


def test_too_many_files_are_rejected(stub, stub_client):
    response = stub_client.post("/api/analyze", files=files_for(*[f"c{i}.docx" for i in range(7)]))
    assert response.status_code == 413 and response.json()["error"]["code"] == "too_many_files"
    assert stub.calls == []


@pytest.mark.parametrize("files, form, status, code", [
    (None, {}, 400, "nothing_to_do"),
    (None, {"question": "   "}, 400, "nothing_to_do"),
    ("one", {"roles": ["contract", "cv"]}, 400, "invalid_request"),
    ("one", {"roles": ["employer"]}, 400, "invalid_request"),
    ("one", {"target_job": "{not json"}, 400, "invalid_request"),
    ("one", {"target_job": json.dumps({"required_skills": ["x"]})}, 400, "invalid_request"),
    ("one", {"task": "make_coffee"}, 400, "invalid_request"),
    ("one", {"labels": ["a", "b"]}, 400, "invalid_request"),
])
def test_invalid_requests_are_refused_with_clear_errors(stub, stub_client, files, form, status, code):
    response = analyze(stub_client, files_for("contract.docx") if files else None, **form)
    assert response.status_code == status
    if code:
        assert response.json()["error"]["code"] == code
    assert stub.calls == []


def test_empty_files_are_rejected(stub, stub_client):
    response = stub_client.post("/api/analyze", files=[("files", ("contract.docx", b"", DOCX_TYPE))])
    assert response.status_code == 400 and response.json()["error"]["code"] == "empty_file"
    assert stub.calls == []


@pytest.mark.parametrize("body, status", [({}, 422), ({"question": ""}, 422), ({"question": "x" * 2001}, 422)])
def test_ask_validates_its_body(stub, stub_client, body, status):
    assert stub_client.post("/api/ask", json=body).status_code == status
    assert stub.calls == []


def test_path_traversal_in_a_filename_is_reduced_to_a_base_name(stub, stub_client):
    stub_client.post("/api/analyze", files=[("files", ("../../etc/passwd.docx", contract_bytes(), DOCX_TYPE))])
    [request] = stub.calls
    assert request.documents[0].filename == "passwd.docx"


# --------------------------------------------------------------------------- failures
@pytest.mark.parametrize("status, http_status", [
    (AnalysisStatus.SUCCESS, 200), (AnalysisStatus.PARTIAL, 200), (AnalysisStatus.INSUFFICIENT_EVIDENCE, 200),
    (AnalysisStatus.INVALID_INPUT, 422), (AnalysisStatus.RAG_ERROR, 502), (AnalysisStatus.ANALYSIS_ERROR, 500),
])
def test_orchestrator_status_maps_to_http_status(status, http_status):
    stub = StubOrchestrator(status=status)
    client = TestClient(create_app(stub))
    response = client.post("/api/analyze", files=[("files", ("c.docx", contract_bytes(), DOCX_TYPE))])
    assert response.status_code == http_status == HTTP_STATUS[status]
    assert response.json()["status"] == status.value and response.json()["summary"] == "scripted result"


def test_an_orchestrator_failure_is_reported_without_internals():
    stub = StubOrchestrator(raises=RuntimeError("secret internal detail"))
    client = TestClient(create_app(stub), raise_server_exceptions=False)
    response = client.post("/api/ask", json={"question": "probation?"})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "secret internal detail" not in response.text and response.json()["error"]["detail"] == "RuntimeError"


def test_a_failing_rag_is_a_bad_gateway(client, real_orchestrator):
    real_orchestrator.rag.error_code, real_orchestrator.rag.answer = "vector_db_unreachable", None
    response = client.post("/api/ask", json={"question": "probation?"})
    assert response.status_code == 502 and response.json()["status"] == "rag_error"
    assert response.json()["errors"][0]["code"] == "routed_agent_failed"


def test_an_unusable_upload_is_unprocessable(client, sample_bytes):
    response = client.post("/api/analyze", files=[("files", ("scan.pdf", sample_bytes["sample_scanned.pdf"],
                                                             "application/pdf"))])
    assert response.status_code == 422 and response.json()["status"] == "invalid_input"
    assert response.json()["documents"][0]["document_status"] == "ocr_required"


# --------------------------------------------------------------------------- the API only calls the Orchestrator
def test_the_api_calls_handle_once_per_request(stub, stub_client, sample_bytes):
    analyze(stub_client, files_for("a.docx", "b.docx"), roles=["contract", "contract"])
    stub_client.post("/api/ask", json={"question": "probation?"})
    stub_client.get("/api/config")
    stub_client.get("/api/health")

    assert len(stub.calls) == 2  # only the two routed requests
    assert [len(call.documents) for call in stub.calls] == [2, 0]
    assert all(isinstance(call, SanadRequest) for call in stub.calls)


def test_the_api_passes_the_uploaded_bytes_unchanged(stub, stub_client):
    data = contract_bytes()
    analyze(stub_client, [("files", ("contract.docx", data, DOCX_TYPE))], roles=["contract"], labels=["Offer A"])
    [request] = stub.calls
    document = request.documents[0]
    assert document.content == data and document.role.value == "contract" and document.label == "Offer A"
    assert document.path is None  # nothing is written to disk
