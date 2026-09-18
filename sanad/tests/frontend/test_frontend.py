"""The user interface: it speaks HTTP to the API and renders what comes back, nothing more."""

from __future__ import annotations

import pytest

from frontend import view
from frontend.client import SanadApiClient
from tests.api.conftest import contract_bytes

pytest.importorskip("fastapi", reason="the frontend tests exercise the UI against the real API app")


@pytest.fixture
def ui(client) -> SanadApiClient:
    """The frontend client wired to the API through an in-process HTTP client."""
    return SanadApiClient(base_url="http://testserver", http_client=client)


@pytest.fixture
def stub_ui(stub_client) -> SanadApiClient:
    return SanadApiClient(base_url="http://testserver", http_client=stub_client)


def upload(name: str, *replacements: tuple[str, str], role: str = "auto"):
    return (name, contract_bytes(*replacements), role)


# --------------------------------------------------------------------------- the client speaks to the API only
def test_the_client_sends_files_roles_labels_and_options(stub, stub_ui):
    stub_ui.analyze([upload("a.docx", role="contract"), upload("b.docx", ("12,000", "9,000"), role="contract")],
                    labels=["Offer A", "Offer B"], priorities=["basic_salary"], task="contract_comparison",
                    target_job_json='{"title": "Data Analyst"}')
    [request] = stub.calls
    assert [d.filename for d in request.documents] == ["a.docx", "b.docx"]
    assert [d.role.value for d in request.documents] == ["contract", "contract"]
    assert [d.label for d in request.documents] == ["Offer A", "Offer B"]
    assert request.task.value == "contract_comparison" and request.priorities == ["basic_salary"]
    assert request.target_job.title == "Data Analyst"


def test_a_question_uses_the_ask_endpoint(stub, stub_ui):
    response = stub_ui.ask("What is the maximum probation period?")
    assert response.ok and stub.calls[0].question == "What is the maximum probation period?"
    assert stub.calls[0].documents == []


def test_api_errors_are_surfaced_to_the_user(stub, stub_ui):
    rejected = stub_ui.analyze([("scan.png", b"0" * 32, "auto")])
    assert not rejected.ok and rejected.status_code == 415
    assert rejected.error["code"] == "unsupported_file_type" and "PDF" not in rejected.message
    assert "not a supported file type" in rejected.message
    assert stub.calls == []


def test_an_unreachable_api_is_reported_not_crashed():
    response = SanadApiClient(base_url="http://127.0.0.1:59999", timeout_s=1).health()
    assert not response.ok and response.error["code"] == "api_unreachable"
    assert "could not be reached" in response.message


def test_the_client_reads_the_api_configuration(ui):
    config = ui.config()
    assert config.ok and config.result["allowed_file_types"] == [".pdf", ".docx"]
    assert config.result["max_contracts"] == 5


# --------------------------------------------------------------------------- rendering a single analysis
def test_a_contract_analysis_is_rendered_with_its_evidence(ui, sample_bytes):
    response = ui.analyze([upload("contract.docx", role="contract"),
                           ("cv.docx", sample_bytes["sample_cv_en.docx"], "cv")])
    result = response.result

    summary = view.overview(result)
    assert summary["status_label"] == "Completed" and summary["route_label"] == "Contract analysis"
    assert summary["agent"] == "AnalysisAgent" and summary["disclaimer"]

    rows = view.document_rows(result)
    assert [row["File"] for row in rows] == ["contract.docx", "cv.docx"]
    assert [row["Role"] for row in rows] == ["contract", "cv"]

    analysis = view.contract_analysis(result)
    findings = view.finding_rows(analysis)
    probation = next(row for row in findings if row["Field"] == "probation period")
    assert probation["Status"] == "Requires review" and probation["Articles"] == "53"
    assert probation["Contract says"] == "90 days" and probation["Source text"]
    assert probation["Finding"].startswith("F")

    evidence = view.evidence_items(analysis)
    assert evidence and all(item["citation"] and item["arabic_text"] for item in evidence)
    assert any(item["article_number"] == 53 for item in evidence)

    compatibility = view.compatibility_rows(view.cv_analysis(result))
    assert compatibility and compatibility[0]["CV evidence"]
    assert view.caveats(result)


def test_salary_information_is_rendered_without_inventing_market_data(ui):
    result = ui.analyze([upload("contract.docx", role="contract")], task="salary_benchmark").result
    salary = view.salary_view(result)
    assert salary["contract_salary"] == "12,000 SAR" and salary["job_title"] == "Data Analyst"
    assert salary["market_range"] == "no market data" and salary["sources"] == []
    assert salary["status"] == "not_configured"


def test_a_regulatory_answer_is_rendered_with_its_articles(ui):
    result = ui.ask("ما هي مدة الإجازة السنوية؟").result
    answer = view.regulatory_answer(result)
    assert answer["answer"] == "Scripted answer from the existing RAG."
    assert answer["evidence"] and answer["evidence"][0]["citation"] and answer["evidence"][0]["arabic_text"]
    assert view.overview(result)["route_label"] == "Regulatory question"


# --------------------------------------------------------------------------- rendering a comparison
def test_a_comparison_is_rendered_as_a_table_with_the_recommendation(ui):
    response = ui.analyze([upload("offer_a.docx", role="contract"),
                           upload("offer_b.docx", ("12,000", "9,000"), ("30 days", "21 days"), role="contract")],
                          labels=["Offer A", "Offer B"])
    comparison = response.result["comparison"]

    rows = view.comparison_rows(comparison)
    salary_row = next(row for row in rows if row["Dimension"] == "Basic salary")
    assert salary_row["Offer A"] == "12,000 SAR/month" and salary_row["Offer B"] == "9,000 SAR/month"
    assert salary_row["Leads"] == "Offer A" and salary_row["Ranked"] == "higher is better"
    assert {row["Category"] for row in rows} >= {"salary", "compliance", "benefits", "contract terms"}

    recommendation = view.recommendation_view(comparison)
    assert recommendation["status"] == "preferred_contract" and recommendation["preferred"] == "Offer A"
    assert recommendation["factors"] and all(factor["Evidence"] for factor in recommendation["factors"])
    assert recommendation["caveats"] and "no language model" in recommendation["basis"]

    risks = view.risk_rows(comparison)
    assert risks and [risk["Severity"] for risk in risks] == sorted(
        (risk["Severity"] for risk in risks), key=lambda s: {"high": 0, "medium": 1, "info": 2}[s])
    assert all(risk["Contract"] in {"Offer A", "Offer B"} for risk in risks)


def test_unusable_uploads_are_shown_to_the_user(ui, sample_bytes):
    response = ui.analyze([upload("offer_a.docx", role="contract"),
                           upload("offer_b.docx", ("12,000", "9,000"), role="contract"),
                           ("scan.pdf", sample_bytes["sample_scanned.pdf"], "auto")])
    result = response.result
    assert response.status_code == 200 and view.overview(result)["status_label"] == "Completed with gaps"
    scanned = next(row for row in view.document_rows(result) if row["File"] == "scan.pdf")
    assert scanned["Usable"] == "no" and "OCR" in scanned["Notes"]
    assert any("could not be used" in warning for warning in view.overview(result)["warnings"])


def test_a_rejected_request_is_rendered_from_the_structured_result(ui, sample_bytes):
    response = ui.analyze([("scan.pdf", sample_bytes["sample_scanned.pdf"], "auto")])
    assert response.status_code == 422 and response.result is not None
    summary = view.overview(response.result)
    assert summary["status_label"] == "Request rejected" and summary["route_label"] == "Not routed"
    assert summary["errors"] and any("unusable_document" in message for message in summary["errors"])
