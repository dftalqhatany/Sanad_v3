"""Salary benchmarking through the HTTP API."""

from __future__ import annotations

import pytest

from agents import AnalysisAgent, ContractAnalysisAgent, ContractComparisonAgent
from agents.salary_web import WebSearchSalaryProvider
from config import SalarySettings
from orchestrator import SanadOrchestrator
from tests.api.conftest import DOCX_TYPE, contract_bytes
from tests.fakes.regulatory_adapter import FakeRegulatoryAdapter
from tests.fakes.web_search import FakeWebSearchClient
from tests.fixtures import salary_pages as pages

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api import create_app  # noqa: E402

SALARY_SETTINGS = SalarySettings(enabled=True, fx_rates_to_sar={"USD": 3.75})


@pytest.fixture
def salary_client(knowledge_base) -> TestClient:
    provider = WebSearchSalaryProvider(FakeWebSearchClient(pages=[pages.PAYLAB, pages.SAUDI_SALARY]), SALARY_SETTINGS)
    analysis = AnalysisAgent(ContractAnalysisAgent(FakeRegulatoryAdapter(knowledge_base), salary_provider=provider))
    orchestrator = SanadOrchestrator(analysis, ContractComparisonAgent(analysis),
                                     FakeRegulatoryAdapter(knowledge_base))
    return TestClient(create_app(orchestrator, salary_settings=SALARY_SETTINGS))


def test_the_api_returns_a_sourced_salary_range(salary_client):
    response = salary_client.post("/api/analyze",
                                  files=[("files", ("contract.docx", contract_bytes(), DOCX_TYPE))],
                                  data={"task": "salary_benchmark", "roles": ["contract"], "years_experience": "3",
                                        "salary_location": "Riyadh"})
    body = response.json()
    benchmark = body["salary_benchmark"]

    assert response.status_code == 200 and body["routing"]["route"] == "salary_benchmark"
    assert benchmark["status"] == "success" and (benchmark["market_min"], benchmark["market_max"]) == (9080, 18706)
    assert benchmark["currency"] == "SAR" and benchmark["period"] == "monthly"
    assert benchmark["query"].endswith("3 years experience 2026")
    assert benchmark["sources"][0]["url"] == pages.PAYLAB.url and benchmark["sources"][0]["tier"] == "market"
    assert benchmark["sources"][0]["retrieved_at"] == "2026-09-18"
    assert benchmark["observations"] and benchmark["observations"][0]["quote"]
    assert benchmark["limitations"]


def test_the_api_reports_that_benchmarking_is_configured(salary_client):
    assert salary_client.get("/api/config").json()["salary_benchmarking_enabled"] is True


def test_a_bad_years_of_experience_is_refused(salary_client):
    response = salary_client.post("/api/analyze", files=[("files", ("c.docx", contract_bytes(), DOCX_TYPE))],
                                  data={"task": "salary_benchmark", "years_experience": "a while"})
    assert response.status_code == 400 and response.json()["error"]["code"] == "invalid_request"


def test_without_benchmarking_the_api_says_so(client):
    assert client.get("/api/config").json()["salary_benchmarking_enabled"] is False
    response = client.post("/api/analyze", files=[("files", ("c.docx", contract_bytes(), DOCX_TYPE))],
                           data={"task": "salary_benchmark", "roles": ["contract"]})
    benchmark = response.json()["salary_benchmark"]
    assert benchmark["status"] == "not_configured" and benchmark["market_min"] is None
    assert any("SANAD_SEARCH_PROVIDER" in limitation for limitation in benchmark["limitations"])
