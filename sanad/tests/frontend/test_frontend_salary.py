"""The user interface shows the salary range, its sources, the evidence and the caveats."""

from __future__ import annotations

import pytest

from frontend import view
from frontend.client import SanadApiClient
from tests.api.conftest import contract_bytes
from tests.fixtures import salary_pages as pages

pytest.importorskip("fastapi")


@pytest.fixture
def ui(salary_client) -> SanadApiClient:
    return SanadApiClient(base_url="http://testserver", http_client=salary_client)


def test_the_salary_panel_shows_the_range_sources_and_evidence(ui):
    response = ui.analyze([("contract.docx", contract_bytes(), "contract")], task="salary_benchmark",
                          salary={"years_experience": "3"})
    salary = view.salary_view(response.result)

    assert salary["status"] == "success"
    assert salary["market_range"] == "9,080 - 18,706 SAR" and salary["period"] == "monthly"
    assert salary["contract_salary"] == "12,000 SAR" and salary["position"] == "inside the observed market range"
    assert salary["basis"] == "total compensation" and salary["data_quality"] == "market"
    assert salary["query"].endswith("3 years experience 2026")
    assert salary["sources"][0]["url"] == pages.PAYLAB.url and salary["sources"][0]["tier"] == "market"
    assert salary["sources"][0]["retrieved_at"] == "2026-09-18"

    evidence = salary["evidence"]
    assert evidence and evidence[0]["Quote"] and evidence[0]["URL"].startswith("https://")
    assert evidence[0]["Used"] == "yes" and "SAR / monthly" in evidence[0]["Figure"]
    assert salary["limitations"]
    assert set(salary["limitations"]) <= set(view.caveats(response.result))


def test_a_missing_benchmark_is_shown_as_such(ui, client):
    plain = SanadApiClient(base_url="http://testserver", http_client=client)
    response = plain.analyze([("contract.docx", contract_bytes(), "contract")], task="salary_benchmark")
    salary = view.salary_view(response.result)
    assert salary["status"] == "not_configured" and salary["market_range"] == "no market data"
    assert salary["position"] is None and salary["sources"] == [] and salary["evidence"] == []
