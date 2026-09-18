"""Salary benchmarking: the interface, the 'nothing configured' provider, and how the real one is built.

SalaryBenchmark refuses a market range without sources, so no provider can invent data. The real
provider (sanad.agents.salary_web) reads documented web sources; it is used as soon as salary
benchmarking is configured (see SalarySettings), and otherwise this module reports 'not_configured'.
"""

from __future__ import annotations

from typing import Protocol

from sanad.config import SalarySettings
from sanad.models.analysis import SalaryBenchmark, SalaryQuery
from sanad.models.extraction import ContractExtraction, FieldStatus


class SalaryBenchmarkProvider(Protocol):
    name: str

    def benchmark(self, contract: ContractExtraction, context: SalaryQuery | None = None) -> SalaryBenchmark: ...


class UnavailableSalaryBenchmarkProvider:
    name = "none"

    def benchmark(self, contract: ContractExtraction, context: SalaryQuery | None = None) -> SalaryBenchmark:
        found = lambda field: field.value if field.status is FieldStatus.FOUND else None  # noqa: E731
        return SalaryBenchmark(
            status="not_configured",
            provider=self.name,
            message="Salary benchmarking is not configured; no market salary data was used or estimated.",
            contract_salary=found(contract.salary),
            job_title=found(contract.job_title),
            location=found(contract.work_location),
            data_quality="insufficient",
            limitations=["Salary benchmarking is not configured; set SANAD_SEARCH_PROVIDER with an API key, or "
                         "SANAD_SALARY_ENABLED=1 to read the documented sources directly."],
        )


def build_salary_provider(salary_settings: SalarySettings | None = None, *, client=None) -> SalaryBenchmarkProvider:
    """The real web-search provider when benchmarking is configured (or a client is injected), else 'not configured'."""
    salary_settings = salary_settings or SalarySettings.from_env()
    if client is None and not salary_settings.is_configured:
        return UnavailableSalaryBenchmarkProvider()
    from sanad.agents.salary_web import WebSearchSalaryProvider  # imported lazily: it needs the web tools
    from sanad.tools.web_search import HttpWebSearchClient

    return WebSearchSalaryProvider(client or HttpWebSearchClient(salary_settings), salary_settings)
