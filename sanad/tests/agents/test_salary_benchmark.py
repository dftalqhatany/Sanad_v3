"""Real salary benchmarking: what the provider does with the pages a web search brings back."""

from __future__ import annotations

import pytest

from sanad.agents import AnalysisAgent, ContractAnalysisAgent, ContractComparisonAgent
from sanad.agents.salary import UnavailableSalaryBenchmarkProvider, build_salary_provider
from sanad.agents.salary_web import WebSearchSalaryProvider, build_query
from sanad.config import SalarySettings
from sanad.models.analysis import AnalysisStatus, SalaryQuery
from sanad.models.orchestration import SanadRequest, TaskHint, UploadedDocument
from sanad.orchestrator import SanadOrchestrator
from sanad.tools.web_search import WebSearchError
from tests.fakes.web_search import FakePage, FakeWebSearchClient
from tests.fixtures import salary_pages as pages

CONTRACT_LINES = ["Job Title: Data Analyst", "Basic Salary: 12,000 SAR per month", "Work Location: Riyadh",
                  "Annual Leave: 30 days", "Probation Period: 90 days",
                  "Working Hours: 8 hours per day, 40 hours per week"]
SETTINGS = SalarySettings(enabled=True, fx_rates_to_sar={"USD": 3.75})


@pytest.fixture
def contract(make_contract):
    _, extraction = make_contract(CONTRACT_LINES)
    return extraction


def provider(*page_list: FakePage, error: WebSearchError | None = None, raises: BaseException | None = None,
             settings: SalarySettings = SETTINGS) -> WebSearchSalaryProvider:
    client = FakeWebSearchClient(pages=list(page_list), error=error, raises=raises)
    return WebSearchSalaryProvider(client, settings, clock=lambda: __import__("datetime").datetime(2026, 9, 18))


# --------------------------------------------------------------------------- the query
def test_the_query_describes_the_job_the_place_and_the_experience(contract):
    agent = provider(pages.PAYLAB, pages.SAUDI_SALARY)
    agent.benchmark(contract, SalaryQuery(years_experience=3))
    assert agent.client.queries == ["Data Analyst salary Riyadh Saudi Arabia 3 years experience 2026"]
    assert build_query("Data Engineer", "Riyadh", SalaryQuery(seniority="senior"), 2026) == \
        "Data Engineer salary Riyadh Saudi Arabia senior 2026"


def test_the_contract_supplies_the_job_and_city_when_the_user_gives_none(contract):
    agent = provider(pages.PAYLAB, pages.SAUDI_SALARY)
    result = agent.benchmark(contract)
    assert agent.client.queries == ["Data Analyst salary Riyadh Saudi Arabia 2026"]
    assert result.job_title == "Data Analyst" and result.location == "Riyadh"


def test_an_explicit_query_overrides_the_contract(contract):
    agent = provider(pages.PAYLAB, pages.SAUDI_SALARY)
    agent.benchmark(contract, SalaryQuery(job_title="Data Engineer", location="Jeddah", years_experience=5))
    assert agent.client.queries == ["Data Engineer salary Jeddah Saudi Arabia 5 years experience 2026"]


def test_without_a_job_title_nothing_is_searched(make_contract):
    _, no_title = make_contract(["Basic Salary: 12,000 SAR per month"])
    agent = provider(pages.PAYLAB)
    result = agent.benchmark(no_title)
    assert result.status == "insufficient_data" and agent.client.queries == []
    assert "No job title" in result.message and result.market_min is None


# --------------------------------------------------------------------------- a successful benchmark
def test_several_market_sources_give_a_sourced_range(contract):
    result = provider(pages.PAYLAB, pages.SAUDI_SALARY, pages.KAGGLE).benchmark(contract)

    assert result.status == "success" and result.provider == "web_search"
    assert (result.market_min, result.market_max) == (9080, 18706)
    assert result.currency == "SAR" and result.period == "monthly"
    assert result.data_quality == "market" and result.basis == "total_compensation"
    assert [source.url for source in result.sources] == [pages.PAYLAB.url]
    assert result.contract_salary.amount == 12000 and "inside this range" in result.message
    assert result.market_min < result.market_max  # never a single point


def test_official_sources_outrank_market_sources(contract):
    result = provider(pages.SAUDI_OPEN_DATA, pages.PAYLAB, pages.SAUDI_SALARY).benchmark(contract)
    assert result.data_quality == "official"
    assert [source.tier for source in result.sources] == ["official"]
    assert (result.market_min, result.market_max) == (14200, 16850)
    assert all(observation.tier == "official" for observation in result.observations if observation.url in
               {source.url for source in result.sources})


def test_a_single_official_figure_is_widened_by_the_next_tier_and_marked_mixed(contract):
    result = provider(pages.SINGLE_OFFICIAL, pages.SAUDI_SALARY).benchmark(contract)
    assert result.status == "success" and result.data_quality == "mixed"
    assert {source.tier for source in result.sources} == {"official", "market"}
    assert any("official and market" in limitation for limitation in result.limitations)


def test_annual_figures_are_normalised_before_the_range(contract):
    result = provider(pages.ANNUAL_SOURCE, pages.KAGGLE).benchmark(contract)
    assert result.status == "success" and result.period == "monthly"
    assert (result.market_min, result.market_max) == (14000, 18000)  # 168,000-216,000 SAR a year
    assert result.data_quality == "market" and [s.url for s in result.sources] == [pages.ANNUAL_SOURCE.url]
    assert any("normalised to a month" in limitation for limitation in result.limitations)
    assert any(o.url == pages.KAGGLE.url for o in result.observations)  # the weaker source is reported, not used


def test_converted_currencies_are_flagged_and_unconvertible_ones_left_out(contract):
    result = provider(pages.USD_SOURCE, pages.EUR_SOURCE, pages.KAGGLE).benchmark(contract)
    assert result.status == "success"
    assert (result.market_min, result.market_max) == (11250, 15750)
    assert any("converted to SAR" in limitation for limitation in result.limitations)
    assert any("EUR were left out" in limitation for limitation in result.limitations)
    assert [o.currency for o in result.observations if o.url == pages.EUR_SOURCE.url] == ["EUR"]


def test_base_salary_and_total_compensation_are_never_put_in_one_range(contract):
    result = provider(pages.PAYLAB, pages.KAGGLE, pages.ANNUAL_SOURCE).benchmark(contract)
    assert result.basis == "base_salary"
    assert [source.url for source in result.sources] == [pages.ANNUAL_SOURCE.url]
    assert (result.market_min, result.market_max) == (14000, 18000)  # the Paylab total-pay range is not merged in
    assert any("total compensation" in limitation and "not merged" in limitation for limitation in result.limitations)
    assert any(o.basis == "total_compensation" for o in result.observations)  # still reported


def test_every_figure_keeps_its_url_quote_and_retrieval_date(contract):
    result = provider(pages.PAYLAB, pages.SAUDI_SALARY).benchmark(contract)
    assert result.observations and all(o.url.startswith("https://") and o.quote and o.retrieved_at == "2026-09-18"
                                       for o in result.observations)
    assert all(source.retrieved_at == "2026-09-18" and source.url for source in result.sources)
    assert result.searched_urls == [pages.PAYLAB.url, pages.SAUDI_SALARY.url]
    assert result.query == "Data Analyst salary Riyadh Saudi Arabia 2026"


# --------------------------------------------------------------------------- refusals
def test_linkedin_alone_is_never_a_benchmark(contract):
    result = provider(pages.LINKEDIN).benchmark(contract)
    assert result.status == "insufficient_data" and result.market_min is None
    assert result.observations[0].tier == "lead_only" and result.observations[0].usable is False
    assert any("LinkedIn" in limitation for limitation in result.limitations)


def test_linkedin_never_widens_a_range_from_real_sources(contract):
    result = provider(pages.PAYLAB, pages.LINKEDIN).benchmark(contract)
    assert result.status == "success" and result.market_max == 18706  # not the 30,000 SAR from the post
    assert pages.LINKEDIN.url not in [source.url for source in result.sources]
    assert any(o.url == pages.LINKEDIN.url and not o.usable for o in result.observations)


def test_sources_that_disagree_are_reported_not_averaged(contract):
    result = provider(pages.KAGGLE, pages.CONFLICTING_SOURCE).benchmark(contract)
    assert result.status == "insufficient_data" and result.data_quality == "conflicting"
    assert result.market_min is None and result.market_max is None
    assert any("disagree" in limitation for limitation in result.limitations)
    assert any("1,200 - 2,000 SAR per month" in limitation for limitation in result.limitations)
    assert any(pages.CONFLICTING_SOURCE.url in limitation for limitation in result.limitations)
    assert any("11,500" in limitation for limitation in result.limitations)


def test_a_single_figure_from_a_single_source_is_not_a_range(contract):
    result = provider(pages.KAGGLE).benchmark(contract)
    assert result.status == "insufficient_data" and result.market_min is None
    assert any("single figure" in limitation for limitation in result.limitations)
    assert "11,500" in result.message


def test_pages_without_usable_figures_give_insufficient_data(contract):
    result = provider(pages.NO_SALARY_SOURCE, pages.NO_PERIOD_SOURCE).benchmark(contract)
    assert result.status == "insufficient_data" and result.data_quality == "insufficient"
    assert any("monthly or annual" in limitation for limitation in result.limitations)
    assert result.searched_urls == [pages.NO_SALARY_SOURCE.url, pages.NO_PERIOD_SOURCE.url]


def test_no_sources_at_all_is_explicit(contract):
    result = provider().benchmark(contract)
    assert result.status == "insufficient_data" and "No salary source" in result.message


def test_a_search_failure_is_reported_as_an_error(contract):
    result = provider(error=WebSearchError("search_unavailable", "saudisalary.com could not be reached.")).benchmark(contract)
    assert result.status == "error" and result.market_min is None
    assert "could not be reached" in result.message
    assert any("search_unavailable" in limitation for limitation in result.limitations)


def test_a_source_timeout_is_reported_as_an_error(contract):
    result = provider(error=WebSearchError("search_timeout", "stats.gov.sa did not answer within 10s.")).benchmark(contract)
    assert result.status == "error" and "did not answer" in result.message
    assert any("search_timeout" in limitation for limitation in result.limitations)


def test_a_broken_client_never_breaks_the_analysis(contract):
    result = provider(raises=RuntimeError("boom")).benchmark(contract)
    assert result.status == "error" and "RuntimeError" in result.limitations[0]


def test_figures_about_another_job_are_flagged(contract):
    result = provider(pages.PAYLAB, pages.SAUDI_SALARY).benchmark(contract, SalaryQuery(job_title="Petroleum Engineer"))
    assert any("Petroleum Engineer" in note for o in result.observations for note in o.notes)
    assert any("not specific" in note for o in result.observations for note in o.notes)


# --------------------------------------------------------------------------- wiring
def test_nothing_configured_means_no_benchmarking():
    assert isinstance(build_salary_provider(SalarySettings()), UnavailableSalaryBenchmarkProvider)
    assert isinstance(build_salary_provider(SalarySettings(enabled=True)), WebSearchSalaryProvider)
    assert isinstance(build_salary_provider(SalarySettings(provider="tavily", api_key="k")), WebSearchSalaryProvider)
    injected = build_salary_provider(SalarySettings(), client=FakeWebSearchClient())
    assert isinstance(injected, WebSearchSalaryProvider)


def test_the_contract_analysis_carries_the_benchmark(contract, fake_rag):
    agent = ContractAnalysisAgent(fake_rag, salary_provider=provider(pages.PAYLAB, pages.SAUDI_SALARY))
    benchmark = agent.analyze(contract).salary_benchmark
    assert benchmark.status == "success" and benchmark.sources
    assert agent.salary_benchmark(contract, SalaryQuery(years_experience=2)).status == "success"


# --------------------------------------------------------------------------- comparison and orchestrator
def test_the_comparison_shows_the_market_evidence(contract, make_contract, fake_rag):
    _, cheaper = make_contract([line.replace("12,000", "8,000") for line in CONTRACT_LINES])
    analysis = AnalysisAgent(ContractAnalysisAgent(fake_rag, salary_provider=provider(pages.PAYLAB, pages.SAUDI_SALARY)))
    result = ContractComparisonAgent(analysis).compare([contract, cheaper])

    market = result.dimension("salary_vs_market")
    assert market.comparable and market.ranking == "not_ranked"
    values = {value.contract_id: value for value in market.values}
    assert "9,080 - 18,706 SAR per monthly" in values["contract_1"].display
    assert "https://saudiarabia.paylab.com/en/salaryinfo" in values["contract_1"].display
    assert "is inside that range" in values["contract_1"].display
    assert "is below that range" in values["contract_2"].display
    assert values["contract_1"].notes  # the benchmark's limitations travel with the value
    assert any("cited sources" in caveat for caveat in result.recommendation.caveats)


def test_a_contract_without_market_data_is_named_not_assumed(contract, make_contract, fake_rag):
    _, other_job = make_contract(["Job Title: Falconry Instructor", "Basic Salary: 8,000 SAR per month",
                                  "Annual Leave: 21 days"])

    class PerJobProvider:
        name = "per-job"

        def benchmark(self, contract_extraction, context=None):
            title = contract_extraction.job_title.value
            pages_for = (pages.PAYLAB, pages.SAUDI_SALARY) if title == "Data Analyst" else (pages.NO_SALARY_SOURCE,)
            return provider(*pages_for).benchmark(contract_extraction, context)

    analysis = AnalysisAgent(ContractAnalysisAgent(fake_rag, salary_provider=PerJobProvider()))
    result = ContractComparisonAgent(analysis).compare([contract, other_job])

    market = result.dimension("salary_vs_market")
    assert not market.comparable
    assert {v.contract_id: v.status for v in market.values} == {"contract_1": "available", "contract_2": "not_available"}
    caveat = next(c for c in result.recommendation.caveats if "Market salary data" in c)
    assert "contract_1 only" in caveat and "contract_2 has no market evidence" in caveat
    assert "says nothing about whether its salary is high or low" in caveat


def test_the_orchestrator_salary_route_returns_the_benchmark(fake_rag, knowledge_base, sample_bytes):
    from tests.fixtures.documents.builders import docx_from_paragraphs

    analysis = AnalysisAgent(ContractAnalysisAgent(fake_rag, salary_provider=provider(pages.PAYLAB, pages.SAUDI_SALARY)))
    orchestrator = SanadOrchestrator(analysis, ContractComparisonAgent(analysis), fake_rag)
    upload = UploadedDocument(filename="contract.docx", content=docx_from_paragraphs(CONTRACT_LINES), role="contract")

    result = orchestrator.handle(SanadRequest(documents=[upload], task=TaskHint.SALARY_BENCHMARK,
                                              salary_query=SalaryQuery(years_experience=3)))

    assert result.routing.route.value == "salary_benchmark" and result.status is AnalysisStatus.SUCCESS
    assert result.salary_benchmark.status == "success" and result.salary_benchmark.sources
    assert result.salary_benchmark.query.endswith("3 years experience 2026")
    assert "9,080" in result.summary or "9,080" in result.salary_benchmark.message
