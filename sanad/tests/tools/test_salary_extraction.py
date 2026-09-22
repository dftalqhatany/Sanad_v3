"""Reading salary figures out of pages: what is kept, what is normalised, what is refused."""

from __future__ import annotations

import pytest

from tools.salary_extraction import ExtractionSettings, extract_observations, plain_text, read_number
from tools.web_search import WebSearchResult
from tests.fakes.web_search import FakePage, FakeWebSearchClient
from tests.fixtures import salary_pages as pages

SETTINGS = ExtractionSettings(fx_rates_to_sar={"USD": 3.75})


def read(page: FakePage, settings: ExtractionSettings = SETTINGS):
    [result] = FakeWebSearchClient(pages=[page]).search("Data Analyst salary Riyadh Saudi Arabia 2026")
    return extract_observations(result, settings)


def test_numbers_written_in_either_style_are_read_the_same():
    assert read_number("12,000") == 12000
    assert read_number("10.238") == 10238      # GASTAT writes thousands with a dot
    assert read_number("1.234.567") == 1234567
    assert read_number("4 080") == 4080
    assert read_number("12000.50") == 12000.50


def test_a_monthly_range_is_read_with_its_quote_and_source():
    [observation] = read(pages.PAYLAB)
    assert (observation.minimum, observation.maximum) == (9080, 18706)
    assert observation.currency == "SAR" and observation.period == "monthly"
    assert (observation.monthly_min, observation.monthly_max) == (9080, 18706)
    assert observation.tier == "market" and observation.source_name == "Paylab Saudi Arabia"
    assert observation.url == pages.PAYLAB.url and observation.retrieved_at == "2026-09-18"
    assert "9,080 SAR to 18,706 SAR" in observation.quote
    assert observation.usable


def test_an_annual_figure_is_normalised_to_a_month():
    [observation] = read(pages.ANNUAL_SOURCE)
    assert observation.period == "annual" and (observation.minimum, observation.maximum) == (168000, 216000)
    assert (observation.monthly_min, observation.monthly_max) == (14000, 18000)
    assert any("/ 12" in note for note in observation.notes)


@pytest.mark.parametrize("page, currency, converted", [(pages.USD_SOURCE, "USD", True), (pages.EUR_SOURCE, "EUR", False)])
def test_currencies_are_converted_only_with_a_configured_rate(page, currency, converted):
    [observation] = read(page)
    assert observation.currency == currency
    assert observation.usable is converted
    if converted:
        assert (observation.monthly_min, observation.monthly_max) == (11250, 15750)
        assert observation.converted_from == "USD" and any("3.75" in note for note in observation.notes)
    else:
        assert observation.converted_from is None
        assert any("no exchange rate is configured" in note for note in observation.notes)
        assert (observation.monthly_min, observation.monthly_max) == (2800, 3900)  # left in EUR, never mixed in


def test_base_salary_and_total_compensation_are_labelled_apart():
    assert read(pages.GASTAT)[0].basis == "total_compensation"   # "basic, allowances, bonuses, overtimes"
    assert read(pages.PAYLAB)[0].basis == "total_compensation"   # "total monthly salary including bonuses"
    assert read(pages.KAGGLE)[0].basis == "base_salary"
    assert read(pages.ANNUAL_SOURCE)[0].basis == "base_salary"


def test_a_figure_without_a_stated_period_is_reported_but_not_usable():
    observations = read(pages.NO_PERIOD_SOURCE)
    assert observations and not any(o.usable for o in observations)
    assert all("monthly or annual" in " ".join(o.notes) for o in observations)


def test_a_page_without_salary_figures_yields_nothing():
    assert read(pages.NO_SALARY_SOURCE) == []


def test_arabic_monthly_grades_are_read():
    observations = read(pages.SAUDI_SALARY)
    assert (observations[0].minimum, observations[0].maximum) == (12575, 17655)
    assert observations[0].currency == "SAR" and observations[0].period == "monthly"


def test_json_data_is_read_field_by_field():
    observations = read(pages.SAUDI_OPEN_DATA)
    assert [o.minimum for o in observations] == [14200, 16850]
    assert all(o.tier == "official" and o.period == "monthly" for o in observations)
    assert "average_monthly_salary_sar" in observations[0].quote


def test_linkedin_figures_are_never_usable():
    [observation] = read(pages.LINKEDIN)
    assert observation.tier == "lead_only" and observation.usable is False
    assert any("discovery lead" in note for note in observation.notes)


def test_implausible_figures_are_dropped():
    page = FakePage(url="https://saudisalary.com/x", title="x",
                    content="<p>Salary of 12 SAR per month and 9,000,000 SAR per month.</p>")
    assert read(page) == []


def test_html_and_json_are_reduced_to_text():
    html = WebSearchResult(query="q", url="https://saudisalary.com/a", content="<p>Salary <b>12,000</b> SAR</p>")
    assert "<" not in plain_text(html) and "12,000" in plain_text(html)
    data = WebSearchResult(query="q", url="https://open.data.gov.sa/a.json", content='{"salary": 9000}',
                           content_type="application/json")
    assert "salary: 9000" in plain_text(data)
