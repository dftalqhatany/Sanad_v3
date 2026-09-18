"""Small stand-ins for the documented salary sources, written in the same shapes the real pages use.

The wording mirrors what those pages actually publish (GASTAT writes thousands with a dot, Paylab
states a total monthly range including bonuses, SaudiSalary lists Arabic monthly grades), so the
extraction rules are tested against realistic text without fetching anything.
"""

from __future__ import annotations

from tests.fakes.web_search import FakePage

GASTAT = FakePage(
    url="https://www.stats.gov.sa/en/w/gastat-saudi-workers-monthly-average-wage-in-four-sectors-10.238-sar",
    title="GASTAT: Saudi workers' monthly average wage in four sectors 10.238 SAR",
    content=(
        "<html><body><h1>GASTAT: Saudi workers' monthly average wage</h1>"
        "<p>The average monthly wage of Saudi workers in four sectors reached (10.238) SAR. "
        "The wage includes basic, allowances, bonuses, overtimes and other compensations.</p>"
        "<p>In the private sector the average monthly wage of Saudi workers registered (7.339) SAR.</p>"
        "</body></html>"),
)
SAUDI_OPEN_DATA = FakePage(
    url="https://open.data.gov.sa/en/datasets/view/9801e8a2/preview/parsed/Average%20Salary%20per%20Department%202026.json",
    title="Average Salary per Department 2026",
    content_type="application/json",
    content=(
        '[{"department": "Data and Analytics", "average_monthly_salary_sar": 14200, "period": "monthly"},'
        ' {"department": "Engineering", "average_monthly_salary_sar": 16850, "period": "monthly"}]'),
)
PAYLAB = FakePage(
    url="https://saudiarabia.paylab.com/en/salaryinfo",
    title="Salaries in Saudi Arabia - Paylab.com",
    content=("<p>The salary range for people working as a Data Analyst in Saudi Arabia is typically from 9,080 SAR "
             "to 18,706 SAR per month - total monthly salary including bonuses.</p>"),
)
SAUDI_SALARY = FakePage(
    url="https://saudisalary.com/data-analyst-salary",
    title="رواتب محلل البيانات في السعودية",
    content=("<table><tr><td>محلل بيانات — الراتب الشهري 12,575 — 17,655 ر.س</td></tr>"
             "<tr><td>الدرجة الأولى | 12,575 ر.س شهرياً</td></tr></table>"),
)
KAGGLE = FakePage(
    url="https://www.kaggle.com/datasets/amirmahdiabbootalebi/salary-by-job-title-and-country",
    title="Salary by Job Title and Country",
    content="<p>Data Analyst in Saudi Arabia: average base salary 11,500 SAR per month (community dataset).</p>",
)
LINKEDIN = FakePage(
    url="https://lnkd.in/p/d96_HwV7",
    title="Salary insights post",
    content="<p>A Data Analyst in Riyadh earns about 30,000 SAR per month, someone told me.</p>",
)
USD_SOURCE = FakePage(
    url="https://saudiarabia.paylab.com/en/salaryinfo/usd",
    title="Data Analyst salaries in USD",
    content="<p>Data Analyst base salary is typically from 3,000 USD to 4,200 USD per month.</p>",
)
EUR_SOURCE = FakePage(
    url="https://saudiarabia.paylab.com/en/salaryinfo/eur",
    title="Data Analyst salaries in EUR",
    content="<p>Data Analyst base salary is typically from 2,800 EUR to 3,900 EUR per month.</p>",
)
ANNUAL_SOURCE = FakePage(
    url="https://saudisalary.com/data-analyst-annual",
    title="Data Analyst annual salary",
    content="<p>Data Analyst base salary in Saudi Arabia is 168,000 SAR to 216,000 SAR per year.</p>",
)
NO_PERIOD_SOURCE = FakePage(
    url="https://saudisalary.com/data-analyst-unclear",
    title="Data Analyst pay",
    content="<p>Data Analyst salary in Riyadh: 13,000 SAR to 15,000 SAR.</p>",
)
NO_SALARY_SOURCE = FakePage(
    url="https://saudisalary.com/about",
    title="About SaudiSalary",
    content="<p>SaudiSalary publishes public sector pay scales for Saudi Arabia.</p>",
)
SINGLE_OFFICIAL = FakePage(
    url="https://www.stats.gov.sa/en/w/average-monthly-wage",
    title="GASTAT: average monthly wage",
    content="<p>The average monthly wage in the private sector reached 11,000 SAR.</p>",
)
CONFLICTING_SOURCE = FakePage(
    url="https://www.kaggle.com/datasets/other/conflicting",
    title="Another dataset",
    content="<p>Data Analyst base salary in Saudi Arabia is 1,200 SAR to 2,000 SAR per month.</p>",
)
