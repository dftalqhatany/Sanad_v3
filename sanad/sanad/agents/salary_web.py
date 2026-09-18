"""Salary benchmarking from real web sources.

    WebSearchClient -> WebSearchSalaryProvider -> SalaryBenchmark

The provider plugs into the existing SalaryBenchmarkProvider interface, so the Analysis Agent, the
Comparison Agent, the Orchestrator, the API and the user interface need no new wiring.

What it will not do: invent a figure, average sources that disagree, mix base salary with total
compensation, convert a currency without a configured rate, or let a LinkedIn post set a range.
When a defensible range cannot be built it returns 'insufficient_data' and says why.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sanad.config import SalarySettings
from sanad.models.analysis import (
    SalaryBasis,
    SalaryBenchmark,
    SalaryObservation,
    SalaryQuery,
    SalarySource,
)
from sanad.models.extraction import ContractExtraction, FieldStatus
from sanad.tools.salary_extraction import ExtractionSettings, extract_observations, plain_text
from sanad.tools.salary_sources import SOURCE_TYPE_BY_TIER, profile_for, tier_rank
from sanad.tools.web_search import WebSearchClient, WebSearchError, WebSearchResult

logger = logging.getLogger(__name__)

SEARCH_TIERS = ("official", "market", "supplementary")
CONFLICT_RATIO = 4.0  # a union spanning more than 4x with non-overlapping sources is a disagreement, not a range
BASIS_LABELS = {"base_salary": "base salary", "total_compensation": "total compensation (allowances and bonuses "
                                                                    "included)", "unspecified": "pay of an unstated kind"}


class WebSearchSalaryProvider:
    name = "web_search"

    def __init__(self, client: WebSearchClient, settings: SalarySettings | None = None, clock=None) -> None:
        self.client = client
        self.settings = settings or SalarySettings.from_env()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.extraction = ExtractionSettings(fx_rates_to_sar=dict(self.settings.fx_rates_to_sar))

    # ------------------------------------------------------------------ public
    def benchmark(self, contract: ContractExtraction, context: SalaryQuery | None = None) -> SalaryBenchmark:
        found = lambda field: field.value if field.status is FieldStatus.FOUND else None  # noqa: E731
        job_title = (context.job_title if context and context.job_title else None) or found(contract.job_title)
        location = (context.location if context and context.location else None) or found(contract.work_location)
        base = dict(provider=self.name, contract_salary=found(contract.salary), job_title=job_title, location=location)

        if not job_title:
            return SalaryBenchmark(**base, status="insufficient_data", data_quality="insufficient",
                                   message="No job title was found in the contract, so no salary benchmark was "
                                           "searched for. Give the job title explicitly to benchmark this salary.",
                                   limitations=["A benchmark needs a job title; the contract does not state one."])

        query = build_query(job_title, location, context, self.clock().year)
        try:
            results = self.client.search(query)
        except WebSearchError as exc:
            logger.warning("Salary search failed (%s)", exc.code)
            return SalaryBenchmark(**base, status="error", query=query, data_quality="insufficient",
                                   message=f"The salary search could not be completed: {exc.message}",
                                   limitations=[f"Search error '{exc.code}'; no market data was used."])
        except Exception as exc:  # a broken client must not break the analysis
            logger.warning("Salary search raised %s", type(exc).__name__)
            return SalaryBenchmark(**base, status="error", query=query, data_quality="insufficient",
                                   message="The salary search could not be completed.",
                                   limitations=[f"Unexpected search failure ({type(exc).__name__})."])

        if not results:
            return SalaryBenchmark(**base, status="insufficient_data", query=query, data_quality="insufficient",
                                   message="No salary source was available for this job, so no market range is "
                                           "reported.",
                                   limitations=["The configured sources returned nothing for this query."])

        observations, limitations = self._observations(results, job_title)
        searched = [result.url for result in results]
        if not any(o.usable and o.tier != "lead_only" for o in observations):
            return SalaryBenchmark(**base, status="insufficient_data", query=query, observations=observations,
                                   searched_urls=searched, data_quality="insufficient",
                                   message="No usable salary figure could be read from the sources that answered, so "
                                           "no market range is reported.",
                                   limitations=limitations or ["No source stated a salary with a currency and period."])
        return self._benchmark(base, query, observations, searched, limitations)

    # ------------------------------------------------------------------ internals
    def _observations(self, results: list[WebSearchResult], job_title: str) -> tuple[list[SalaryObservation], list[str]]:
        observations: list[SalaryObservation] = []
        limitations: list[str] = []
        for result in results:
            profile = profile_for(result.url)
            found = extract_observations(result, self.extraction, profile)
            if found and not _mentions(plain_text(result) + " " + result.title, job_title):
                for item in found:
                    item.notes.append(f"The page does not mention '{job_title}'; the figures are not specific to it.")
            observations += found
        for currency in sorted({o.currency for o in observations if o.currency != "SAR" and o.converted_from is None
                                and not o.usable}):
            limitations.append(f"Figures in {currency} were left out: no exchange rate is configured for it.")
        if any(o.tier == "lead_only" for o in observations):
            limitations.append("LinkedIn figures were found but are treated as leads only; they do not set the range.")
        if any(not o.usable and "monthly or annual" in " ".join(o.notes) for o in observations):
            limitations.append("Some figures did not state whether they were monthly or annual and were left out.")
        return observations, limitations

    def _benchmark(self, base: dict, query: str, observations: list[SalaryObservation], searched: list[str],
                   limitations: list[str]) -> SalaryBenchmark:
        usable = [o for o in observations if o.usable and o.tier != "lead_only"]
        basis, basis_note = _choose_basis(usable)
        limitations = [*limitations, *basis_note]
        group = [o for o in usable if o.basis == basis]

        chosen: list[SalaryObservation] = []
        tiers_used: list[str] = []
        for tier in SEARCH_TIERS:
            tier_items = [o for o in group if o.tier == tier]
            if not tier_items:
                continue
            chosen += tier_items
            tiers_used.append(tier)
            if _is_range(chosen):
                break

        if not chosen:
            return SalaryBenchmark(**base, status="insufficient_data", query=query, observations=observations,
                                   searched_urls=searched, data_quality="insufficient", limitations=limitations,
                                   message="No source of sufficient quality reported a comparable salary figure.")
        if not _is_range(chosen):
            limitations.append("Only a single figure from a single source was found; a range cannot be defended.")
            return SalaryBenchmark(**base, status="insufficient_data", query=query, observations=observations,
                                   searched_urls=searched, data_quality="insufficient", limitations=limitations,
                                   message=f"Only one figure was found ({_describe(chosen[0])}), which is not enough "
                                           "for a market range.")

        conflict = _conflict(chosen)
        if conflict:
            limitations += conflict
            return SalaryBenchmark(**base, status="insufficient_data", query=query, observations=observations,
                                   searched_urls=searched, data_quality="conflicting", limitations=limitations,
                                   message="The sources disagree too much for a defensible range; each source's own "
                                           "figures are reported instead of an average.")

        low = min(o.monthly_min for o in chosen)
        high = max(o.monthly_max for o in chosen)
        quality = tiers_used[0] if len(tiers_used) == 1 else "mixed"
        sources = _sources(chosen)
        limitations.append(f"The range is the span of {len(sources)} source(s) at the "
                           f"{' and '.join(tiers_used)} level, not a statistical estimate.")
        if any(o.converted_from for o in chosen):
            limitations.append("Some figures were converted to SAR at the configured rate.")
        if any(o.period == "annual" for o in chosen):
            limitations.append("Annual figures were normalised to a month by dividing by 12.")
        if any("does not mention" in note for o in chosen for note in o.notes):
            limitations.append("Some sources report general Saudi pay levels rather than this exact job title.")

        message = (f"Observed {BASIS_LABELS[basis]} between {low:,.0f} and {high:,.0f} SAR per month "
                   f"across {len(sources)} source(s): {', '.join(s.name for s in sources)}.")
        contract_salary = base.get("contract_salary")
        if contract_salary is not None and contract_salary.amount and contract_salary.period in (None, "monthly"):
            message += " " + _position(contract_salary.amount, low, high)
        return SalaryBenchmark(**base, status="success", query=query, observations=observations, searched_urls=searched,
                               market_min=round(low, 2), market_max=round(high, 2), currency="SAR", period="monthly",
                               basis=basis, data_quality=quality, sources=sources, limitations=limitations,
                               message=message)


# --------------------------------------------------------------------------- helpers
def build_query(job_title: str, location: str | None, context: SalaryQuery | None, year: int) -> str:
    parts = [job_title, "salary"]
    if location:
        parts.append(location)
    parts.append("Saudi Arabia")
    if context and context.years_experience is not None:
        parts.append(f"{context.years_experience:g} years experience")
    elif context and context.seniority:
        parts.append(context.seniority)
    parts.append(str(year))
    return " ".join(parts)


def _mentions(text: str, job_title: str) -> bool:
    lowered = text.lower()
    words = [word for word in job_title.lower().split() if len(word) > 2]
    return bool(words) and all(word in lowered for word in words)


def _choose_basis(usable: list[SalaryObservation]) -> tuple[SalaryBasis, list[str]]:
    """One basis only: base salary and total compensation are never put in the same range."""
    groups: dict[SalaryBasis, list[SalaryObservation]] = {}
    for observation in usable:
        groups.setdefault(observation.basis, []).append(observation)
    ranked = sorted(groups.items(), key=lambda item: (min(tier_rank(o.tier) for o in item[1]),
                                                      -len({o.url for o in item[1]})))
    basis = ranked[0][0]
    notes = []
    for other, items in ranked[1:]:
        notes.append(f"{len(items)} figure(s) describing {BASIS_LABELS[other]} were reported separately and not "
                     f"merged into the {BASIS_LABELS[basis]} range.")
    return basis, notes


def _is_range(observations: list[SalaryObservation]) -> bool:
    """A range needs either a published range, or two different figures to span."""
    if any(o.monthly_max > o.monthly_min for o in observations):
        return True
    return len({round(o.monthly_min) for o in observations}) >= 2


def _conflict(observations: list[SalaryObservation]) -> list[str]:
    low = min(o.monthly_min for o in observations)
    high = max(o.monthly_max for o in observations)
    if low <= 0 or high / low <= CONFLICT_RATIO:
        return []
    by_page: dict[str, tuple[str, float, float]] = {}
    for observation in observations:
        current = by_page.get(observation.url)
        by_page[observation.url] = (
            observation.source_name,
            min(observation.monthly_min, current[1]) if current else observation.monthly_min,
            max(observation.monthly_max, current[2]) if current else observation.monthly_max,
        )
    spans = [(low_, high_) for _, low_, high_ in by_page.values()]
    disjoint = any(a[1] < b[0] or b[1] < a[0] for i, a in enumerate(spans) for b in spans[i + 1:])
    if not disjoint:
        return []
    return [f"The sources disagree: {high / low:.1f}x between the lowest and highest figure."] + [
        f"{name} <{url}>: {low_:,.0f} - {high_:,.0f} SAR per month."
        for url, (name, low_, high_) in sorted(by_page.items())]


def _sources(observations: list[SalaryObservation]) -> list[SalarySource]:
    sources: dict[str, SalarySource] = {}
    for observation in observations:
        sources.setdefault(observation.url, SalarySource(
            name=observation.source_name, url=observation.url, source_type=SOURCE_TYPE_BY_TIER[observation.tier],
            retrieved_at=observation.retrieved_at or None, tier=observation.tier))
    return list(sources.values())


def _describe(observation: SalaryObservation) -> str:
    span = (f"{observation.monthly_min:,.0f}" if observation.monthly_min == observation.monthly_max
            else f"{observation.monthly_min:,.0f}-{observation.monthly_max:,.0f}")
    return f"{span} SAR/month from {observation.source_name}"


def _position(amount: float, low: float, high: float) -> str:
    if amount < low:
        return f"The contract salary ({amount:,.0f} SAR) is below this range."
    if amount > high:
        return f"The contract salary ({amount:,.0f} SAR) is above this range."
    return f"The contract salary ({amount:,.0f} SAR) is inside this range."
