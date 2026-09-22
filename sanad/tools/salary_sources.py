"""Which salary sources exist, and how much weight each one carries.

Official Saudi statistics outrank market salary sites, which outrank supplementary datasets.
LinkedIn (and its lnkd.in links) is a discovery lead only: its figures are never used as evidence
for a range on their own, and are reported separately even when they agree.
"""

from __future__ import annotations

from dataclasses import dataclass

from models.analysis import SourceTier
from tools.web_search import domain_of

TIER_ORDER: tuple[SourceTier, ...] = ("official", "market", "supplementary", "lead_only")
SOURCE_TYPE_BY_TIER = {"official": "official_statistics", "market": "market_survey", "supplementary": "other",
                       "lead_only": "other"}


@dataclass(frozen=True)
class SourceProfile:
    domain: str
    name: str
    tier: SourceTier
    note: str = ""


KNOWN_SOURCES: tuple[SourceProfile, ...] = (
    SourceProfile("stats.gov.sa", "GASTAT (General Authority for Statistics)", "official",
                  "Official Saudi wage statistics."),
    SourceProfile("open.data.gov.sa", "Saudi Open Data", "official", "Official Saudi open data portal."),
    SourceProfile("saudisalary.com", "SaudiSalary", "market", "Market salary site."),
    SourceProfile("paylab.com", "Paylab Saudi Arabia", "market", "Market salary survey site."),
    SourceProfile("kaggle.com", "Kaggle dataset", "supplementary",
                  "Community dataset; coverage and date are not guaranteed."),
    SourceProfile("linkedin.com", "LinkedIn", "lead_only",
                  "Discovery lead only; its figures are never used as evidence on their own."),
    SourceProfile("lnkd.in", "LinkedIn", "lead_only",
                  "Discovery lead only; its figures are never used as evidence on their own."),
)


def profile_for(url: str) -> SourceProfile:
    host = domain_of(url)
    for profile in KNOWN_SOURCES:
        if host == profile.domain or host.endswith("." + profile.domain):
            return profile
    return SourceProfile(host or "unknown", host or "unknown source", "supplementary",
                         "Not one of the documented salary sources.")


def tier_rank(tier: SourceTier) -> int:
    return TIER_ORDER.index(tier)
