"""Deterministic, explainable recommendation over compared contract dimensions.

Two documented rules; no language model is involved:
  * priorities: the caller's ordered dimensions are applied one after another; each keeps only the
    contracts that lead on it. A contract is preferred when it is the only one left.
  * dominance (default): a contract is preferred only when it leads or ties on EVERY ranked dimension
    that could be compared, and at least one of those dimensions separates the contracts.
Otherwise the result is 'no_clear_preference' with the trade-offs, or 'not_possible' when nothing
could be compared. Every decision factor carries the compared values and their contract sources.
"""

from __future__ import annotations

from agents.comparison_dimensions import UNUSABLE_ANALYSIS
from models.analysis import AnalysisStatus
from models.comparison import (
    ComparedContract,
    ContractRisk,
    DecisionFactor,
    DimensionComparison,
    Recommendation,
)


def _names(ids: list[str], labels: dict[str, str]) -> str:
    return ", ".join(labels[i] for i in ids)


def _caveats(contracts: list[ComparedContract], dimensions: list[DimensionComparison]) -> list[str]:
    caveats = [f"{d.label} could not be compared: {d.explanation}" for d in dimensions
               if d.ranking != "not_ranked" and not d.comparable]
    market = next((d for d in dimensions if d.dimension == "salary_vs_market"), None)
    if market is not None and not market.comparable:
        with_data = [v.contract_id for v in market.values if v.status == "available"]
        without = [v.contract_id for v in market.values if v.status != "available"]
        if with_data:
            caveats.append(f"Market salary data was found for {', '.join(with_data)} only; "
                           f"{', '.join(without)} has no market evidence, which says nothing about whether its salary "
                           "is high or low.")
        else:
            caveats.append("No sourced market salary data was available, so salaries were not compared with the market.")
    elif market is not None and market.comparable:
        caveats.append("Market salary figures come from the cited sources as published; treat them as evidence to "
                       "check, not as a valuation.")
    compliance = next((d for d in dimensions if d.dimension == "compliance"), None)
    if compliance is not None and compliance.comparable and any(v.notes for v in compliance.values):
        caveats.append("Compliance was compared only by evidence-grounded non-compliant findings; without an automated "
                       "interpretation every regulatory finding still requires human review.")
    rag_failed = [c.contract_id for c in contracts if c.analysis_status is AnalysisStatus.RAG_ERROR]
    if rag_failed:
        caveats.append("The regulatory RAG was unavailable for " + ", ".join(rag_failed)
                       + "; no regulatory conclusion was made for it.")
    return caveats


def _factor(dimension: DimensionComparison, favours: list[str], explanation: str, ids: list[str]) -> DecisionFactor:
    return DecisionFactor(dimension=dimension.dimension, label=dimension.label, favours=favours, explanation=explanation,
                          evidence=[v for v in dimension.values if v.contract_id in ids])


def recommend(contracts: list[ComparedContract], dimensions: list[DimensionComparison], priorities: list[str],
              risks: list[ContractRisk]) -> Recommendation:
    labels = {c.contract_id: c.contract_id for c in contracts}
    caveats = _caveats(contracts, dimensions)
    unusable = [c for c in contracts if c.analysis_status in UNUSABLE_ANALYSIS]
    if unusable:
        return Recommendation(
            status="not_possible", method="none", caveats=caveats,
            explanation="No recommendation is made because the individual analysis failed for "
                        + ", ".join(f"{c.contract_id} ({c.analysis_status.value})" for c in unusable) + ".")

    result = (_by_priorities(contracts, dimensions, priorities, labels, caveats) if priorities
              else _by_dominance(contracts, dimensions, labels, caveats))
    if result.preferred_contract_id:
        high = [r for r in risks if r.contract_id == result.preferred_contract_id and r.severity == "high"]
        if high:
            result.caveats.append(f"{result.preferred_contract_id} still has {len(high)} high-severity issue(s); "
                                  "see the risks before deciding.")
    return result


def _by_priorities(contracts, dimensions, priorities, labels, caveats) -> Recommendation:
    by_name = {d.dimension: d for d in dimensions}
    candidates = [c.contract_id for c in contracts]
    factors: list[DecisionFactor] = []
    trade_offs: list[str] = []
    for name in priorities:
        dimension = by_name.get(name)
        if dimension is None:
            caveats.append(f"The priority '{name}' does not apply to this comparison and was skipped.")
            continue
        if not dimension.comparable:
            caveats.append(f"The priority '{dimension.label}' was skipped because it could not be compared.")
            continue
        values = {v.contract_id: v.value for v in dimension.values if v.contract_id in candidates}
        target = max(values.values()) if dimension.ranking == "higher_is_better" else min(values.values())
        leaders = [cid for cid in candidates if values[cid] == target]
        if len(leaders) == len(candidates):
            trade_offs.append(f"{dimension.label}: {_names(candidates, labels)} are equal.")
            continue
        displays = "; ".join(f"{labels[v.contract_id]}: {v.display}" for v in dimension.values if v.contract_id in candidates)
        factors.append(_factor(dimension, leaders, f"Priority '{dimension.label}': {_names(leaders, labels)} "
                                                   f"{'leads' if len(leaders) == 1 else 'lead'} ({displays}).", candidates))
        candidates = leaders
        if len(candidates) == 1:
            break

    if len(candidates) == 1:
        preferred = candidates[0]
        return Recommendation(
            status="preferred_contract", method="priorities", preferred_contract_id=preferred,
            priorities_used=list(priorities), decision_factors=factors, trade_offs=trade_offs, caveats=caveats,
            explanation=f"{labels[preferred]} is preferred by applying your priorities in order: "
                        + " then ".join(f.label for f in factors) + ".")
    if not factors and not trade_offs:
        explanation = ("None of the given priorities could be compared across the contracts, so no contract is "
                       "preferred. See the caveats for the reasons.")
    else:
        explanation = (f"After applying the priorities, {_names(candidates, labels)} remain equal on every priority "
                       "that could be compared.")
    return Recommendation(
        status="no_clear_preference", method="priorities", priorities_used=list(priorities), decision_factors=factors,
        trade_offs=trade_offs, caveats=caveats, explanation=explanation)


def _by_dominance(contracts, dimensions, labels, caveats) -> Recommendation:
    ids = [c.contract_id for c in contracts]
    ranked = [d for d in dimensions if d.ranking != "not_ranked" and d.comparable]
    if not ranked:
        return Recommendation(status="not_possible", method="dominance", caveats=caveats,
                              explanation="No ranked dimension could be compared across all contracts, so no "
                                          "recommendation is made.")
    separating = [d for d in ranked if not d.all_equal]
    if not separating:
        return Recommendation(status="no_clear_preference", method="dominance", caveats=caveats,
                              explanation="The contracts are equal on every dimension that could be compared ("
                                          + ", ".join(d.label for d in ranked) + ").")
    dominant = [cid for cid in ids if all(cid in d.best_contract_ids for d in separating)]
    factors = [_factor(d, d.best_contract_ids, d.explanation, ids) for d in separating]
    if len(dominant) == 1:
        preferred = dominant[0]
        equal = [d.label for d in ranked if d.all_equal]
        explanation = (f"{labels[preferred]} is at least as good as every other contract on all {len(ranked)} compared "
                       f"ranked dimensions and better on: " + ", ".join(d.label for d in separating) + ".")
        if equal:
            explanation += " Equal on: " + ", ".join(equal) + "."
        return Recommendation(status="preferred_contract", method="dominance", preferred_contract_id=preferred,
                              decision_factors=factors, caveats=caveats, explanation=explanation)
    if dominant:
        return Recommendation(
            status="no_clear_preference", method="dominance", decision_factors=factors, caveats=caveats,
            explanation=f"{_names(dominant, labels)} are equal to each other and at least as good as the other "
                        "contracts on every compared dimension, so no single contract is preferred.")
    trade_offs = [f"{d.label}: {_names(d.best_contract_ids, labels)} "
                  f"{'leads' if len(d.best_contract_ids) == 1 else 'lead'}. {d.explanation}" for d in separating]
    return Recommendation(
        status="no_clear_preference", method="dominance", decision_factors=factors, trade_offs=trade_offs,
        caveats=caveats,
        explanation="No contract is at least as good as the others on every compared dimension; the choice depends on "
                    "which trade-offs matter most to you. Provide priorities to apply them in order.")
