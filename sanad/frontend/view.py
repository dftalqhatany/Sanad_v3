"""Turns an OrchestratorResult JSON payload into rows the user interface renders.

Pure functions over the API payload: no decisions, no thresholds, no re-ranking, no analysis. Text
that the backend already wrote (explanations, caveats, recommendations) is passed through unchanged.
"""

from __future__ import annotations

from typing import Any

STATUS_LABELS = {
    "success": "Completed",
    "partial": "Completed with gaps",
    "insufficient_evidence": "Not enough evidence",
    "invalid_input": "Request rejected",
    "rag_error": "Regulatory search unavailable",
    "analysis_error": "Failed",
}
ROUTE_LABELS = {
    "contract_analysis": "Contract analysis",
    "cv_analysis": "CV analysis",
    "contract_comparison": "Contract comparison",
    "regulatory_question": "Regulatory question",
    "salary_benchmark": "Salary benchmarking",
    "none": "Not routed",
}
FINDING_LABELS = {
    "compliant": "Compliant", "non_compliant": "Non-compliant", "requires_review": "Requires review",
    "insufficient_evidence": "Insufficient evidence", "not_applicable": "Document fact",
    "not_found": "Not found in contract", "error": "Check failed",
}
SEVERITY_ORDER = {"high": 0, "medium": 1, "info": 2}


def overview(result: dict[str, Any]) -> dict[str, Any]:
    routing = result.get("routing", {})
    return {
        "status": result.get("status"),
        "status_label": STATUS_LABELS.get(result.get("status", ""), result.get("status", "")),
        "route": routing.get("route"),
        "route_label": ROUTE_LABELS.get(routing.get("route", ""), routing.get("route", "")),
        "agent": routing.get("agent"),
        "rule": routing.get("rule"),
        "reason": routing.get("reason"),
        "summary": result.get("summary", ""),
        "warnings": result.get("warnings", []),
        "errors": [f"{e.get('code')}: {e.get('message')}" for e in result.get("errors", [])],
        "disclaimer": result.get("disclaimer", ""),
    }


def document_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "File": document.get("filename"),
            "Role": (document.get("role") or "not identified"),
            "Role from": document.get("role_source"),
            "Document": document.get("document_status"),
            "Extraction": document.get("extraction_status") or "-",
            "Pages": document.get("page_count") or "-",
            "Usable": "yes" if document.get("usable") else "no",
            "Notes": " ".join(document.get("notes", []) + [e.get("message", "") for e in document.get("errors", [])]),
        }
        for document in result.get("documents", [])
    ]


# --------------------------------------------------------------------------- single analysis
def contract_analysis(result: dict[str, Any]) -> dict[str, Any] | None:
    analysis = (result.get("analysis") or {}).get("contract_analysis")
    return analysis or None


def cv_analysis(result: dict[str, Any]) -> dict[str, Any] | None:
    return (result.get("analysis") or {}).get("cv_analysis") or None


def finding_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for finding in analysis.get("findings", []):
        fact = finding.get("contract_fact", {})
        rows.append({
            "Finding": finding.get("finding_id"),
            "Field": finding.get("field", "").replace("_", " "),
            "Status": FINDING_LABELS.get(finding.get("status", ""), finding.get("status")),
            "Contract says": fact.get("raw_value") or "-",
            "Page": fact.get("page_number") or "-",
            "Articles": ", ".join(str(e.get("article_number")) for e in finding.get("regulatory_evidence", [])
                                  if e.get("article_number")) or "-",
            "Explanation": finding.get("explanation", ""),
            "Source text": fact.get("source_text") or "",
        })
    return rows


def evidence_items(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Every retrieved article, with its citation and both texts (Arabic is the legal reference)."""
    return [
        {
            "evidence_id": item.get("evidence_id"),
            "citation": item.get("citation"),
            "article_number": item.get("article_number"),
            "article_name": item.get("article_name"),
            "part": item.get("part"),
            "chapter": item.get("chapter"),
            "arabic_text": item.get("arabic_text"),
            "english_text": item.get("english_text"),
            "rank": item.get("rank"),
            "score": item.get("score"),
            "source": item.get("source"),
        }
        for item in analysis.get("evidence", [])
    ]


def compatibility_rows(cv: dict[str, Any] | None) -> list[dict[str, Any]]:
    compatibility = (cv or {}).get("job_compatibility")
    if not compatibility:
        return []
    return [
        {
            "Requirement": requirement.get("requirement"),
            "Type": requirement.get("requirement_type", "").replace("_", " "),
            "Importance": requirement.get("importance"),
            "Status": requirement.get("status", "").replace("_", " "),
            "Explanation": requirement.get("explanation"),
            "CV evidence": "; ".join(e.get("text", "") for e in requirement.get("cv_evidence", [])),
        }
        for requirement in compatibility.get("requirements", [])
    ]


# --------------------------------------------------------------------------- comparison
def comparison_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per compared dimension, one column per contract, plus who leads and why."""
    names = {c["contract_id"]: c.get("label") or c["contract_id"] for c in comparison.get("contracts", [])}
    rows = []
    for section, dimensions in (comparison.get("comparison") or {}).items():
        for dimension in dimensions:
            row = {"Category": section.replace("_", " "), "Dimension": dimension.get("label"),
                   "Ranked": {"higher_is_better": "higher is better", "lower_is_better": "lower is better",
                              "not_ranked": "not ranked"}.get(dimension.get("ranking"), "")}
            for value in dimension.get("values", []):
                row[names.get(value["contract_id"], value["contract_id"])] = value.get("display", "")
            row["Leads"] = ", ".join(names.get(i, i) for i in dimension.get("best_contract_ids", [])) or "-"
            row["Explanation"] = dimension.get("explanation", "")
            rows.append(row)
    return rows


def recommendation_view(comparison: dict[str, Any]) -> dict[str, Any]:
    recommendation = comparison.get("recommendation") or {}
    names = {c["contract_id"]: c.get("label") or c["contract_id"] for c in comparison.get("contracts", [])}
    preferred = recommendation.get("preferred_contract_id")
    return {
        "status": recommendation.get("status"),
        "preferred": names.get(preferred, preferred),
        "method": recommendation.get("method"),
        "explanation": recommendation.get("explanation", ""),
        "factors": [
            {
                "Dimension": factor.get("label"),
                "Favours": ", ".join(names.get(i, i) for i in factor.get("favours", [])),
                "Why": factor.get("explanation"),
                "Evidence": "; ".join(f"{names.get(v['contract_id'], v['contract_id'])}: {v.get('display', '')}"
                                      for v in factor.get("evidence", [])),
            }
            for factor in recommendation.get("decision_factors", [])
        ],
        "trade_offs": recommendation.get("trade_offs", []),
        "caveats": recommendation.get("caveats", []),
        "basis": recommendation.get("basis", ""),
    }


def risk_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    names = {c["contract_id"]: c.get("label") or c["contract_id"] for c in comparison.get("contracts", [])}
    risks = sorted(comparison.get("risks", []), key=lambda r: (SEVERITY_ORDER.get(r.get("severity", "info"), 3),
                                                               r.get("contract_id", "")))
    return [
        {
            "Contract": names.get(risk.get("contract_id"), risk.get("contract_id")),
            "Severity": risk.get("severity"),
            "Type": risk.get("code", "").replace("_", " "),
            "Field": (risk.get("field") or "-").replace("_", " "),
            "Finding": risk.get("finding_id") or "-",
            "Message": risk.get("message"),
        }
        for risk in risks
    ]


# --------------------------------------------------------------------------- salary and questions
def salary_view(result: dict[str, Any]) -> dict[str, Any] | None:
    benchmark = result.get("salary_benchmark")
    if benchmark is None:
        analysis = contract_analysis(result)
        benchmark = (analysis or {}).get("salary_benchmark")
    if not benchmark:
        return None
    salary = benchmark.get("contract_salary") or {}
    has_range = benchmark.get("market_min") is not None and benchmark.get("market_max") is not None
    basis = {"base_salary": "base salary", "total_compensation": "total compensation",
             "unspecified": "pay of an unstated kind", None: ""}.get(benchmark.get("basis"), "")
    return {
        "status": benchmark.get("status"),
        "message": benchmark.get("message"),
        "contract_salary": (f"{salary.get('amount'):,.0f} {salary.get('currency') or ''}".strip()
                            if salary.get("amount") is not None else "not found in the contract"),
        "job_title": benchmark.get("job_title") or "-",
        "location": benchmark.get("location") or "-",
        "market_range": (f"{benchmark['market_min']:,.0f} - {benchmark['market_max']:,.0f} "
                         f"{benchmark.get('currency') or ''}".strip() if has_range else "no market data"),
        "period": benchmark.get("period") or "-",
        "basis": basis,
        "data_quality": benchmark.get("data_quality") or "-",
        "query": benchmark.get("query"),
        "limitations": list(benchmark.get("limitations", [])),
        "position": _salary_position(salary.get("amount"), benchmark) if has_range else None,
        "sources": [{"name": s.get("name"), "url": s.get("url"), "type": s.get("source_type"),
                     "tier": s.get("tier") or "-", "retrieved_at": s.get("retrieved_at") or "-"}
                    for s in benchmark.get("sources", [])],
        "evidence": [
            {
                "Source": observation.get("source_name"),
                "Tier": observation.get("tier"),
                "Figure": (f"{observation['minimum']:,.0f}" if observation["minimum"] == observation["maximum"]
                           else f"{observation['minimum']:,.0f} - {observation['maximum']:,.0f}")
                          + f" {observation.get('currency', '')} / {observation.get('period', '')}",
                "Monthly SAR": (f"{observation['monthly_min']:,.0f}"
                                if observation["monthly_min"] == observation["monthly_max"]
                                else f"{observation['monthly_min']:,.0f} - {observation['monthly_max']:,.0f}"),
                "Basis": observation.get("basis", "").replace("_", " "),
                "Used": "yes" if observation.get("usable") else "no",
                "Retrieved": observation.get("retrieved_at"),
                "Quote": observation.get("quote", "")[:300],
                "URL": observation.get("url"),
            }
            for observation in benchmark.get("observations", [])
        ],
    }


def _salary_position(amount, benchmark: dict[str, Any]) -> str | None:
    if amount is None:
        return None
    if amount < benchmark["market_min"]:
        return "below the observed market range"
    if amount > benchmark["market_max"]:
        return "above the observed market range"
    return "inside the observed market range"


def regulatory_answer(result: dict[str, Any]) -> dict[str, Any] | None:
    answer = result.get("regulatory_answer")
    if not answer:
        return None
    return {
        "answer": answer.get("answer"),
        "message": answer.get("message"),
        "status": answer.get("status"),
        "evidence": [
            {
                "citation": item.get("citation"),
                "article_number": (item.get("reference") or {}).get("article_number"),
                "arabic_text": item.get("arabic_content"),
                "english_text": item.get("english_content"),
                "score": item.get("score"),
            }
            for item in answer.get("evidence", [])
        ],
        "warnings": answer.get("warnings", []),
        "errors": [f"{e.get('code')}: {e.get('message')}" for e in answer.get("errors", [])],
    }


def caveats(result: dict[str, Any]) -> list[str]:
    """Everything the backend flagged as a limitation, in one list for the user interface."""
    items = list(result.get("warnings", []))
    analysis = contract_analysis(result)
    if analysis:
        items += analysis.get("warnings", [])
    cv = cv_analysis(result)
    if cv:
        items += cv.get("warnings", [])
    comparison = result.get("comparison")
    if comparison:
        items += comparison.get("warnings", [])
        items += (comparison.get("recommendation") or {}).get("caveats", [])
    benchmark = result.get("salary_benchmark") or (contract_analysis(result) or {}).get("salary_benchmark")
    if benchmark:
        items += benchmark.get("limitations", [])
    return list(dict.fromkeys(item for item in items if item))
