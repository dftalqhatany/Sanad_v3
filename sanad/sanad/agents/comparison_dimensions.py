"""Comparison dimensions built from individual contract analyses (never from raw documents).

Each dimension reads the DocumentFact values that the Analysis Agent attached to its findings and
normalises them only by exact arithmetic (annual pay / 12, weeks x 7, years x 12). Values written in
different units, currencies, pay periods or day qualifiers are reported as not comparable rather
than converted by assumption. A ranked dimension names leading contracts only when every contract
has a value in the same unit.
"""

from __future__ import annotations

from collections.abc import Callable

from sanad.agents.shared import label as field_label
from sanad.models.analysis import AnalysisFinding, AnalysisStatus, CompatibilityStatus, FindingStatus
from sanad.models.comparison import Category, ComparedContract, ContractValue, DimensionComparison, Ranking
from sanad.models.extraction import FieldStatus

UNUSABLE_ANALYSIS = (AnalysisStatus.INVALID_INPUT, AnalysisStatus.ANALYSIS_ERROR)
RANKED_DIMENSIONS = ("basic_salary", "stated_pay", "compliance", "cv_compatibility", "weekly_working_hours",
                     "daily_working_hours", "working_days_per_week", "annual_leave", "probation_period")
COMPATIBILITY_LEVEL = {"required_requirements_explicitly_met": 3, "partially_evidenced": 2, "gaps_identified": 1}


# --------------------------------------------------------------------------- formatting / provenance
def number(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def _unique(items) -> list:
    return list(dict.fromkeys(item for item in items if item not in (None, "")))


def provenance(*findings: AnalysisFinding) -> dict:
    fields, ids, raws, texts, pages = [], [], [], [], []
    for finding in findings:
        fact = finding.contract_fact
        spans = fact.sources or [candidate.source for candidate in fact.candidates]
        fields.append(finding.field)
        ids.append(finding.finding_id)
        raws += [fact.raw_value] if fact.raw_value else [candidate.raw_value for candidate in fact.candidates]
        texts += [span.text for span in spans]
        pages += [span.page_number for span in spans]
    return dict(fields_used=_unique(fields), finding_ids=_unique(ids), raw_values=_unique(raws),
                source_texts=_unique(texts), page_numbers=_unique(pages))


def finding_for(entry: ComparedContract, field: str) -> AnalysisFinding | None:
    return next((f for f in entry.contract_analysis.findings if f.field == field), None)


def unavailable(entry: ComparedContract, fields: list[str]) -> ContractValue:
    return ContractValue(contract_id=entry.contract_id, status="analysis_unavailable", fields_used=fields,
                         display=f"No individual analysis is available (status: {entry.analysis_status.value}).")


def missing_value(entry: ComparedContract, finding: AnalysisFinding) -> ContractValue:
    """A value for a field that the analysis did not find, found ambiguously, or could not extract."""
    fact = finding.contract_fact
    name = field_label(finding.field)
    if fact.extraction_status is FieldStatus.NOT_FOUND:
        status, display = "not_found", f"The {name} was not found in the contract."
    elif fact.extraction_status is FieldStatus.AMBIGUOUS:
        written = "; ".join(f"'{c.raw_value}'" for c in fact.candidates) or "an unreadable value"
        status, display = "ambiguous", f"The {name} is ambiguous in the contract ({written})."
    else:
        status, display = "not_extracted", f"The {name} could not be extracted from the contract."
    return ContractValue(contract_id=entry.contract_id, status=status, display=display, **provenance(finding))


def found(entry: ComparedContract, field: str) -> tuple[AnalysisFinding | None, ContractValue | None]:
    """(finding, None) when the field was found; otherwise (finding or None, the value explaining why not)."""
    if entry.analysis_status in UNUSABLE_ANALYSIS:
        return None, unavailable(entry, [field])
    finding = finding_for(entry, field)
    if finding is None:
        return None, unavailable(entry, [field])
    if finding.contract_fact.extraction_status is not FieldStatus.FOUND:
        return finding, missing_value(entry, finding)
    return finding, None


def not_comparable(entry: ComparedContract, finding: AnalysisFinding, reason: str, *extra: AnalysisFinding) -> ContractValue:
    return ContractValue(contract_id=entry.contract_id, status="not_comparable", display=reason,
                         **provenance(finding, *extra))


# --------------------------------------------------------------------------- normalisation
def monthly_money(value: dict | None) -> tuple[float | None, str | None, str | None, str | None]:
    """(monthly amount, currency, problem, note). Only an explicitly written period is used."""
    value = value or {}
    amount, currency, period = value.get("amount"), value.get("currency"), value.get("period")
    if amount is None:
        percentage = value.get("percentage")
        if percentage is not None:
            basis = f" {value.get('percentage_basis')}" if value.get("percentage_basis") else ""
            return None, None, f"is written as a percentage ({number(percentage)}%{basis}), not as an amount", None
        return None, None, "has no written amount", None
    if not currency:
        return None, None, "has no written currency", None
    if period == "monthly":
        return float(amount), currency, None, None
    if period == "annual":
        return float(amount) / 12, currency, None, f"converted from {number(amount)} {currency} per year (/ 12)"
    return None, None, "has no written pay period (monthly or annual)", None


def duration(value: dict) -> tuple[float, str, str | None]:
    """(count, unit label, note): days and weeks -> days; months and years -> months; qualifier kept."""
    count, unit, qualifier = float(value["count"]), value["unit"], value.get("qualifier")
    note = None
    if unit == "week":
        count, base, note = count * 7, "days", f"converted from {number(value['count'])} weeks (x 7)"
    elif unit == "year":
        count, base, note = count * 12, "months", f"converted from {number(value['count'])} years (x 12)"
    else:
        base = "days" if unit == "day" else "months"
    return count, f"{qualifier} {base}" if qualifier else base, note


# --------------------------------------------------------------------------- dimension assembly
def build_dimension(name: str, label: str, category: Category, ranking: Ranking, rationale: str,
                    values: list[ContractValue], names: dict[str, str]) -> DimensionComparison:
    available = all(v.status == "available" for v in values)
    units = {v.unit for v in values}
    ranked = ranking != "not_ranked"
    comparable = available and len(units) == 1 and (not ranked or all(v.value is not None for v in values))
    best: list[str] = []
    all_equal = False
    if ranked and comparable:
        numbers = [v.value for v in values]
        target = max(numbers) if ranking == "higher_is_better" else min(numbers)
        best = [v.contract_id for v in values if v.value == target]
        all_equal = len(best) == len(values)

    reasons = [f"{names[v.contract_id]}: {v.display.rstrip('.')}" for v in values if v.status != "available"]
    if not ranked:
        explanation = "Shown for information; this dimension is not ranked."
        if reasons:
            explanation += " Not available for " + "; ".join(reasons) + "."
    elif not comparable:
        if available and len(units) > 1:
            reasons.append("the values are written in different units (" + ", ".join(sorted(u or "none" for u in units)) + ")")
        explanation = "Not ranked because " + "; ".join(reasons) + "."
    elif all_equal:
        explanation = f"All contracts are equal ({values[0].display})."
    else:
        leaders = ", ".join(names[c] for c in best)
        others = "; ".join(f"{names[v.contract_id]}: {v.display}" for v in values if v.contract_id not in best)
        lead_display = next(v.display for v in values if v.contract_id == best[0])
        explanation = f"{leaders} {'leads' if len(best) == 1 else 'lead'} ({lead_display}). Others: {others}."
    return DimensionComparison(dimension=name, label=label, category=category, ranking=ranking, rationale=rationale,
                               values=values, comparable=comparable, all_equal=all_equal, best_contract_ids=best,
                               explanation=explanation)


# --------------------------------------------------------------------------- value readers
def basic_salary(entry: ComparedContract) -> ContractValue:
    finding, missing = found(entry, "salary")
    if missing:
        return missing
    amount, currency, problem, note = monthly_money(finding.contract_fact.value)
    if problem:
        return not_comparable(entry, finding, f"The basic salary {problem}.")
    return ContractValue(contract_id=entry.contract_id, status="available", value=amount, unit=f"{currency}/month",
                         display=f"{number(amount)} {currency}/month", notes=[note] if note else [],
                         **provenance(finding))


def stated_pay(entry: ComparedContract) -> ContractValue:
    total, _ = found(entry, "total_salary")
    if total is not None and total.contract_fact.extraction_status is FieldStatus.FOUND:
        amount, currency, problem, note = monthly_money(total.contract_fact.value)
        if problem:
            return not_comparable(entry, total, f"The total salary {problem}.")
        return ContractValue(contract_id=entry.contract_id, status="available", value=amount, unit=f"{currency}/month",
                             display=f"{number(amount)} {currency}/month (total salary written in the contract)",
                             notes=[note] if note else [], **provenance(total))

    salary, missing = found(entry, "salary")
    if missing:
        return missing.model_copy(update={"display": missing.display + " The stated pay cannot be added up."})
    amount, currency, problem, note = monthly_money(salary.contract_fact.value)
    if problem:
        return not_comparable(entry, salary, f"The basic salary {problem}, so the stated pay is not added up.")
    parts, used, notes, skipped = [f"basic {number(amount)}"], [salary], [note] if note else [], []
    total_amount = amount
    for field in ("housing_allowance", "transportation_allowance", "other_allowances"):
        finding = finding_for(entry, field)
        if finding is None or finding.contract_fact.extraction_status is FieldStatus.NOT_FOUND:
            skipped.append(field_label(field))
            continue
        if finding.contract_fact.extraction_status is not FieldStatus.FOUND:
            return not_comparable(entry, salary, f"The {field_label(field)} is ambiguous or unreadable, so the "
                                                 "stated pay is not added up.", finding)
        items = finding.contract_fact.value if isinstance(finding.contract_fact.value, list) else [finding.contract_fact.value]
        for item in items:
            name = item.get("name") or field_label(field)
            item_amount, item_currency, item_problem, item_note = monthly_money(item)
            if item_problem:
                return not_comparable(entry, salary, f"The {name} {item_problem}, so the stated pay is not added up.", finding)
            if item_currency != currency:
                return not_comparable(entry, salary, f"The {name} is in {item_currency} while the basic salary is in "
                                                     f"{currency}; the stated pay is not added up.", finding)
            total_amount += item_amount
            parts.append(f"{name} {number(item_amount)}")
            notes += [item_note] if item_note else []
        used.append(finding)
    if skipped:
        notes.append("Not found in the contract, so not included: " + ", ".join(skipped) + ".")
    return ContractValue(contract_id=entry.contract_id, status="available", value=total_amount, unit=f"{currency}/month",
                         display=f"{number(total_amount)} {currency}/month (" + " + ".join(parts) + ")",
                         notes=notes, **provenance(*used))


def salary_vs_market(entry: ComparedContract) -> ContractValue:
    benchmark = entry.contract_analysis.salary_benchmark
    if entry.analysis_status in UNUSABLE_ANALYSIS or benchmark is None:
        return unavailable(entry, ["salary_benchmark"])
    if benchmark.status != "success":
        return ContractValue(contract_id=entry.contract_id, status="not_available", fields_used=["salary_benchmark"],
                             display=f"No market salary data ({benchmark.status}): {benchmark.message}",
                             notes=list(benchmark.limitations))
    low, high = number(benchmark.market_min), number(benchmark.market_max)
    sources = "; ".join(f"{s.name} <{s.url}>" for s in benchmark.sources)
    salary_finding = finding_for(entry, "salary")
    position = ""
    contract_salary = benchmark.contract_salary
    if contract_salary is not None and contract_salary.amount is not None:
        amount = contract_salary.amount
        where = ("below" if amount < benchmark.market_min else
                 "above" if amount > benchmark.market_max else "inside")
        position = f" The contract salary ({number(amount)} {contract_salary.currency or ''}".rstrip() +                    f") is {where} that range."
    basis = {"base_salary": "base salary", "total_compensation": "total compensation",
             "unspecified": "pay of an unstated kind"}[benchmark.basis or "unspecified"]
    return ContractValue(
        contract_id=entry.contract_id, status="available", fields_used=["salary_benchmark", "salary"],
        unit=f"{benchmark.currency}/{benchmark.period}" if benchmark.currency else None,
        finding_ids=[salary_finding.finding_id] if salary_finding else [],
        display=f"Market {basis} {low} - {high} {benchmark.currency or ''} per {benchmark.period or 'month'} "
                f"({benchmark.data_quality} sources: {sources}).{position}",
        notes=list(benchmark.limitations))


def compliance(entry: ComparedContract) -> ContractValue:
    analysis = entry.contract_analysis
    fields = ["regulatory_findings"]
    if entry.analysis_status in UNUSABLE_ANALYSIS:
        return unavailable(entry, fields)
    failed = [f for f in analysis.findings if f.status is FindingStatus.ERROR]
    if entry.analysis_status is AnalysisStatus.RAG_ERROR or failed:
        codes = ", ".join(sorted({e.code for e in analysis.errors})) or "unknown error"
        return ContractValue(contract_id=entry.contract_id, status="not_comparable", fields_used=fields,
                             finding_ids=[f.finding_id for f in failed],
                             display=f"The regulatory checks could not be completed ({codes}).")
    counts = {status: [f for f in analysis.findings if f.status is status] for status in FindingStatus}
    non_compliant = counts[FindingStatus.NON_COMPLIANT]
    notes = []
    if analysis.interpreter == "none":
        notes.append("No automated interpretation was configured, so no finding could be labelled non-compliant; "
                     "0 does not mean the contract is compliant.")
    display = (f"{len(non_compliant)} non-compliant (evidence-grounded), "
               f"{len(counts[FindingStatus.COMPLIANT])} compliant, "
               f"{len([f for f in counts[FindingStatus.REQUIRES_REVIEW] if f.topic])} require review, "
               f"{len(counts[FindingStatus.INSUFFICIENT_EVIDENCE])} with insufficient evidence")
    return ContractValue(contract_id=entry.contract_id, status="available", value=float(len(non_compliant)),
                         unit="non-compliant findings", display=display, fields_used=fields,
                         finding_ids=[f.finding_id for f in non_compliant],
                         source_texts=_unique(s.text for f in non_compliant for s in f.contract_fact.sources),
                         notes=notes)


def cv_compatibility(entry: ComparedContract) -> ContractValue:
    cv = entry.cv_analysis
    job_title, _ = found(entry, "job_title")
    title_provenance = provenance(job_title) if job_title is not None else {"fields_used": ["job_title"]}
    if cv is None or cv.status not in (AnalysisStatus.SUCCESS, AnalysisStatus.PARTIAL):
        status = cv.status.value if cv is not None else "not run"
        return ContractValue(contract_id=entry.contract_id, status="analysis_unavailable",
                             display=f"No usable CV analysis (status: {status}).", **title_provenance)
    compatibility = cv.job_compatibility
    if compatibility is None:
        return ContractValue(contract_id=entry.contract_id, status="not_found", **title_provenance,
                             display="The contract job title was not found, so the CV was not compared with this job.")
    counts = ", ".join(f"{n} {s.replace('_', ' ')}" for s, n in compatibility.status_counts.items())
    level = COMPATIBILITY_LEVEL.get(compatibility.overall)
    evidence = [e for r in compatibility.requirements for e in r.cv_evidence]
    base = dict(contract_id=entry.contract_id, fields_used=[*title_provenance["fields_used"],
                *_unique(f"cv.{e.cv_field}" for e in evidence)],
                finding_ids=title_provenance.get("finding_ids", []), raw_values=title_provenance.get("raw_values", []),
                source_texts=_unique([*title_provenance.get("source_texts", []), *(e.source.text for e in evidence)]),
                page_numbers=title_provenance.get("page_numbers", []))
    display = f"{compatibility.overall.replace('_', ' ')} for '{compatibility.target_job.title}' ({counts})"
    if level is None:
        return ContractValue(status="not_comparable", display=display + "; the CV does not give enough information.",
                             **base)
    return ContractValue(status="available", value=float(level), display=display,
                         unit="compatibility level (3 explicitly met, 2 partially evidenced, 1 gaps identified)",
                         **base)


def working_hours(key: str, unit: str) -> Callable[[ComparedContract], ContractValue]:
    def read(entry: ComparedContract) -> ContractValue:
        finding, missing = found(entry, "working_hours")
        if missing:
            return missing
        hours = (finding.contract_fact.value or {}).get(key)
        if hours is None:
            return not_comparable(entry, finding, f"Working hours are written ('{finding.contract_fact.raw_value}'), "
                                                  f"but not as hours {unit}.")
        return ContractValue(contract_id=entry.contract_id, status="available", value=float(hours), unit=f"hours {unit}",
                             display=f"{number(hours)} hours {unit}", **provenance(finding))
    return read


def working_days(entry: ComparedContract) -> ContractValue:
    finding, missing = found(entry, "working_days")
    if missing:
        return missing
    value = finding.contract_fact.value or {}
    if value.get("days_per_week") is not None:
        days, display = value["days_per_week"], f"{value['days_per_week']} days per week"
    elif value.get("days"):
        days = len(value["days"])
        display = f"{days} days per week ({', '.join(d.capitalize() for d in value['days'])})"
    else:
        return not_comparable(entry, finding, "Working days are written, but not as a number of days per week.")
    return ContractValue(contract_id=entry.contract_id, status="available", value=float(days), unit="days/week",
                         display=display, **provenance(finding))


def duration_reader(field: str, *, ranked: bool) -> Callable[[ComparedContract], ContractValue]:
    def read(entry: ComparedContract) -> ContractValue:
        finding, missing = found(entry, field)
        if missing:
            return missing
        count, unit, note = duration(finding.contract_fact.value)
        return ContractValue(contract_id=entry.contract_id, status="available", value=count if ranked else None,
                             unit=unit, display=f"{number(count)} {unit}", notes=[note] if note else [],
                             **provenance(finding))
    return read


def allowances(entry: ComparedContract) -> ContractValue:
    fields = ["housing_allowance", "transportation_allowance", "other_allowances"]
    if entry.analysis_status in UNUSABLE_ANALYSIS:
        return unavailable(entry, fields)
    findings = [f for f in (finding_for(entry, name) for name in fields) if f is not None]
    present = [f for f in findings if f.contract_fact.extraction_status is not FieldStatus.NOT_FOUND]
    if not present:
        return ContractValue(contract_id=entry.contract_id, status="not_found", fields_used=fields,
                             display="No allowances were found in the contract.")
    parts = []
    for finding in present:
        fact = finding.contract_fact
        if fact.extraction_status is not FieldStatus.FOUND:
            parts.append(f"{field_label(finding.field)}: {fact.extraction_status.value.replace('_', ' ')}")
            continue
        for item in fact.value if isinstance(fact.value, list) else [fact.value]:
            name = item.get("name") or field_label(finding.field)
            if item.get("amount") is not None:
                period = {"monthly": "/month", "annual": "/year"}.get(item.get("period"), " (period not written)")
                parts.append(f"{name}: {number(item['amount'])} {item.get('currency') or ''}{period}".replace("  ", " "))
            elif item.get("percentage") is not None:
                parts.append(f"{name}: {number(item['percentage'])}% {item.get('percentage_basis') or ''}".strip())
            else:
                parts.append(f"{name}: {fact.raw_value}")
    return ContractValue(contract_id=entry.contract_id, status="available", display="; ".join(parts),
                         **provenance(*present))


def contract_term(entry: ComparedContract) -> ContractValue:
    if entry.analysis_status in UNUSABLE_ANALYSIS:
        return unavailable(entry, ["contract_type", "contract_duration", "end_date"])
    findings = {name: finding_for(entry, name) for name in ("contract_type", "contract_duration", "end_date")}
    present = [f for f in findings.values() if f is not None and f.contract_fact.extraction_status is FieldStatus.FOUND]
    if not present:
        return ContractValue(contract_id=entry.contract_id, status="not_found",
                             fields_used=list(findings), display="The contract type and duration were not found.")
    parts = []
    for finding in present:
        value = finding.contract_fact.value
        if finding.field == "contract_type":
            parts.append(f"type: {(value.get('normalized') or value.get('raw')).replace('_', ' ')}")
        elif finding.field == "contract_duration":
            count, unit, _ = duration(value)
            parts.append(f"duration: {number(count)} {unit}")
        else:
            parts.append(f"ends: {value.get('iso_date') or value.get('raw')}")
    return ContractValue(contract_id=entry.contract_id, status="available", display="; ".join(parts),
                         **provenance(*present))


def text_field(field: str) -> Callable[[ComparedContract], ContractValue]:
    def read(entry: ComparedContract) -> ContractValue:
        finding, missing = found(entry, field)
        if missing:
            return missing
        return ContractValue(contract_id=entry.contract_id, status="available", display=str(finding.contract_fact.value),
                             **provenance(finding))
    return read


# --------------------------------------------------------------------------- dimension catalogue
EMPLOYEE_VIEW = "From the employee's point of view"
DIMENSIONS: tuple[tuple[str, str, Category, Ranking, str, Callable[[ComparedContract], ContractValue]], ...] = (
    ("basic_salary", "Basic salary", "salary", "higher_is_better",
     f"{EMPLOYEE_VIEW}, a higher basic salary is better. Only amounts with a written currency and pay period are compared.",
     basic_salary),
    ("stated_pay", "Stated monthly pay (total salary, or basic salary plus allowances with written amounts)", "salary",
     "higher_is_better",
     f"{EMPLOYEE_VIEW}, higher pay is better. A written total salary is used when present; percentages are never "
     "converted into amounts.", stated_pay),
    ("salary_vs_market", "Salary compared with market data", "salary", "not_ranked",
     "Shown only when a salary benchmark with cited sources is available; market data is never estimated.",
     salary_vs_market),
    ("compliance", "Evidence-grounded non-compliant findings", "compliance", "lower_is_better",
     "Fewer findings that the analysis labelled non-compliant, with verified regulatory evidence, is better. Findings "
     "that require review or lack evidence are listed as risks, not counted as violations.", compliance),
    ("cv_compatibility", "CV compatibility with the contract job title", "cv_compatibility", "higher_is_better",
     "A job whose stated requirements are explicitly met by the CV ranks higher. Based only on explicit CV statements.",
     cv_compatibility),
    ("weekly_working_hours", "Weekly working hours", "working_conditions", "lower_is_better",
     f"{EMPLOYEE_VIEW}, fewer written working hours per week is better.", working_hours("hours_per_week", "per week")),
    ("daily_working_hours", "Daily working hours", "working_conditions", "lower_is_better",
     f"{EMPLOYEE_VIEW}, fewer written working hours per day is better.", working_hours("hours_per_day", "per day")),
    ("working_days_per_week", "Working days per week", "working_conditions", "lower_is_better",
     f"{EMPLOYEE_VIEW}, fewer working days per week is better.", working_days),
    ("annual_leave", "Annual leave", "benefits", "higher_is_better",
     f"{EMPLOYEE_VIEW}, more annual leave is better. Working days and calendar days are not mixed.",
     duration_reader("annual_leave", ranked=True)),
    ("allowances", "Allowances", "benefits", "not_ranked",
     "Listed as written; their amounts are included in the stated pay where possible.", allowances),
    ("probation_period", "Probation period", "contract_terms", "lower_is_better",
     f"{EMPLOYEE_VIEW}, a shorter probation period is better. Days and months are not converted into each other.",
     duration_reader("probation_period", ranked=True)),
    ("notice_period", "Notice period", "contract_terms", "not_ranked",
     "Not ranked: a longer notice period protects the employee's job but also binds the employee.",
     duration_reader("notice_period", ranked=False)),
    ("contract_term", "Contract type and duration", "contract_terms", "not_ranked",
     "Not ranked: whether a fixed-term or indefinite contract is preferable depends on the person.", contract_term),
    ("work_location", "Work location", "contract_terms", "not_ranked",
     "Not ranked: location preference is personal.", text_field("work_location")),
)


def build_dimensions(contracts: list[ComparedContract], *, include_cv: bool) -> list[DimensionComparison]:
    names = {c.contract_id: c.contract_id for c in contracts}
    return [
        build_dimension(name, label, category, ranking, rationale, [reader(entry) for entry in contracts], names)
        for name, label, category, ranking, rationale, reader in DIMENSIONS
        if include_cv or name != "cv_compatibility"
    ]


def compatibility_gaps(entry: ComparedContract) -> list:
    compatibility = entry.cv_analysis.job_compatibility if entry.cv_analysis else None
    if compatibility is None:
        return []
    return [r for r in compatibility.requirements if r.status is CompatibilityStatus.MISSING_REQUIREMENT]
