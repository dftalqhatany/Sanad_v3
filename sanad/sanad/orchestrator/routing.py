"""Deterministic routing: uploaded documents (+ optional question and task hint) -> one existing component.

No language model is involved. The table below is the whole decision:

  task hint            condition                                   route                      component
  -------------------- ------------------------------------------- -------------------------- ---------------------------
  regulatory_question  a question was asked                        regulatory_question        RegulatoryRAGAdapter
  salary_benchmark     at least one usable contract                salary_benchmark           salary benchmark interface
  contract_analysis    exactly one usable contract                 contract_analysis          AnalysisAgent
  cv_analysis          at least one usable CV                      cv_analysis                AnalysisAgent
  contract_comparison  at least two usable contracts               contract_comparison        ContractComparisonAgent
  auto                 2+ usable contracts                         contract_comparison        ContractComparisonAgent
  auto                 exactly 1 usable contract                   contract_analysis          AnalysisAgent
  auto                 no contract, at least one usable CV         cv_analysis                AnalysisAgent
  auto                 no usable document, a question              regulatory_question        RegulatoryRAGAdapter
  any                  nothing usable                              none                       - (explicit error)

When uploads were unusable, a comparison falls back to a single-contract analysis (recorded as
fallback_from) instead of failing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sanad.models.common import ErrorInfo
from sanad.models.orchestration import Route, RoutingDecision, TaskHint
from sanad.orchestrator.errors import OrchestratorErrorCode, orchestrator_error

AGENTS = {
    Route.REGULATORY_QUESTION: "RegulatoryRAGAdapter",
    Route.CONTRACT_ANALYSIS: "AnalysisAgent",
    Route.CV_ANALYSIS: "AnalysisAgent",
    Route.CONTRACT_COMPARISON: "ContractComparisonAgent",
    Route.SALARY_BENCHMARK: "SalaryBenchmarkProvider",
    Route.NONE: "-",
}


@dataclass
class RoutingOutcome:
    decision: RoutingDecision
    errors: list[ErrorInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def route_request(task: TaskHint, *, has_question: bool, contracts: int, cvs: int, unusable: int = 0,
                  max_contracts: int = 5) -> RoutingOutcome:
    """Pure function: counts of USABLE documents in, one route out. Never calls an agent."""
    outcome = RoutingOutcome(decision=RoutingDecision(route=Route.NONE, agent="-", rule="none",
                                                      reason="No route was selected.", task_hint=task,
                                                      contract_count=contracts, cv_count=cvs, has_question=has_question))
    if cvs > 1:
        outcome.warnings.append(f"{cvs} CVs were uploaded; only the first one is used.")
    if contracts > max_contracts:
        return _fail(outcome, OrchestratorErrorCode.TOO_MANY_CONTRACTS,
                     f"At most {max_contracts} contracts can be handled at once; {contracts} were uploaded.")

    if task is TaskHint.REGULATORY_QUESTION:
        if not has_question:
            return _fail(outcome, OrchestratorErrorCode.QUESTION_MISSING,
                         "A regulatory question was requested, but no question was given.")
        if contracts or cvs:
            outcome.warnings.append("The uploaded documents were not used: a regulatory question was requested.")
        return _route(outcome, Route.REGULATORY_QUESTION, "task hint 'regulatory_question'",
                      "The question was sent to the existing Saudi Labor Law RAG through its adapter.")

    if task is TaskHint.SALARY_BENCHMARK:
        if not contracts:
            return _fail(outcome, OrchestratorErrorCode.NO_CONTRACT_UPLOADED,
                         "Salary benchmarking needs the employment contract whose salary should be checked.")
        if contracts > 1:
            outcome.warnings.append("Several contracts were uploaded; the first one was benchmarked.")
        return _route(outcome, Route.SALARY_BENCHMARK, "task hint 'salary_benchmark'",
                      "The contract's salary was sent to the salary benchmarking interface.")

    if task is TaskHint.CV_ANALYSIS:
        if not cvs:
            return _fail(outcome, OrchestratorErrorCode.TASK_NOT_POSSIBLE, "A CV analysis needs a usable CV.")
        return _route(outcome, Route.CV_ANALYSIS, "task hint 'cv_analysis'", "The CV was sent to the Analysis Agent.")

    if task is TaskHint.CONTRACT_ANALYSIS:
        if not contracts:
            return _fail(outcome, OrchestratorErrorCode.NO_CONTRACT_UPLOADED,
                         "A contract analysis needs a usable employment contract.")
        if contracts > 1:
            return _fail(outcome, OrchestratorErrorCode.TASK_NOT_POSSIBLE,
                         f"{contracts} contracts were uploaded; ask for a comparison, or upload one contract.")
        return _route(outcome, Route.CONTRACT_ANALYSIS, "task hint 'contract_analysis'",
                      _analysis_reason(cvs))

    if task is TaskHint.CONTRACT_COMPARISON:
        if contracts >= 2:
            return _route(outcome, Route.CONTRACT_COMPARISON, "task hint 'contract_comparison'", _comparison_reason(contracts, cvs))
        if contracts == 1 and unusable:
            return _fallback(outcome, cvs)
        return _fail(outcome, OrchestratorErrorCode.TASK_NOT_POSSIBLE,
                     f"A comparison needs at least 2 usable contracts; {contracts} were available.")

    # --- automatic routing
    if contracts >= 2:
        return _route(outcome, Route.CONTRACT_COMPARISON, "2 or more usable contracts", _comparison_reason(contracts, cvs))
    if contracts == 1:
        if unusable:
            return _fallback(outcome, cvs)
        return _route(outcome, Route.CONTRACT_ANALYSIS, "exactly 1 usable contract", _analysis_reason(cvs))
    if cvs:
        return _route(outcome, Route.CV_ANALYSIS, "a CV and no contract", "The CV was sent to the Analysis Agent.")
    if has_question:
        return _route(outcome, Route.REGULATORY_QUESTION, "a question and no usable document",
                      "The question was sent to the existing Saudi Labor Law RAG through its adapter.")
    return _fail(outcome, OrchestratorErrorCode.NOTHING_TO_DO,
                 "Nothing could be done: no usable document and no question.")


# --------------------------------------------------------------------------- helpers
def _route(outcome: RoutingOutcome, route: Route, rule: str, reason: str) -> RoutingOutcome:
    outcome.decision = outcome.decision.model_copy(update={"route": route, "agent": AGENTS[route], "rule": rule,
                                                           "reason": reason})
    return outcome


def _fallback(outcome: RoutingOutcome, cvs: int) -> RoutingOutcome:
    outcome.warnings.append("Only one contract could be used, so it was analysed on its own instead of compared.")
    result = _route(outcome, Route.CONTRACT_ANALYSIS, "1 usable contract after unusable uploads", _analysis_reason(cvs))
    result.decision = result.decision.model_copy(update={"fallback_from": Route.CONTRACT_COMPARISON})
    return result


def _fail(outcome: RoutingOutcome, code: OrchestratorErrorCode, message: str) -> RoutingOutcome:
    outcome.errors.append(orchestrator_error(code, "routing", message))
    outcome.decision = outcome.decision.model_copy(update={"reason": message})
    return outcome


def _analysis_reason(cvs: int) -> str:
    return ("The contract and the CV were sent to the Analysis Agent (regulatory analysis plus CV/job fit)." if cvs
            else "The contract was sent to the Analysis Agent for regulatory analysis.")


def _comparison_reason(contracts: int, cvs: int) -> str:
    return (f"{contracts} contracts were sent to the Comparison Agent, which analyses each one individually"
            + (" and compares the CV with each contract's job title." if cvs else " before comparing them."))
