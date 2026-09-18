"""Sanad Orchestrator (Phase 6): the single entry point.

    SanadRequest (question and/or uploaded PDF/DOCX)
        -> DocumentIntake        (existing parsers + existing extraction)
        -> route_request()       (deterministic table in sanad/orchestrator/routing.py)
        -> exactly one existing component:
             RegulatoryRAGAdapter        regulatory question
             AnalysisAgent               one contract (with or without a CV), or a CV on its own
             ContractComparisonAgent     2..5 contracts (with or without a CV)
             salary benchmark interface  salary benchmarking
        -> OrchestratorResult carrying that component's structured result unchanged

The Orchestrator performs no analysis, retrieval, parsing or ranking of its own, and never calls the
legacy RAG, an LLM or a parser where an existing abstraction exists.
"""

from __future__ import annotations

import logging
from typing import Protocol

from sanad.agents.analysis import AnalysisAgent
from sanad.agents.comparison import ContractComparisonAgent
from sanad.config import AnalysisSettings, DocumentProcessingSettings, SanadSettings
from sanad.models.analysis import AnalysisStatus
from sanad.models.common import ErrorInfo, ResultStatus
from sanad.models.orchestration import (
    DocumentRole,
    OrchestratorResult,
    Route,
    RoutingDecision,
    SanadRequest,
    TaskHint,
)
from sanad.models.regulatory import RegulatoryAnswerResult, RegulatoryEvidenceResult, RegulatoryQuery
from sanad.orchestrator.errors import OrchestratorErrorCode, orchestrator_error
from sanad.orchestrator.intake import DocumentIntake, IntakeItem
from sanad.orchestrator.routing import route_request

logger = logging.getLogger(__name__)

STATUS_FROM_RESULT = {
    ResultStatus.SUCCESS: AnalysisStatus.SUCCESS,
    ResultStatus.PARTIAL: AnalysisStatus.PARTIAL,
    ResultStatus.INSUFFICIENT_EVIDENCE: AnalysisStatus.INSUFFICIENT_EVIDENCE,
    ResultStatus.ERROR: AnalysisStatus.RAG_ERROR,
}


class RegulatoryQuestionSource(Protocol):
    """The Phase 2 adapter (the regulatory RAG agent's interface), never the legacy modules themselves."""

    def ask(self, query: RegulatoryQuery | str, api_key: str | None = None) -> RegulatoryAnswerResult: ...

    def retrieve_evidence(self, query: RegulatoryQuery | str, top_k: int | None = None) -> RegulatoryEvidenceResult: ...


class SanadOrchestrator:
    def __init__(self, analysis_agent: AnalysisAgent, comparison_agent: ContractComparisonAgent,
                 regulatory_source: RegulatoryQuestionSource, intake: DocumentIntake | None = None,
                 *, generate_answers: bool = True) -> None:
        self.analysis_agent = analysis_agent
        self.comparison_agent = comparison_agent
        self.regulatory_source = regulatory_source
        self.intake = intake or DocumentIntake()
        self.generate_answers = generate_answers

    @classmethod
    def from_settings(cls, settings: SanadSettings | None = None, analysis_settings: AnalysisSettings | None = None,
                      document_settings: DocumentProcessingSettings | None = None, *,
                      evidence_source: RegulatoryQuestionSource | None = None, **analysis_options) -> "SanadOrchestrator":
        """Wires the existing components; the regulatory adapter is built once and shared."""
        settings = settings or SanadSettings.from_env()
        if evidence_source is None:
            from sanad.rag.adapter import RegulatoryRAGAdapter  # imported lazily: nothing else loads the RAG

            evidence_source = RegulatoryRAGAdapter(settings)
        analysis_agent = AnalysisAgent.from_settings(settings, analysis_settings, evidence_source=evidence_source,
                                                     **analysis_options)
        comparison_agent = ContractComparisonAgent(analysis_agent)
        intake = DocumentIntake(settings=document_settings)
        return cls(analysis_agent, comparison_agent, evidence_source, intake,
                   generate_answers=bool(settings.openai_api_key))

    # ------------------------------------------------------------------ public
    def handle(self, request: SanadRequest | dict) -> OrchestratorResult:
        try:
            request = request if isinstance(request, SanadRequest) else SanadRequest.model_validate(request)
        except Exception as exc:
            return self._unrouted(TaskHint.AUTO, [orchestrator_error(OrchestratorErrorCode.NOTHING_TO_DO, "input",
                                                                     "The request is not a valid Sanad request.", exc)])
        try:
            return self._handle(request)
        except Exception as exc:  # a failure is reported, never hidden behind an empty success
            logger.warning("Orchestration failed: %s", type(exc).__name__)
            error = orchestrator_error(OrchestratorErrorCode.ORCHESTRATION_FAILED, "orchestration",
                                       "The request could not be completed.", exc)
            return self._unrouted(request.task, [error], summary="The request could not be completed.")

    # ------------------------------------------------------------------ workflow
    def _handle(self, request: SanadRequest) -> OrchestratorResult:
        declared_contracts = sum(1 for d in request.documents if d.role is DocumentRole.CONTRACT)
        if declared_contracts > self.comparison_agent.max_contracts:  # refuse before parsing anything
            error = orchestrator_error(OrchestratorErrorCode.TOO_MANY_CONTRACTS, "input",
                                       f"At most {self.comparison_agent.max_contracts} contracts can be handled at "
                                       f"once; {declared_contracts} were uploaded.")
            return self._unrouted(request.task, [error], contracts=declared_contracts)

        items = [self.intake.load(upload) for upload in request.documents]
        contracts = [item for item in items if item.usable and item.contract is not None]
        cvs = [item for item in items if item.usable and item.cv is not None]
        document_errors = [error for item in items for error in item.errors]

        routing = route_request(request.task, has_question=bool(request.question), contracts=len(contracts),
                                cvs=len(cvs), unusable=len(items) - len(contracts) - len(cvs),
                                max_contracts=self.comparison_agent.max_contracts)
        summaries = [item.summary for item in items]
        if routing.errors:
            return OrchestratorResult(status=AnalysisStatus.INVALID_INPUT, routing=routing.decision,
                                      documents=summaries, errors=[*routing.errors, *document_errors],
                                      warnings=routing.warnings,
                                      summary=f"No component was called: {routing.decision.reason}")

        warnings = list(routing.warnings)
        if request.question and routing.decision.route is not Route.REGULATORY_QUESTION:
            warnings.append("The question was not answered here; ask it without documents to query the labor law.")
        result = self._dispatch(routing.decision, request, contracts, cvs)
        return self._finish(result, routing.decision, summaries, document_errors, warnings)

    def _dispatch(self, decision: RoutingDecision, request: SanadRequest, contracts: list[IntakeItem],
                  cvs: list[IntakeItem]) -> dict:
        cv = cvs[0].cv if cvs else None
        route = decision.route
        if route is Route.REGULATORY_QUESTION:
            answer = self._ask(request.question)
            return {"regulatory_answer": answer, "status": STATUS_FROM_RESULT[answer.status],
                    "summary": self._question_summary(answer)}
        if route is Route.SALARY_BENCHMARK:
            benchmark = self.analysis_agent.benchmark_salary(contracts[0].contract, request.salary_query)
            status = {"success": AnalysisStatus.SUCCESS, "error": AnalysisStatus.PARTIAL}.get(
                benchmark.status, AnalysisStatus.INSUFFICIENT_EVIDENCE)
            return {"salary_benchmark": benchmark, "salary_contract_id": contracts[0].summary.document_id,
                    "status": status, "summary": f"Salary benchmarking ({benchmark.status}): {benchmark.message}"}
        if route is Route.CV_ANALYSIS:
            bundle = self.analysis_agent.analyze(cv=cv, target_job=request.target_job)
            return {"analysis": bundle, "status": bundle.status, "summary": bundle.cv_analysis.overall_summary}
        if route is Route.CONTRACT_ANALYSIS:
            bundle = self._analyse_contract(contracts[0], cv, request)
            summary = bundle.contract_analysis.overall_summary
            if bundle.cv_analysis is not None:
                summary += " " + bundle.cv_analysis.overall_summary
            return {"analysis": bundle, "status": bundle.status, "summary": summary}
        comparison = self.comparison_agent.compare(
            [item.contract for item in contracts], cv, priorities=request.priorities,
            labels=[item.summary.label or item.summary.filename for item in contracts],
        )
        return {"comparison": comparison, "status": comparison.status, "summary": comparison.summary}

    def _analyse_contract(self, item: IntakeItem, cv, request: SanadRequest):
        if request.target_job is not None:
            return self.analysis_agent.analyze(contract=item.contract, cv=cv, target_job=request.target_job)
        return self.analysis_agent.analyze(contract=item.contract, cv=cv, use_contract_job_title=cv is not None)

    def _ask(self, question: str) -> RegulatoryAnswerResult:
        """Existing adapter only: the legacy answer pipeline when a key is configured, otherwise retrieval."""
        if self.generate_answers:
            answer = self.regulatory_source.ask(question)
            if not (answer.status is ResultStatus.ERROR and answer.errors[0].code == "missing_api_key"):
                return answer
        evidence = self.regulatory_source.retrieve_evidence(question)
        return RegulatoryAnswerResult(
            **evidence.model_dump(),
            message="No answer text was generated (no OPENAI_API_KEY is configured); the retrieved articles are "
                    "returned as evidence.",
        )

    # ------------------------------------------------------------------ result assembly
    def _finish(self, result: dict, decision: RoutingDecision, summaries: list, document_errors: list[ErrorInfo],
                warnings: list[str]) -> OrchestratorResult:
        status = result.pop("status")
        unusable = [s for s in summaries if not s.usable]
        if unusable and status in (AnalysisStatus.SUCCESS, AnalysisStatus.INSUFFICIENT_EVIDENCE):
            status = AnalysisStatus.PARTIAL
        if unusable:
            warnings.append(f"{len(unusable)} uploaded document(s) could not be used: "
                            + "; ".join(f"{s.filename} ({s.document_status.value})" for s in unusable))
        errors = list(document_errors)
        if status in (AnalysisStatus.RAG_ERROR, AnalysisStatus.ANALYSIS_ERROR, AnalysisStatus.INVALID_INPUT) and not errors:
            errors.append(orchestrator_error(OrchestratorErrorCode.ROUTED_AGENT_FAILED, decision.route.value,
                                             f"{decision.agent} ended with status '{status.value}'."))
        logger.info("Orchestrated request: route=%s status=%s documents=%d", decision.route.value, status.value,
                    len(summaries))
        return OrchestratorResult(status=status, routing=decision, documents=summaries, warnings=warnings,
                                  errors=errors, **result)

    def _unrouted(self, task: TaskHint, errors: list[ErrorInfo], *, contracts: int = 0,
                  summary: str = "No component was called: the request could not be routed.") -> OrchestratorResult:
        decision = RoutingDecision(route=Route.NONE, agent="-", rule="none", reason=errors[0].message, task_hint=task,
                                   contract_count=contracts)
        status = (AnalysisStatus.ANALYSIS_ERROR if errors[0].code == OrchestratorErrorCode.ORCHESTRATION_FAILED.value
                  else AnalysisStatus.INVALID_INPUT)
        return OrchestratorResult(status=status, routing=decision, errors=errors, summary=summary)

    @staticmethod
    def _question_summary(answer: RegulatoryAnswerResult) -> str:
        if answer.status is ResultStatus.ERROR:
            return f"The existing regulatory RAG could not answer the question ({answer.errors[0].code})."
        articles = ", ".join(str(item.reference.article_number) for item in answer.evidence if item.reference.article_number)
        text = "An answer was generated from the existing Saudi Labor Law RAG." if answer.answer else \
            "No answer text was generated; the retrieved articles are returned as evidence."
        return text + (f" Articles: {articles}." if articles else " No article was retrieved.")
