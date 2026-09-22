"""Comparison & Recommendation Agent (Phase 5).

Workflow:
  contracts A, B, C (+ optional CV)
    -> AnalysisAgent.analyze() once per contract   (never bypassed; no raw-document comparison)
    -> individual analyses (regulatory findings, CV compatibility with that contract's job title, salary benchmark)
    -> comparison dimensions built from the analyses, with provenance
    -> risks per contract
    -> deterministic recommendation with decision factors (or an explicit 'no clear preference')
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from agents.analysis import AnalysisAgent
from agents.comparison_dimensions import (
    RANKED_DIMENSIONS,
    UNUSABLE_ANALYSIS,
    build_dimensions,
    compatibility_gaps,
)
from agents.errors import ComparisonErrorCode, analysis_error
from agents.recommendation import recommend
from agents.shared import label as field_label
from config import AnalysisSettings, SanadSettings
from models.analysis import AnalysisStatus, FindingStatus
from models.common import ErrorInfo
from models.comparison import (
    ComparedContract,
    ComparisonSections,
    ContractComparisonResult,
    ContractRisk,
    DimensionComparison,
)
from models.extraction import ContractExtraction, CvExtraction, FieldStatus

logger = logging.getLogger(f"sanad.{__name__}")  # one "sanad" logging namespace, as before the flattening

MIN_CONTRACTS = 2
DEFAULT_MAX_CONTRACTS = 5
KEY_TERMS = ("salary", "working_hours", "annual_leave", "probation_period", "notice_period", "contract_type")


class ContractComparisonAgent:
    def __init__(self, analysis_agent: AnalysisAgent, *, max_contracts: int = DEFAULT_MAX_CONTRACTS) -> None:
        if analysis_agent.contract_agent is None:
            raise ValueError("the comparison agent needs an AnalysisAgent with a ContractAnalysisAgent")
        if max_contracts < MIN_CONTRACTS:
            raise ValueError(f"max_contracts must be at least {MIN_CONTRACTS}")
        self.analysis_agent = analysis_agent
        self.max_contracts = max_contracts

    @classmethod
    def from_settings(cls, settings: SanadSettings | None = None, analysis_settings: AnalysisSettings | None = None, *,
                      max_contracts: int = DEFAULT_MAX_CONTRACTS, **analysis_options) -> "ContractComparisonAgent":
        """Builds the existing AnalysisAgent (options such as evidence_source / llm_client are passed to it)."""
        return cls(AnalysisAgent.from_settings(settings, analysis_settings, **analysis_options), max_contracts=max_contracts)

    # ------------------------------------------------------------------ public
    def compare(self, contracts: Sequence[ContractExtraction], cv: CvExtraction | None = None, *,
                priorities: Sequence[str] = (), labels: Sequence[str] | None = None) -> ContractComparisonResult:
        errors = self._validate(contracts, cv, priorities, labels)
        if errors:
            return ContractComparisonResult(status=AnalysisStatus.INVALID_INPUT, errors=errors,
                                            summary="No comparison was performed: invalid input.")
        compared: list[ComparedContract] = []
        try:
            for position, contract in enumerate(contracts, 1):
                compared.append(self._analyse(position, contract, cv, labels))
            result = self._compare(compared, cv, list(priorities))
        except Exception as exc:  # explicit failure instead of a partial comparison presented as complete
            logger.warning("Contract comparison failed after %d analyses: %s", len(compared), type(exc).__name__)
            error = analysis_error(ComparisonErrorCode.COMPARISON_FAILED, "comparison",
                                   "The contract comparison failed unexpectedly; no comparison is reported.", exc)
            return ContractComparisonResult(status=AnalysisStatus.ANALYSIS_ERROR, contracts=compared, errors=[error],
                                            summary="The comparison could not be completed.")
        logger.info("Contract comparison of %d contracts: status=%s recommendation=%s", len(compared),
                    result.status.value, result.recommendation.status if result.recommendation else None)
        return result

    # ------------------------------------------------------------------ workflow
    def _analyse(self, position: int, contract: ContractExtraction, cv: CvExtraction | None,
                 labels: Sequence[str] | None) -> ComparedContract:
        bundle = self.analysis_agent.analyze(contract=contract, cv=cv, use_contract_job_title=cv is not None)
        analysis = bundle.contract_analysis
        return ComparedContract(
            contract_id=f"contract_{position}", label=labels[position - 1] if labels else contract.filename,
            document_id=contract.document_id, filename=contract.filename, document_status=contract.document_status,
            analysis_status=analysis.status, contract_analysis=analysis, cv_analysis=bundle.cv_analysis,
        )

    def _compare(self, compared: list[ComparedContract], cv: CvExtraction | None,
                 priorities: list[str]) -> ContractComparisonResult:
        dimensions = build_dimensions(compared, include_cv=cv is not None)
        sections = ComparisonSections()
        for dimension in dimensions:
            getattr(sections, dimension.category).append(dimension)
        risks = [risk for entry in compared for risk in self._risks(entry)]
        recommendation = recommend(compared, dimensions, priorities, risks)

        errors = [
            analysis_error(ComparisonErrorCode.CONTRACT_ANALYSIS_FAILED, "contract_analysis",
                           f"{entry.contract_id}: the individual analysis ended with status "
                           f"'{entry.analysis_status.value}'.")
            for entry in compared if entry.analysis_status in (*UNUSABLE_ANALYSIS, AnalysisStatus.RAG_ERROR)
        ]
        warnings = self._warnings(compared, cv)
        status = self._status(compared, dimensions, cv)
        return ContractComparisonResult(
            status=status, contracts=compared, comparison=sections, risks=risks, recommendation=recommendation,
            summary=self._summary(compared, dimensions, recommendation), cv_document_id=cv.document_id if cv else None,
            warnings=warnings, errors=errors,
        )

    # ------------------------------------------------------------------ helpers
    def _validate(self, contracts, cv, priorities, labels) -> list[ErrorInfo]:
        def error(code: ComparisonErrorCode, message: str) -> ErrorInfo:
            return analysis_error(code, "input", message)

        if isinstance(contracts, (str, bytes)) or not isinstance(contracts, Sequence):
            return [error(ComparisonErrorCode.INVALID_INPUT, "contracts must be a list of contract extractions.")]
        errors = []
        if len(contracts) < MIN_CONTRACTS:
            errors.append(error(ComparisonErrorCode.TOO_FEW_CONTRACTS,
                                f"At least {MIN_CONTRACTS} contracts are needed for a comparison; got {len(contracts)}."))
        if len(contracts) > self.max_contracts:
            errors.append(error(ComparisonErrorCode.TOO_MANY_CONTRACTS,
                                f"At most {self.max_contracts} contracts can be compared at once; got {len(contracts)}."))
        wrong = [str(i) for i, item in enumerate(contracts, 1) if not isinstance(item, ContractExtraction)]
        if wrong:
            errors.append(error(ComparisonErrorCode.INVALID_INPUT,
                                f"Items {', '.join(wrong)} are not contract extractions."))
        if cv is not None and not isinstance(cv, CvExtraction):
            errors.append(error(ComparisonErrorCode.INVALID_INPUT, f"cv must be a CvExtraction, got {type(cv).__name__}."))
        if isinstance(priorities, (str, bytes)):
            errors.append(error(ComparisonErrorCode.INVALID_PRIORITY, "priorities must be a list of dimension names."))
        else:
            unknown = [p for p in priorities if p not in RANKED_DIMENSIONS]
            if unknown:
                errors.append(error(ComparisonErrorCode.INVALID_PRIORITY,
                                    f"Unknown priorities: {', '.join(map(str, unknown))}. Allowed: {', '.join(RANKED_DIMENSIONS)}."))
            if len(set(priorities)) != len(priorities):
                errors.append(error(ComparisonErrorCode.INVALID_PRIORITY, "Each priority may be given only once."))
            if "cv_compatibility" in priorities and cv is None:
                errors.append(error(ComparisonErrorCode.INVALID_PRIORITY, "The 'cv_compatibility' priority needs a CV."))
        if labels is not None and (isinstance(labels, (str, bytes)) or len(labels) != len(contracts)
                                   or not all(isinstance(x, str) and x.strip() for x in labels)):
            errors.append(error(ComparisonErrorCode.INVALID_INPUT, "labels must be one non-empty string per contract."))
        return errors

    @staticmethod
    def _risks(entry: ComparedContract) -> list[ContractRisk]:
        cid = entry.contract_id
        analysis = entry.contract_analysis
        if entry.analysis_status in UNUSABLE_ANALYSIS:
            return [ContractRisk(contract_id=cid, severity="high", code="analysis_failed",
                                 message=f"The individual analysis failed ({entry.analysis_status.value}); this "
                                         "contract's terms and compliance are unknown.")]
        risks = []
        if entry.analysis_status is AnalysisStatus.RAG_ERROR:
            risks.append(ContractRisk(contract_id=cid, severity="medium", code="regulatory_check_error",
                                      message="The regulatory RAG was unavailable, so no clause was checked against "
                                              "the Saudi Labor Law."))
        for finding in analysis.findings:
            name = field_label(finding.field)
            fact = finding.contract_fact
            base = dict(contract_id=cid, field=finding.field, finding_id=finding.finding_id)
            if finding.status is FindingStatus.NON_COMPLIANT:
                citations = "; ".join(e.citation for e in finding.regulatory_evidence)
                risks.append(ContractRisk(**base, severity="high", code="non_compliant_finding",
                                          message=f"{name}: appears inconsistent with {citations}."))
            elif finding.status is FindingStatus.ERROR and entry.analysis_status is not AnalysisStatus.RAG_ERROR:
                risks.append(ContractRisk(**base, severity="medium", code="regulatory_check_error",
                                          message=f"{name}: the regulatory check could not be performed."))
            elif finding.status is FindingStatus.INSUFFICIENT_EVIDENCE:
                risks.append(ContractRisk(**base, severity="medium", code="insufficient_regulatory_evidence",
                                          message=f"{name}: no relevant regulatory evidence was retrieved."))
            elif fact.extraction_status is FieldStatus.AMBIGUOUS:
                risks.append(ContractRisk(**base, severity="medium", code="ambiguous_term",
                                          message=f"{name}: the contract states this term ambiguously."))
            elif finding.status is FindingStatus.REQUIRES_REVIEW and finding.topic is not None:
                risks.append(ContractRisk(**base, severity="info", code="requires_review",
                                          message=f"{name}: relevant regulation was retrieved and needs human review."))
            if finding.field in KEY_TERMS and fact.extraction_status is FieldStatus.NOT_FOUND:
                risks.append(ContractRisk(**base, severity="info", code="key_term_not_found",
                                          message=f"{name}: not found in the contract (this is not a finding of "
                                                  "non-compliance)."))
        for requirement in compatibility_gaps(entry):
            risks.append(ContractRisk(contract_id=cid, severity="info", code="cv_requirement_gap",
                                      field=f"cv.{requirement.requirement_type}",
                                      message=f"The CV does not mention '{requirement.requirement}' for this job."))
        return risks

    @staticmethod
    def _warnings(compared: list[ComparedContract], cv: CvExtraction | None) -> list[str]:
        warnings = []
        seen: dict[str, str] = {}
        for entry in compared:
            if entry.document_id in seen:
                warnings.append(f"{entry.contract_id} is the same document as {seen[entry.document_id]}.")
            seen.setdefault(entry.document_id, entry.contract_id)
        if cv is not None and any(e.cv_analysis is not None and e.cv_analysis.job_compatibility is None for e in compared):
            warnings.append("For some contracts the job title was not found, so the CV could not be compared with them.")
        return warnings

    @staticmethod
    def _status(compared: list[ComparedContract], dimensions: list[DimensionComparison],
                cv: CvExtraction | None) -> AnalysisStatus:
        statuses = [entry.analysis_status for entry in compared]
        if all(s in UNUSABLE_ANALYSIS for s in statuses):
            return AnalysisStatus.ANALYSIS_ERROR if AnalysisStatus.ANALYSIS_ERROR in statuses else AnalysisStatus.INVALID_INPUT
        if all(s is AnalysisStatus.RAG_ERROR for s in statuses):
            return AnalysisStatus.RAG_ERROR
        cv_usable = cv is None or all(e.cv_analysis is not None and e.cv_analysis.status is AnalysisStatus.SUCCESS
                                      for e in compared)
        if any(s is not AnalysisStatus.SUCCESS for s in statuses) or not cv_usable:
            return AnalysisStatus.PARTIAL
        if not any(d.ranking != "not_ranked" and d.comparable for d in dimensions):
            return AnalysisStatus.INSUFFICIENT_EVIDENCE
        return AnalysisStatus.SUCCESS

    @staticmethod
    def _summary(compared, dimensions, recommendation) -> str:
        ranked = [d for d in dimensions if d.ranking != "not_ranked"]
        comparable = [d for d in ranked if d.comparable]
        text = (f"{len(compared)} contracts were each analysed individually by the Analysis Agent and then compared. "
                f"{len(comparable)} of {len(ranked)} ranked dimensions could be compared across all contracts.")
        if recommendation.status == "preferred_contract":
            text += f" Preferred: {recommendation.preferred_contract_id} ({recommendation.method})."
        elif recommendation.status == "no_clear_preference":
            text += " No single contract is clearly preferable."
        else:
            text += " No recommendation could be made."
        return text
