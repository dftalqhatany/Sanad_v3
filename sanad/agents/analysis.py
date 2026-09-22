"""Analysis Agent: one entry point for contract analysis, CV analysis, or both.

This is a thin facade over ContractAnalysisAgent and CvAnalysisAgent. It does not route requests,
compare contracts or orchestrate other agents (those are later phases). A CV is compared with a job
only when a TargetJob is given, or when the caller explicitly asks to use the contract's job title.
"""

from __future__ import annotations

from agents.contract_analysis import ContractAnalysisAgent
from agents.cv_analysis import CvAnalysisAgent
from agents.interpretation import LLMClient
from agents.regulatory import RegulatoryEvidenceSource
from config import AnalysisSettings, SanadSettings
from models.analysis import AnalysisBundle, AnalysisStatus, SalaryBenchmark, SalaryQuery, TargetJob
from models.extraction import ContractExtraction, CvExtraction, FieldStatus

_USABLE = (AnalysisStatus.SUCCESS, AnalysisStatus.PARTIAL)
_SEVERITY = (AnalysisStatus.ANALYSIS_ERROR, AnalysisStatus.RAG_ERROR, AnalysisStatus.INVALID_INPUT,
             AnalysisStatus.INSUFFICIENT_EVIDENCE, AnalysisStatus.PARTIAL, AnalysisStatus.SUCCESS)


def target_job_from_contract(contract: ContractExtraction) -> TargetJob | None:
    """The contract's job title as a target job, only when it was explicitly found in the contract."""
    field = contract.job_title
    if field.status is not FieldStatus.FOUND or not field.value:
        return None
    return TargetJob(title=field.value, source="contract_job_title",
                     source_span=field.sources[0] if field.sources else None)


def combined_status(statuses: list[AnalysisStatus]) -> AnalysisStatus:
    """Same status -> that status; one usable and one not -> partial; otherwise the most severe."""
    unique = set(statuses)
    if len(unique) == 1:
        return statuses[0]
    if unique & set(_USABLE):
        return AnalysisStatus.PARTIAL
    return min(unique, key=_SEVERITY.index)


class AnalysisAgent:
    def __init__(self, contract_agent: ContractAnalysisAgent | None = None,
                 cv_agent: CvAnalysisAgent | None = None) -> None:
        self.contract_agent = contract_agent
        self.cv_agent = cv_agent or CvAnalysisAgent()

    @classmethod
    def from_settings(cls, settings: SanadSettings | None = None, analysis_settings: AnalysisSettings | None = None, *,
                      evidence_source: RegulatoryEvidenceSource | None = None, llm_client: LLMClient | None = None,
                      **salary_options) -> "AnalysisAgent":
        contract_agent = ContractAnalysisAgent.from_settings(settings, analysis_settings,
                                                             evidence_source=evidence_source, llm_client=llm_client,
                                                             **salary_options)
        return cls(contract_agent, CvAnalysisAgent())

    def benchmark_salary(self, contract: ContractExtraction, context: SalaryQuery | None = None) -> SalaryBenchmark:
        """Delegates to the contract agent's salary benchmarking interface; no market data is invented."""
        if self.contract_agent is None:
            raise ValueError("salary benchmarking requires a ContractAnalysisAgent")
        return self.contract_agent.salary_benchmark(contract, context)

    def analyze(self, *, contract: ContractExtraction | None = None, cv: CvExtraction | None = None,
                target_job: TargetJob | None = None, use_contract_job_title: bool = False) -> AnalysisBundle:
        if contract is None and cv is None:
            raise ValueError("provide a contract extraction, a CV extraction, or both")
        if target_job is not None and use_contract_job_title:
            raise ValueError("give either target_job or use_contract_job_title, not both")

        contract_result = None
        if contract is not None:
            if self.contract_agent is None:
                raise ValueError("contract analysis requires a ContractAnalysisAgent (with the regulatory RAG adapter)")
            contract_result = self.contract_agent.analyze(contract)

        cv_result = None
        if cv is not None:
            if use_contract_job_title:
                if not isinstance(contract, ContractExtraction):
                    raise ValueError("use_contract_job_title requires a contract extraction")
                target_job = target_job_from_contract(contract)
            cv_result = self.cv_agent.analyze(cv, target_job)
            if use_contract_job_title and target_job is None:
                cv_result.warnings.append("The contract job title was not found, so the CV was not compared with a job.")

        statuses = [result.status for result in (contract_result, cv_result) if result is not None]
        return AnalysisBundle(status=combined_status(statuses), contract_analysis=contract_result, cv_analysis=cv_result)
