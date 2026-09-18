"""Contract Analysis Agent.

Workflow (per contract):
  ContractExtraction (Phase 3 facts + provenance)
    -> fields that are present (found / ambiguous) and have a regulatory topic
    -> focused questions per topic
    -> RegulatoryRAGAdapter.retrieve_evidence()  (existing RAG, the only regulatory source)
    -> relevance check on the retrieved Arabic article text
    -> interpretation (none by default, or an evidence-grounded LLM)
    -> ContractAnalysisResult with explicit statuses

Rules that are never broken:
  * a field not found in the extraction is NOT_FOUND - never non-compliant, never queried
  * missing or irrelevant evidence is INSUFFICIENT_EVIDENCE - never non-compliant
  * ambiguous contract values are REQUIRES_REVIEW - never sent to an interpreter
  * an unavailable RAG is RAG_ERROR with 'error' findings - never a successful analysis
"""

from __future__ import annotations

import logging

from sanad.agents.errors import AnalysisErrorCode, InterpretationError, analysis_error
from sanad.agents.interpretation import (
    EvidenceInterpreter,
    InterpretationRequest,
    LLMClient,
    LLMEvidenceInterpreter,
    NoInterpreter,
    OpenAIChatClient,
)
from sanad.agents.regulatory import (
    CONTRACT_TOPICS,
    RegulatoryEvidenceCollector,
    RegulatoryEvidenceSource,
    RegulatoryTopic,
    TopicEvidence,
)
from sanad.agents.salary import SalaryBenchmarkProvider, UnavailableSalaryBenchmarkProvider, build_salary_provider
from sanad.agents.shared import fact_from_field, label, status_counts
from sanad.config import AnalysisSettings, SalarySettings, SanadSettings
from sanad.models.analysis import (
    AnalysisFinding,
    AnalysisStatus,
    ContractAnalysisResult,
    DocumentFact,
    FindingStatus,
    Interpretation,
    InterpretationMethod,
    SalaryBenchmark,
    SalaryQuery,
)
from sanad.models.common import ErrorInfo, ResultStatus
from sanad.models.extraction import ContractExtraction, ExtractedField, FieldStatus

logger = logging.getLogger(__name__)

ASSESSMENT_LABELS = {
    FindingStatus.COMPLIANT: "Appears consistent with the cited regulation (evidence-grounded automated interpretation)",
    FindingStatus.NON_COMPLIANT: "Appears inconsistent with the cited regulation (evidence-grounded automated interpretation)",
    FindingStatus.REQUIRES_REVIEW: "Requires human review",
    FindingStatus.INSUFFICIENT_EVIDENCE: "Insufficient regulatory evidence retrieved",
    FindingStatus.NOT_APPLICABLE: "Document fact only; no regulatory check for this field",
    FindingStatus.NOT_FOUND: "Not found in the extracted contract",
    FindingStatus.ERROR: "Regulatory check could not be performed",
}


class ContractAnalysisAgent:
    def __init__(
        self,
        evidence_source: RegulatoryEvidenceSource,
        interpreter: EvidenceInterpreter | None = None,
        salary_provider: SalaryBenchmarkProvider | None = None,
        topics: tuple[RegulatoryTopic, ...] = CONTRACT_TOPICS,
        max_evidence_per_topic: int = 6,
    ) -> None:
        self.evidence_source = evidence_source
        self.interpreter = interpreter or NoInterpreter()
        self.salary_provider = salary_provider or UnavailableSalaryBenchmarkProvider()
        self.topics = topics
        self.topic_by_field = {name: topic for topic in topics for name in topic.fields}
        self.max_evidence_per_topic = max_evidence_per_topic

    @classmethod
    def from_settings(
        cls,
        settings: SanadSettings | None = None,
        analysis_settings: AnalysisSettings | None = None,
        *,
        evidence_source: RegulatoryEvidenceSource | None = None,
        llm_client: LLMClient | None = None,
        salary_settings: SalarySettings | None = None,
        salary_provider: SalaryBenchmarkProvider | None = None,
    ) -> "ContractAnalysisAgent":
        """Existing RAG adapter + an LLM interpreter only when OPENAI_API_KEY is configured."""
        settings = settings or SanadSettings.from_env()
        analysis_settings = analysis_settings or AnalysisSettings.from_env()
        if evidence_source is None:
            from sanad.rag.adapter import RegulatoryRAGAdapter

            evidence_source = RegulatoryRAGAdapter(settings)
        if llm_client is None and settings.openai_api_key:
            llm_client = OpenAIChatClient(settings.openai_api_key, analysis_settings.llm_model, analysis_settings.llm_timeout_s)
        interpreter = LLMEvidenceInterpreter(llm_client) if llm_client is not None else NoInterpreter()
        return cls(evidence_source, interpreter, salary_provider or build_salary_provider(salary_settings),
                   max_evidence_per_topic=analysis_settings.max_evidence_per_topic)

    # ------------------------------------------------------------------ public
    def analyze(self, contract: ContractExtraction) -> ContractAnalysisResult:
        if not isinstance(contract, ContractExtraction):
            return self._invalid(contract, f"Expected a ContractExtraction, got {type(contract).__name__}.")
        try:
            result = self._analyze(contract)
        except Exception as exc:  # explicit failure instead of an empty "successful" analysis
            logger.warning("Contract analysis failed for document %s: %s", contract.document_id, type(exc).__name__)
            error = analysis_error(AnalysisErrorCode.ANALYSIS_FAILED, "analysis",
                                   "The contract analysis failed unexpectedly; no findings are reported.", exc)
            return self._empty(contract, AnalysisStatus.ANALYSIS_ERROR, [error], "The analysis could not be completed.")
        logger.info("Contract analysis for document %s: status=%s findings=%d", contract.document_id,
                    result.status.value, len(result.findings))
        return result

    def salary_benchmark(self, contract: ContractExtraction, context: SalaryQuery | None = None) -> SalaryBenchmark:
        """The configured salary benchmarking interface, used on its own (no regulatory analysis)."""
        if not isinstance(contract, ContractExtraction):
            raise TypeError(f"Expected a ContractExtraction, got {type(contract).__name__}.")
        return self._salary_benchmark(contract, [], context)

    # ------------------------------------------------------------------ workflow
    def _analyze(self, contract: ContractExtraction) -> ContractAnalysisResult:
        if contract.status is ResultStatus.ERROR:
            errors = [analysis_error(AnalysisErrorCode.EXTRACTION_NOT_USABLE, "input",
                                     "The contract extraction has no usable text, so it cannot be analysed."),
                      *contract.errors]
            return self._empty(contract, AnalysisStatus.INVALID_INPUT, errors,
                               "No analysis was performed because the contract text was not available.")

        fields = contract.fields()
        facts = {name: fact_from_field(name, field) for name, field in fields.items()}

        needed: dict[str, list[str]] = {}
        for name, field in fields.items():
            topic = self.topic_by_field.get(name)
            if topic is not None and field.status in (FieldStatus.FOUND, FieldStatus.AMBIGUOUS):
                needed.setdefault(topic.name, []).append(name)

        collector = RegulatoryEvidenceCollector(self.evidence_source, max_evidence_per_topic=self.max_evidence_per_topic)
        topic_evidence: dict[str, TopicEvidence] = {}
        for topic in self.topics:
            if topic.name in needed:
                topic_evidence[topic.name] = collector.collect(topic, needed[topic.name])

        interpretation_errors: list[ErrorInfo] = []
        findings = [
            self._finding(position, name, field, facts, topic_evidence, interpretation_errors)
            for position, (name, field) in enumerate(fields.items(), 1)
        ]

        warnings: list[str] = []
        salary = self._salary_benchmark(contract, warnings)
        errors = [*collector.errors, *interpretation_errors]
        status = self._overall_status(contract, findings, collector, interpretation_errors)
        if contract.status is ResultStatus.PARTIAL:
            warnings.append("The contract was only partly readable; fields on unread pages appear as not found.")
        if isinstance(self.interpreter, NoInterpreter) and topic_evidence:
            warnings.append("No automated interpretation is configured, so no field was labelled compliant or non-compliant.")
        warnings.append("English article text in the knowledge base is a machine translation; the Arabic text is the legal reference.")

        counts = status_counts(f.status for f in findings)
        return ContractAnalysisResult(
            status=status,
            document_id=contract.document_id,
            filename=contract.filename,
            document_status=contract.document_status,
            extraction_status=contract.status,
            overall_summary=self._summary(findings, counts, collector),
            status_counts=counts,
            findings=findings,
            regulatory_checks=[item.check for item in topic_evidence.values()],
            evidence=list(collector.registry.values()),
            interpreter=self.interpreter.name,
            salary_benchmark=salary,
            warnings=warnings,
            errors=errors,
        )

    def _finding(self, position: int, name: str, field: ExtractedField, facts: dict[str, DocumentFact],
                 topic_evidence: dict[str, TopicEvidence], interpretation_errors: list[ErrorInfo]) -> AnalysisFinding:
        fact = facts[name]
        topic = self.topic_by_field.get(name)
        base = dict(finding_id=f"F{position:02d}", field=name, topic=topic.name if topic else None, contract_fact=fact,
                    source_span=fact.sources[0] if fact.sources else None)

        if field.status is FieldStatus.NOT_FOUND:
            return self._make(base, FindingStatus.NOT_FOUND,
                              f"'{label(name)}' was not found in the extracted contract. This does not mean the contract "
                              "lacks it or that it is non-compliant; the clause may be worded in a way the extractor "
                              "does not recognise. No regulatory question was asked for it.")
        if field.status is FieldStatus.NOT_EXTRACTED:
            return self._make({**base, "topic": None}, FindingStatus.REQUIRES_REVIEW,
                              f"'{label(name)}' could not be extracted from the document.")
        if topic is None:
            if field.status is FieldStatus.AMBIGUOUS:
                return self._make(base, FindingStatus.REQUIRES_REVIEW, self._ambiguous_text(name, fact))
            return self._make(base, FindingStatus.NOT_APPLICABLE,
                              f"'{label(name)}' is recorded as a document fact. Sanad's MVP performs no regulatory "
                              "check for this field.")

        evidence = topic_evidence[topic.name]
        if evidence.check.status == "error":
            codes = ", ".join(sorted({e.code for e in evidence.check.errors}))
            return self._make(base, FindingStatus.ERROR,
                              f"The regulatory check for '{label(name)}' could not be performed because the existing "
                              f"RAG returned an error ({codes}). No compliance conclusion was made.")
        if field.status is FieldStatus.AMBIGUOUS:
            return self._make(base, FindingStatus.REQUIRES_REVIEW, self._ambiguous_text(name, fact),
                              regulatory_evidence=evidence.relevant)
        if not evidence.relevant:
            return self._make(base, FindingStatus.INSUFFICIENT_EVIDENCE,
                              f"The existing RAG did not return an article that addresses '{label(name)}' "
                              f"({len(evidence.check.retrieved_evidence_ids)} articles checked). "
                              "No compliance conclusion was made.")

        request = InterpretationRequest(
            field=name, topic=topic, fact=fact, evidence=evidence.relevant,
            related_facts=[facts[r] for r in topic.related_fields
                           if r != name and r in facts and facts[r].extraction_status is FieldStatus.FOUND],
        )
        try:
            interpretation = self.interpreter.interpret(request)
        except InterpretationError as exc:
            interpretation_errors.append(ErrorInfo(code=exc.code.value, stage="interpretation",
                                                   message=f"{exc.message} Field: {name}.", exception_type=exc.exception_type))
            interpretation = Interpretation(
                method=InterpretationMethod.LLM, assessment=FindingStatus.REQUIRES_REVIEW,
                explanation="Automated interpretation failed; the retrieved evidence must be reviewed manually.",
                cited_evidence_ids=[ref.evidence_id for ref in evidence.relevant], notes=[exc.message],
            )
        status = interpretation.assessment
        if status in (FindingStatus.COMPLIANT, FindingStatus.NON_COMPLIANT) and not interpretation.grounded:
            status = FindingStatus.REQUIRES_REVIEW
        if status not in (FindingStatus.COMPLIANT, FindingStatus.NON_COMPLIANT, FindingStatus.REQUIRES_REVIEW,
                          FindingStatus.INSUFFICIENT_EVIDENCE):
            status = FindingStatus.REQUIRES_REVIEW
        cited = [ref for ref in evidence.relevant if ref.evidence_id in interpretation.cited_evidence_ids]
        prefix = ("Agent interpretation (LLM, verified against the quoted evidence): "
                  if interpretation.grounded else "Agent note: ")
        return self._make(base, status, prefix + interpretation.explanation,
                          regulatory_evidence=cited or evidence.relevant, interpretation=interpretation,
                          notes=list(interpretation.notes))

    @staticmethod
    def _make(base: dict, status: FindingStatus, explanation: str, **extra) -> AnalysisFinding:
        return AnalysisFinding(**base, status=status, assessment=ASSESSMENT_LABELS[status], explanation=explanation, **extra)

    @staticmethod
    def _ambiguous_text(name: str, fact: DocumentFact) -> str:
        written = "; ".join(f"'{c.raw_value}'" for c in fact.candidates) or "an unreadable value"
        return (f"The contract states '{label(name)}' ambiguously ({written}). No compliance conclusion is drawn until "
                "the intended value is confirmed.")

    def _salary_benchmark(self, contract: ContractExtraction, warnings: list[str],
                          context: SalaryQuery | None = None) -> SalaryBenchmark:
        try:
            return self.salary_provider.benchmark(contract, context)
        except Exception as exc:
            warnings.append(f"Salary benchmarking failed ({type(exc).__name__}); no market data is reported.")
            return SalaryBenchmark(status="error", provider=getattr(self.salary_provider, "name", "unknown"),
                                   message="Salary benchmarking failed; no market data is reported.")

    @staticmethod
    def _overall_status(contract: ContractExtraction, findings: list[AnalysisFinding],
                        collector: RegulatoryEvidenceCollector, interpretation_errors: list[ErrorInfo]) -> AnalysisStatus:
        if collector.rag_unavailable is not None:
            return AnalysisStatus.RAG_ERROR
        if any(f.status is FindingStatus.ERROR for f in findings) or interpretation_errors or collector.errors:
            return AnalysisStatus.PARTIAL
        checked = [f for f in findings if f.topic is not None and f.contract_fact.extraction_status is FieldStatus.FOUND]
        if checked and all(f.status is FindingStatus.INSUFFICIENT_EVIDENCE for f in checked):
            return AnalysisStatus.INSUFFICIENT_EVIDENCE
        if contract.status is ResultStatus.PARTIAL:
            return AnalysisStatus.PARTIAL
        return AnalysisStatus.SUCCESS

    @staticmethod
    def _summary(findings: list[AnalysisFinding], counts: dict[str, int], collector: RegulatoryEvidenceCollector) -> str:
        parts = ", ".join(f"{count} {status.replace('_', ' ')}" for status, count in counts.items())
        text = f"{len(findings)} contract fields reviewed: {parts}."
        if collector.rag_unavailable is not None:
            text += " The existing regulatory RAG was unavailable, so no regulatory conclusion was made."
        else:
            text += (f" {collector.queries_sent} focused questions were sent to the existing Saudi Labor Law RAG, "
                     f"which returned {len(collector.registry)} distinct articles.")
        return text

    def _empty(self, contract, status: AnalysisStatus, errors: list[ErrorInfo], summary: str) -> ContractAnalysisResult:
        return ContractAnalysisResult(
            status=status, document_id=contract.document_id, filename=contract.filename,
            document_status=contract.document_status, extraction_status=contract.status, overall_summary=summary,
            interpreter=self.interpreter.name, errors=errors,
        )

    def _invalid(self, value, message: str) -> ContractAnalysisResult:
        if not all(hasattr(value, attribute) for attribute in ("document_id", "filename", "document_status", "status")):
            raise TypeError(message)
        error = analysis_error(AnalysisErrorCode.INVALID_INPUT, "input", message)
        return self._empty(value, AnalysisStatus.INVALID_INPUT, [error], "No analysis was performed: invalid input.")
