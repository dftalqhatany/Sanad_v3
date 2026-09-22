"""Sanad agents (Phase 4: Contract & CV Analysis Agent; Phase 5: Comparison & Recommendation Agent).

The contract agent obtains regulatory evidence only through sanad.rag.adapter.RegulatoryRAGAdapter.
Importing this package does not load the existing RAG, Qdrant, embeddings or OpenAI.
"""

from agents.analysis import AnalysisAgent, target_job_from_contract
from agents.comparison import ContractComparisonAgent
from agents.contract_analysis import ContractAnalysisAgent
from agents.cv_analysis import CvAnalysisAgent
from agents.interpretation import LLMEvidenceInterpreter, NoInterpreter, OpenAIChatClient
from agents.regulatory import CONTRACT_TOPICS, RegulatoryEvidenceCollector
from agents.salary import UnavailableSalaryBenchmarkProvider

__all__ = [
    "AnalysisAgent",
    "CONTRACT_TOPICS",
    "ContractComparisonAgent",
    "ContractAnalysisAgent",
    "CvAnalysisAgent",
    "LLMEvidenceInterpreter",
    "NoInterpreter",
    "OpenAIChatClient",
    "RegulatoryEvidenceCollector",
    "UnavailableSalaryBenchmarkProvider",
    "target_job_from_contract",
]
