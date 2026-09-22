from __future__ import annotations

import pytest

from agents import AnalysisAgent, ContractAnalysisAgent, ContractComparisonAgent
from models.orchestration import DocumentRole, UploadedDocument
from orchestrator import SanadOrchestrator
from tests.fakes.regulatory_adapter import FakeRegulatoryAdapter
from tests.fixtures.documents.builders import docx_from_paragraphs
from tests.fixtures.documents.fixtures import parsed, processor, sample_bytes, sample_paths  # noqa: F401

CONTRACT_LINES = ["Employment Contract", "Job Title: Data Analyst", "Contract Type: Fixed-term",
                  "Basic Salary: 12,000 SAR per month", "Housing Allowance: 3,000 SAR per month",
                  "Working Hours: 8 hours per day, 40 hours per week", "Working Days: Sunday to Thursday",
                  "Annual Leave: 30 days", "Probation Period: 90 days", "Notice Period: 60 days",
                  "Work Location: Riyadh"]


class SpyAnalysisAgent(AnalysisAgent):
    def __init__(self, contract_agent):
        super().__init__(contract_agent)
        self.calls: list[dict] = []
        self.salary_calls: list = []

    def analyze(self, **kwargs):
        self.calls.append(kwargs)
        return super().analyze(**kwargs)

    def benchmark_salary(self, contract, context=None):
        self.salary_calls.append({"contract": contract, "context": context})
        return super().benchmark_salary(contract, context)


class SpyComparisonAgent(ContractComparisonAgent):
    def __init__(self, analysis_agent, **kwargs):
        super().__init__(analysis_agent, **kwargs)
        self.calls: list[dict] = []

    def compare(self, contracts, cv=None, **kwargs):
        self.calls.append({"contracts": list(contracts), "cv": cv, **kwargs})
        return super().compare(contracts, cv, **kwargs)


@pytest.fixture
def rag(knowledge_base) -> FakeRegulatoryAdapter:
    return FakeRegulatoryAdapter(knowledge_base)


@pytest.fixture
def analysis(rag) -> SpyAnalysisAgent:
    return SpyAnalysisAgent(ContractAnalysisAgent(rag))


@pytest.fixture
def comparison(analysis) -> SpyComparisonAgent:
    return SpyComparisonAgent(analysis)


@pytest.fixture
def orchestrator(analysis, comparison, rag) -> SanadOrchestrator:
    return SanadOrchestrator(analysis, comparison, rag)


@pytest.fixture
def upload(sample_bytes):
    """upload('contract', lines=..., role=...) -> UploadedDocument built from synthetic bytes."""
    def make(name: str, *, lines: list[str] | None = None, role: DocumentRole = DocumentRole.AUTO,
             filename: str | None = None, label: str | None = None) -> UploadedDocument:
        if lines is not None:
            data, default_name = docx_from_paragraphs(lines), f"{name}.docx"
        else:
            data, default_name = sample_bytes[name], name
        return UploadedDocument(filename=filename or default_name, content=data, role=role, label=label)
    return make
