from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi", reason="the Phase 7 API needs fastapi (see sanad/requirements.txt)")
from fastapi.testclient import TestClient  # noqa: E402

from sanad.agents import AnalysisAgent, ContractAnalysisAgent, ContractComparisonAgent  # noqa: E402
from sanad.api import create_app  # noqa: E402
from sanad.models.analysis import AnalysisStatus  # noqa: E402
from sanad.models.orchestration import OrchestratorResult, Route, RoutingDecision  # noqa: E402
from sanad.orchestrator import SanadOrchestrator  # noqa: E402
from tests.fakes.regulatory_adapter import FakeRegulatoryAdapter  # noqa: E402
from tests.fixtures.documents.builders import docx_from_paragraphs  # noqa: E402
from tests.fixtures.documents.fixtures import parsed, processor, sample_bytes, sample_paths  # noqa: E402,F401

CONTRACT_LINES = ["Employment Contract", "Job Title: Data Analyst", "Contract Type: Fixed-term",
                  "Basic Salary: 12,000 SAR per month", "Housing Allowance: 3,000 SAR per month",
                  "Working Hours: 8 hours per day, 40 hours per week", "Working Days: Sunday to Thursday",
                  "Annual Leave: 30 days", "Probation Period: 90 days", "Notice Period: 60 days",
                  "Work Location: Riyadh"]
DOCX_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def contract_bytes(*replacements: tuple[str, str]) -> bytes:
    """The sample contract, with literal substitutions such as ("12,000", "9,000")."""
    lines = CONTRACT_LINES
    for old, new in replacements:
        lines = [line.replace(old, new) for line in lines]
    return docx_from_paragraphs(lines)


def docx_upload(name: str, data: bytes | None = None, role: str | None = None):
    return ("files", (name, data if data is not None else contract_bytes(), DOCX_TYPE)), role


class StubOrchestrator:
    """Records what the API passes and returns a canned result; the API may use nothing else."""

    def __init__(self, status: AnalysisStatus = AnalysisStatus.SUCCESS, raises: BaseException | None = None) -> None:
        self.calls: list = []
        self.status = status
        self.raises = raises
        self.comparison_agent = SimpleNamespace(max_contracts=5)

    def handle(self, request):
        self.calls.append(request)
        if self.raises is not None:
            raise self.raises
        errors = ([{"code": "routed_agent_failed", "stage": "x", "message": "scripted failure"}]
                  if self.status in (AnalysisStatus.INVALID_INPUT, AnalysisStatus.RAG_ERROR,
                                     AnalysisStatus.ANALYSIS_ERROR) else [])
        return OrchestratorResult(
            status=self.status, summary="scripted result", errors=errors,
            routing=RoutingDecision(route=Route.CONTRACT_ANALYSIS, agent="AnalysisAgent", rule="stub",
                                    reason="stub", contract_count=len(request.documents)),
        )


@pytest.fixture
def real_orchestrator(knowledge_base) -> SanadOrchestrator:
    rag = FakeRegulatoryAdapter(knowledge_base)
    analysis = AnalysisAgent(ContractAnalysisAgent(rag))
    orchestrator = SanadOrchestrator(analysis, ContractComparisonAgent(analysis), rag)
    orchestrator.rag = rag
    return orchestrator


@pytest.fixture
def client(real_orchestrator) -> TestClient:
    return TestClient(create_app(real_orchestrator))


@pytest.fixture
def stub() -> StubOrchestrator:
    return StubOrchestrator()


@pytest.fixture
def stub_client(stub) -> TestClient:
    return TestClient(create_app(stub))
