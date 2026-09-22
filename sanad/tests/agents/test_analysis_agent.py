"""AnalysisAgent facade: contract and/or CV analysis in one structured bundle (no routing, no comparison)."""

from __future__ import annotations

import pytest

from agents import AnalysisAgent, ContractAnalysisAgent, target_job_from_contract
from agents.analysis import combined_status
from config import AnalysisSettings, SanadSettings
from models.analysis import AnalysisBundle, AnalysisStatus, TargetJob
from tests.fakes.regulatory_adapter import FakeRegulatoryAdapter

S = AnalysisStatus


def test_contract_and_cv_are_analysed_together(contract_en, cv_en, fake_rag):
    bundle = AnalysisAgent(ContractAnalysisAgent(fake_rag)).analyze(contract=contract_en, cv=cv_en)
    assert bundle.status is S.SUCCESS
    assert bundle.contract_analysis.document_id == contract_en.document_id
    assert bundle.cv_analysis.document_id == cv_en.document_id
    assert bundle.cv_analysis.job_compatibility is None  # the CV is not compared unless asked
    assert AnalysisBundle.model_validate_json(bundle.model_dump_json()).model_dump() == bundle.model_dump()


def test_cv_can_be_compared_with_the_contract_job_title_when_explicitly_requested(contract_en, cv_en, fake_rag):
    bundle = AnalysisAgent(ContractAnalysisAgent(fake_rag)).analyze(contract=contract_en, cv=cv_en,
                                                                   use_contract_job_title=True)
    job = bundle.cv_analysis.job_compatibility.target_job
    assert job.title == "Data Analyst" and job.source == "contract_job_title"
    assert job.source_span == contract_en.job_title.sources[0]
    assert bundle.cv_analysis.job_compatibility.requirements[0].status.value == "explicit_match"


def test_contract_job_title_is_not_used_when_not_found(make_contract, cv_en, fake_rag):
    _, contract = make_contract(["Probation Period: 90 days"])
    assert target_job_from_contract(contract) is None
    bundle = AnalysisAgent(ContractAnalysisAgent(fake_rag)).analyze(contract=contract, cv=cv_en,
                                                                   use_contract_job_title=True)
    assert bundle.cv_analysis.job_compatibility is None
    assert any("job title was not found" in w for w in bundle.cv_analysis.warnings)


def test_explicit_target_job_for_cv_only(cv_en):
    bundle = AnalysisAgent().analyze(cv=cv_en, target_job=TargetJob(title="Data Analyst", required_skills=["SQL"]))
    assert bundle.status is S.SUCCESS and bundle.contract_analysis is None
    assert bundle.cv_analysis.job_compatibility.overall == "required_requirements_explicitly_met"


def test_invalid_requests_raise(contract_en, cv_en, fake_rag):
    agent = AnalysisAgent(ContractAnalysisAgent(fake_rag))
    with pytest.raises(ValueError):
        agent.analyze()
    with pytest.raises(ValueError):
        agent.analyze(cv=cv_en, target_job=TargetJob(title="x"), use_contract_job_title=True)
    with pytest.raises(ValueError):
        agent.analyze(cv=cv_en, use_contract_job_title=True)
    with pytest.raises(ValueError):
        AnalysisAgent().analyze(contract=contract_en)


def test_bundle_status_does_not_hide_a_failed_part(contract_en, cv_en, knowledge_base):
    rag = FakeRegulatoryAdapter(knowledge_base, error_code="vector_db_unreachable")
    bundle = AnalysisAgent(ContractAnalysisAgent(rag)).analyze(contract=contract_en, cv=cv_en)
    assert bundle.contract_analysis.status is S.RAG_ERROR and bundle.cv_analysis.status is S.SUCCESS
    assert bundle.status is S.PARTIAL


@pytest.mark.parametrize("statuses, expected", [
    ([S.SUCCESS], S.SUCCESS),
    ([S.RAG_ERROR], S.RAG_ERROR),
    ([S.SUCCESS, S.SUCCESS], S.SUCCESS),
    ([S.SUCCESS, S.INSUFFICIENT_EVIDENCE], S.PARTIAL),
    ([S.RAG_ERROR, S.INVALID_INPUT], S.RAG_ERROR),
    ([S.INSUFFICIENT_EVIDENCE, S.ANALYSIS_ERROR], S.ANALYSIS_ERROR),
])
def test_combined_status(statuses, expected):
    assert combined_status(statuses) is expected


def test_from_settings_wires_the_contract_agent(fake_rag):
    agent = AnalysisAgent.from_settings(SanadSettings(openai_api_key=None), AnalysisSettings(), evidence_source=fake_rag)
    assert agent.contract_agent.evidence_source is fake_rag and agent.cv_agent is not None
