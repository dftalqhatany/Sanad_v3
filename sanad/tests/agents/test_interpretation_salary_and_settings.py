"""Evidence verification of LLM output, the salary benchmark interface, and agent construction from settings."""

from __future__ import annotations

import sys

import pytest

from agents import ContractAnalysisAgent, LLMEvidenceInterpreter, NoInterpreter, OpenAIChatClient
from agents.interpretation import (
    InterpretationRequest,
    LLMInterpretationOutput,
    _strip_code_fence,
    verify_output,
)
from agents.regulatory import CONTRACT_TOPICS, RegulatoryEvidenceCollector
from agents.salary import UnavailableSalaryBenchmarkProvider
from agents.shared import fact_from_field
from config import AnalysisSettings, SanadSettings
from models.analysis import FindingStatus, SalaryBenchmark, SalaryObservation, SalarySource
from models.extraction import FieldStatus
from tests.fakes.llm import FakeLLMClient
from tests.fakes.regulatory_adapter import KB_PROBATION_ART_53, KB_WEEKLY_REST_ART_104

PROBATION = next(t for t in CONTRACT_TOPICS if t.name == "probation")
QUOTE = "وجب النص على ذلك صراحة في عقد العمل"


@pytest.fixture
def probation_request(contract_en, fake_rag):
    collected = RegulatoryEvidenceCollector(fake_rag).collect(PROBATION, ["probation_period"])
    return InterpretationRequest(field="probation_period", topic=PROBATION,
                                 fact=fact_from_field("probation_period", contract_en.probation_period),
                                 evidence=collected.relevant)


def output(**overrides):
    data = {"assessment": "compliant", "explanation": "Supported by the quoted text.",
            "cited_evidence_ids": [f"kb-{KB_PROBATION_ART_53}"],
            "evidence_quotes": [{"evidence_id": f"kb-{KB_PROBATION_ART_53}", "quote": QUOTE}],
            "contract_quote": "ninety (90) days"}
    return LLMInterpretationOutput.model_validate({**data, **overrides})


def test_verified_quote_tolerates_tatweel_diacritics_and_spacing(probation_request):
    spaced = "وجـب   النَّص على ذلك\nصراحةً في عقد العمل"
    result = verify_output(output(evidence_quotes=[{"evidence_id": f"kb-{KB_PROBATION_ART_53}", "quote": spaced}]),
                           probation_request, model="m")
    assert result.assessment is FindingStatus.COMPLIANT and result.grounded
    assert result.evidence_quotes[0].verified and result.evidence_quotes[0].language == "ar"


def test_too_short_quotes_do_not_count(probation_request):
    result = verify_output(output(evidence_quotes=[{"evidence_id": f"kb-{KB_PROBATION_ART_53}", "quote": "العمل"}]),
                           probation_request, model="m")
    assert result.assessment is FindingStatus.REQUIRES_REVIEW and not result.evidence_quotes[0].verified


def test_quote_must_come_from_a_cited_article(probation_request):
    result = verify_output(output(cited_evidence_ids=[]), probation_request, model="m")
    assert result.assessment is FindingStatus.REQUIRES_REVIEW and not result.grounded


def test_article_number_in_evidence_is_allowed_in_explanation(probation_request):
    result = verify_output(output(explanation="المادة (53) تحدد مدة التجربة."), probation_request, model="m")
    assert result.grounded
    result = verify_output(output(explanation="المادة ٨٠ تنطبق أيضا."), probation_request, model="m")
    assert not result.grounded and any("80" in note for note in result.notes)


def test_review_answers_drop_unknown_citations(probation_request):
    result = verify_output(output(assessment="requires_review", cited_evidence_ids=["kb-7", f"kb-{KB_PROBATION_ART_53}"]),
                           probation_request, model="m")
    assert result.cited_evidence_ids == [f"kb-{KB_PROBATION_ART_53}"]
    assert any("kb-7" in note for note in result.notes)


def test_json_in_code_fence_is_accepted():
    assert _strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_no_interpreter_cites_every_relevant_article_and_concludes_nothing(probation_request):
    result = NoInterpreter().interpret(probation_request)
    assert result.assessment is FindingStatus.REQUIRES_REVIEW
    assert result.cited_evidence_ids == [f"kb-{KB_PROBATION_ART_53}"] and not result.grounded


def test_keyword_relevance_requires_a_word_start_match(knowledge_base):
    weekly_rest = next(t for t in CONTRACT_TOPICS if t.name == "weekly_rest")
    probation_article = knowledge_base[KB_PROBATION_ART_53 - 1]["arabic_content"]
    assert "صراحة" in probation_article  # contains the letters of "راحة" inside another word
    assert not weekly_rest.mentioned_in(probation_article)
    assert weekly_rest.mentioned_in(knowledge_base[KB_WEEKLY_REST_ART_104 - 1]["arabic_content"])
    assert PROBATION.mentioned_in(probation_article)  # "للتجربة"


# --------------------------------------------------------------------------- salary benchmark interface
def test_salary_benchmark_is_not_configured_and_reports_no_market_data(contract_en, fake_rag):
    benchmark = ContractAnalysisAgent(fake_rag).analyze(contract_en).salary_benchmark
    assert benchmark.status == "not_configured"
    assert benchmark.market_min is None and benchmark.market_max is None and benchmark.sources == []
    assert benchmark.contract_salary.amount == 12000.0 and benchmark.job_title == "Data Analyst"


def test_salary_benchmark_copies_only_found_values(make_contract):
    _, contract = make_contract(["Basic Salary: 10,000 SAR", "The basic salary is 12,000 SAR per month."])
    assert contract.salary.status is FieldStatus.AMBIGUOUS
    benchmark = UnavailableSalaryBenchmarkProvider().benchmark(contract)
    assert benchmark.contract_salary is None and benchmark.job_title is None


def test_market_salary_data_cannot_exist_without_sources():
    observation = SalaryObservation(source_name="Survey", url="https://example.org/s", tier="market",
                                    retrieved_at="2026-09-18", quote="8,000 to 15,000 SAR per month", minimum=8000,
                                    maximum=15000, currency="SAR", period="monthly", monthly_min=8000,
                                    monthly_max=15000, basis="base_salary")
    source = SalarySource(name="Survey", url="https://example.org/s", source_type="market_survey", tier="market")
    with pytest.raises(ValueError, match="cite at least one source"):
        SalaryBenchmark(status="success", provider="x", message="m", market_min=8000, market_max=15000)
    with pytest.raises(ValueError, match="sourced market range"):
        SalaryBenchmark(status="success", provider="x", message="m")
    with pytest.raises(ValueError, match="usable salary observations"):
        SalaryBenchmark(status="success", provider="x", message="m", market_min=8000, market_max=15000,
                        period="monthly", basis="base_salary", sources=[source])
    with pytest.raises(ValueError, match="stay within the observed figures"):
        SalaryBenchmark(status="success", provider="x", message="m", market_min=3000, market_max=15000,
                        period="monthly", basis="base_salary", sources=[source], observations=[observation])
    ok = SalaryBenchmark(status="success", provider="x", message="m", market_min=8000, market_max=15000,
                         period="monthly", basis="base_salary", data_quality="market", sources=[source],
                         observations=[observation])
    assert ok.sources and ok.observations


def test_failing_salary_provider_is_explicit(contract_en, fake_rag):
    class Broken:
        name = "broken"

        def benchmark(self, contract, context=None):
            raise ConnectionError("offline")

    result = ContractAnalysisAgent(fake_rag, salary_provider=Broken()).analyze(contract_en)
    assert result.salary_benchmark.status == "error" and result.salary_benchmark.market_min is None
    assert any("Salary benchmarking failed" in w for w in result.warnings)


# --------------------------------------------------------------------------- settings / construction
def test_from_settings_without_api_key_uses_no_llm(fake_rag):
    agent = ContractAnalysisAgent.from_settings(SanadSettings(openai_api_key=None), AnalysisSettings(),
                                                evidence_source=fake_rag)
    assert isinstance(agent.interpreter, NoInterpreter) and agent.evidence_source is fake_rag


def test_from_settings_with_api_key_builds_openai_client_without_calling_it(fake_rag, monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)  # any real OpenAI use would fail loudly
    agent = ContractAnalysisAgent.from_settings(SanadSettings(openai_api_key="sk-test-hidden"),
                                                AnalysisSettings(llm_model="gpt-4o-mini"), evidence_source=fake_rag)
    assert isinstance(agent.interpreter, LLMEvidenceInterpreter)
    assert agent.interpreter.name == "llm:gpt-4o-mini"
    assert "sk-test-hidden" not in repr(agent.interpreter.client)


def test_from_settings_uses_the_existing_rag_adapter_by_default():
    from rag.adapter import RegulatoryRAGAdapter

    agent = ContractAnalysisAgent.from_settings(SanadSettings(openai_api_key=None), AnalysisSettings())
    assert isinstance(agent.evidence_source, RegulatoryRAGAdapter)
    assert not agent.evidence_source.is_backend_loaded  # nothing is loaded until the first question


def test_injected_llm_client_is_used(fake_rag):
    llm = FakeLLMClient()
    agent = ContractAnalysisAgent.from_settings(SanadSettings(openai_api_key=None), AnalysisSettings(),
                                                evidence_source=fake_rag, llm_client=llm)
    assert agent.interpreter.client is llm


def test_openai_client_requires_a_key():
    with pytest.raises(ValueError):
        OpenAIChatClient("")


def test_analysis_settings_from_env():
    settings = AnalysisSettings.from_env({"SANAD_ANALYSIS_LLM_MODEL": "gpt-x", "SANAD_ANALYSIS_LLM_TIMEOUT_S": "12"})
    assert settings.llm_model == "gpt-x" and settings.llm_timeout_s == 12.0
    assert AnalysisSettings.from_env({}).llm_model == "gpt-4o-mini"
    with pytest.raises(ValueError):
        AnalysisSettings(llm_timeout_s=0)
