"""Contract Analysis Agent: facts -> focused questions -> existing RAG adapter -> evidence -> interpretation."""

from __future__ import annotations

import json
import logging

import pytest

from agents import ContractAnalysisAgent, LLMEvidenceInterpreter, NoInterpreter
from agents.regulatory import CONTRACT_TOPICS
from extraction import extract_contract
from models.analysis import AnalysisFinding, AnalysisStatus, ContractAnalysisResult, FindingStatus
from models.common import ResultStatus
from models.extraction import FieldStatus
from tests.fakes.llm import FakeLLMClient
from tests.fakes.regulatory_adapter import (
    KB_ARTICLE_1,
    KB_ARTICLE_3,
    KB_PROBATION_ART_53,
    NOISE,
    FakeRegulatoryAdapter,
)
from tests.fixtures.documents.builders import pdf_with_text_pages_and_scanned_page

PROBATION_QUOTE_AR = "على ألا يزيد مجموع المدة في جميع الأحوال على مائة وثمانين يوماً"
COMPLIANCE_LABELS = {FindingStatus.COMPLIANT, FindingStatus.NON_COMPLIANT}


def grounded_probation_verdict(assessment: str, contract_quote: str):
    def respond(payload):
        if payload["contract_fact"]["field"] != "probation_period":
            return {"assessment": "requires_review", "explanation": "Depends on conditions not given.",
                    "cited_evidence_ids": [e["evidence_id"] for e in payload["evidence"]]}
        return {
            "assessment": assessment,
            "explanation": "Article 53 limits the total probation period to 180 days.",
            "cited_evidence_ids": [f"kb-{KB_PROBATION_ART_53}"],
            "evidence_quotes": [{"evidence_id": f"kb-{KB_PROBATION_ART_53}", "quote": PROBATION_QUOTE_AR}],
            "contract_quote": contract_quote,
        }
    return respond


# --------------------------------------------------------------------------- workflow and statuses
def test_full_contract_uses_the_adapter_and_never_labels_compliance_without_an_interpreter(contract_en, fake_rag):
    result = ContractAnalysisAgent(fake_rag).analyze(contract_en)

    assert result.status is AnalysisStatus.SUCCESS
    assert result.interpreter == "none"
    assert len(result.findings) == len(contract_en.fields())
    assert not {f.status for f in result.findings} & COMPLIANCE_LABELS
    assert result.finding("probation_period").status is FindingStatus.REQUIRES_REVIEW
    assert result.finding("employee_name").status is FindingStatus.NOT_APPLICABLE
    assert any("No automated interpretation" in w for w in result.warnings)
    assert result.status_counts == {s: n for s, n in sorted(
        {f.status.value: sum(1 for g in result.findings if g.status is f.status) for f in result.findings}.items())}
    assert result.disclaimer and "not legal advice" in result.disclaimer
    assert ContractAnalysisResult.model_validate_json(result.model_dump_json()).model_dump() == result.model_dump()


def test_questions_are_only_asked_for_topics_whose_fields_are_present(make_contract, fake_rag):
    _, contract = make_contract(["Employee Name: Sample Person", "Probation Period: 90 days", "Annual Leave: 21 days"])
    result = ContractAnalysisAgent(fake_rag).analyze(contract)

    assert fake_rag.topics_queried() == {"probation", "annual_leave"}
    probation = next(t for t in CONTRACT_TOPICS if t.name == "probation")
    assert set(probation.questions) <= set(fake_rag.questions)
    assert [c.topic for c in result.regulatory_checks] == ["probation", "annual_leave"]
    assert result.regulatory_checks[0].triggered_by_fields == ["probation_period"]


def test_not_found_field_is_not_found_never_non_compliant_and_never_queried(make_contract, fake_rag):
    llm = FakeLLMClient(grounded_probation_verdict("non_compliant", "Annual Leave: 21 days"))
    _, contract = make_contract(["Annual Leave: 21 days"])
    result = ContractAnalysisAgent(fake_rag, LLMEvidenceInterpreter(llm)).analyze(contract)

    finding = result.finding("probation_period")
    assert contract.probation_period.status is FieldStatus.NOT_FOUND
    assert finding.status is FindingStatus.NOT_FOUND
    assert finding.regulatory_evidence == [] and finding.interpretation is None
    assert "not found in the extracted contract" in finding.explanation
    assert "does not mean" in finding.explanation and "non-compliant" in finding.explanation
    assert "probation" not in fake_rag.topics_queried()
    assert "probation_period" not in llm.fields_called()
    assert all(f.status is FindingStatus.NOT_FOUND for f in result.findings
               if f.contract_fact.extraction_status is FieldStatus.NOT_FOUND)


def test_no_relevant_article_is_insufficient_evidence_not_non_compliant(make_contract, knowledge_base):
    rag = FakeRegulatoryAdapter(knowledge_base, hits_by_topic={}, default_hits=list(NOISE))
    llm = FakeLLMClient({"assessment": "non_compliant", "explanation": "x"})
    _, contract = make_contract(["Probation Period: 200 days"])
    result = ContractAnalysisAgent(rag, LLMEvidenceInterpreter(llm)).analyze(contract)

    finding = result.finding("probation_period")
    assert finding.status is FindingStatus.INSUFFICIENT_EVIDENCE
    assert finding.regulatory_evidence == [] and finding.interpretation is None
    assert llm.calls == []  # the LLM is only used after relevant evidence was retrieved
    assert result.status is AnalysisStatus.INSUFFICIENT_EVIDENCE
    check = result.regulatory_checks[0]
    assert check.status == "insufficient_evidence"
    assert check.retrieved_evidence_ids == [f"kb-{KB_ARTICLE_1}", f"kb-{KB_ARTICLE_3}"]
    assert check.relevant_evidence_ids == []
    assert not any(f.status is FindingStatus.NON_COMPLIANT for f in result.findings)


def test_zero_score_evidence_is_never_treated_as_relevant(make_contract, knowledge_base):
    rag = FakeRegulatoryAdapter(knowledge_base, hits_by_topic={"probation": [(KB_PROBATION_ART_53, 0.0)]})
    _, contract = make_contract(["Probation Period: 90 days"])
    result = ContractAnalysisAgent(rag).analyze(contract)

    assert result.finding("probation_period").status is FindingStatus.INSUFFICIENT_EVIDENCE
    assert result.status is AnalysisStatus.INSUFFICIENT_EVIDENCE


def test_ambiguous_value_requires_review_with_evidence_and_is_not_sent_to_the_llm(make_contract, fake_rag):
    llm = FakeLLMClient(grounded_probation_verdict("compliant", "90 or 180 days"))
    _, contract = make_contract(["Probation Period: 90 or 180 days", "Annual Leave: 30 days"])
    result = ContractAnalysisAgent(fake_rag, LLMEvidenceInterpreter(llm)).analyze(contract)

    finding = result.finding("probation_period")
    assert contract.probation_period.status is FieldStatus.AMBIGUOUS
    assert finding.status is FindingStatus.REQUIRES_REVIEW
    assert "ambiguous" in finding.explanation and "90 or 180 days" in finding.explanation
    assert [e.article_number for e in finding.regulatory_evidence] == [53]
    assert finding.interpretation is None
    assert llm.fields_called() == ["annual_leave"]


def test_rag_unavailable_is_rag_error_and_no_regulatory_conclusion(contract_en, knowledge_base):
    rag = FakeRegulatoryAdapter(knowledge_base, error_code="vector_db_unreachable")
    llm = FakeLLMClient(grounded_probation_verdict("compliant", "ninety (90) days"))
    result = ContractAnalysisAgent(rag, LLMEvidenceInterpreter(llm)).analyze(contract_en)

    assert result.status is AnalysisStatus.RAG_ERROR
    assert [e.code for e in result.errors] == ["vector_db_unreachable"]
    assert len(rag.calls) == 1  # stops instead of hammering an unavailable RAG
    assert llm.calls == []
    assert result.evidence == []
    assert all(c.status == "error" for c in result.regulatory_checks)
    regulatory = [f for f in result.findings if f.topic is not None and f.status is not FindingStatus.NOT_FOUND]
    assert regulatory and all(f.status is FindingStatus.ERROR for f in regulatory)
    assert not {f.status for f in result.findings} & COMPLIANCE_LABELS
    assert "unavailable" in result.overall_summary


def test_adapter_raising_is_reported_as_rag_error(make_contract, knowledge_base):
    rag = FakeRegulatoryAdapter(knowledge_base, raise_exc=RuntimeError("boom with secret context"))
    _, contract = make_contract(["Probation Period: 90 days"])
    result = ContractAnalysisAgent(rag).analyze(contract)

    assert result.status is AnalysisStatus.RAG_ERROR
    assert result.errors[0].code == "rag_unavailable" and result.errors[0].exception_type == "RuntimeError"
    assert "secret context" not in result.model_dump_json()


def test_initialization_failure_is_rag_error(make_contract, knowledge_base):
    rag = FakeRegulatoryAdapter(knowledge_base, error_code="rag_initialization_failed", error_stage="initialization")
    _, contract = make_contract(["Annual Leave: 21 days"])
    assert ContractAnalysisAgent(rag).analyze(contract).status is AnalysisStatus.RAG_ERROR


def test_one_failed_topic_is_partial_and_other_topics_still_analysed(make_contract, knowledge_base):
    rag = FakeRegulatoryAdapter(knowledge_base, error_code="retrieval_failed", error_topics=("probation",))
    _, contract = make_contract(["Probation Period: 90 days", "Annual Leave: 21 days"])
    result = ContractAnalysisAgent(rag).analyze(contract)

    assert result.status is AnalysisStatus.PARTIAL
    assert result.finding("probation_period").status is FindingStatus.ERROR
    assert result.finding("annual_leave").status is FindingStatus.REQUIRES_REVIEW
    assert {e.code for e in result.errors} == {"retrieval_failed"}


def test_partially_readable_contract_is_partial(processor, fake_rag):
    document = processor.parse_bytes(pdf_with_text_pages_and_scanned_page(), "mixed.pdf")
    contract = extract_contract(document)
    assert contract.status is ResultStatus.PARTIAL
    result = ContractAnalysisAgent(fake_rag).analyze(contract)
    assert result.status is AnalysisStatus.PARTIAL
    assert any("partly readable" in w for w in result.warnings)


# --------------------------------------------------------------------------- invalid input / unexpected failure
def test_wrong_extraction_type_is_invalid_input(cv_en, fake_rag):
    result = ContractAnalysisAgent(fake_rag).analyze(cv_en)
    assert result.status is AnalysisStatus.INVALID_INPUT
    assert result.errors[0].code == "invalid_input" and result.findings == []
    assert fake_rag.calls == []


def test_non_extraction_object_raises_type_error(fake_rag):
    with pytest.raises(TypeError):
        ContractAnalysisAgent(fake_rag).analyze({"salary": 5000})


def test_unreadable_contract_is_invalid_input(parsed, fake_rag):
    contract = extract_contract(parsed["sample_empty.pdf"])
    assert contract.status is ResultStatus.ERROR
    result = ContractAnalysisAgent(fake_rag).analyze(contract)
    assert result.status is AnalysisStatus.INVALID_INPUT
    assert result.errors[0].code == "extraction_not_usable"
    assert fake_rag.calls == []


def test_unexpected_failure_is_analysis_error_not_an_empty_success(make_contract, fake_rag):
    class BrokenInterpreter:
        name = "broken"

        def interpret(self, request):
            raise RuntimeError("bug")

    _, contract = make_contract(["Probation Period: 90 days"])
    result = ContractAnalysisAgent(fake_rag, BrokenInterpreter()).analyze(contract)
    assert result.status is AnalysisStatus.ANALYSIS_ERROR
    assert result.findings == [] and result.errors[0].code == "analysis_failed"


# --------------------------------------------------------------------------- traceability
def test_regulatory_evidence_is_fully_traceable_to_the_knowledge_base(contract_en, fake_rag, knowledge_base):
    result = ContractAnalysisAgent(fake_rag).analyze(contract_en)
    evidence = result.finding("probation_period").regulatory_evidence
    assert [e.evidence_id for e in evidence] == [f"kb-{KB_PROBATION_ART_53}"]

    ref = evidence[0]
    article = knowledge_base[KB_PROBATION_ART_53 - 1]
    assert ref.article_number == article["article_number"] == 53
    assert ref.article_name == article["arabic_name"]
    assert ref.part == article["part_title_ar"] and ref.chapter == article["chapter_title_ar"]
    assert ref.arabic_text == article["arabic_content"] and ref.english_text == article["english_content"]
    assert ref.evidence.raw_metadata == article
    assert "نظام العمل" in ref.citation and article["arabic_name"] in ref.citation
    assert ref.source and ref.rank == 1 and ref.score == pytest.approx(0.83)
    assert {r.topic for r in ref.retrievals} == {"probation"}
    assert {r.query for r in ref.retrievals} == set(next(t for t in CONTRACT_TOPICS if t.name == "probation").questions)

    dumped = json.loads(result.model_dump_json())
    item = next(e for e in dumped["evidence"] if e["evidence_id"] == ref.evidence_id)
    for key in ("source", "article_number", "article_name", "part", "chapter", "arabic_text", "english_text",
                "citation", "rank", "score"):
        assert item[key] not in (None, ""), key
    assert {e["evidence_id"] for e in dumped["evidence"]} >= {
        e.evidence_id for f in result.findings for e in f.regulatory_evidence}


def test_contract_facts_keep_provenance(parsed, fake_rag):
    document = parsed["sample_contract_text.pdf"]
    contract = extract_contract(document)
    result = ContractAnalysisAgent(fake_rag).analyze(contract)

    found = [f for f in result.findings if f.contract_fact.extraction_status is FieldStatus.FOUND]
    assert found
    for finding in found:
        fact = finding.contract_fact
        field = getattr(contract, finding.field)
        assert fact.field == finding.field
        assert fact.raw_value == field.raw_value and fact.source_text == field.source_text
        assert fact.source_text in document.full_text
        assert fact.page_number is not None and fact.page_number == field.page_number
        assert fact.section_id == field.sources[0].section_id or fact.table_id == field.sources[0].table_id
        assert finding.source_span == field.sources[0]


def test_document_fact_evidence_and_interpretation_are_kept_apart(contract_en, fake_rag):
    finding = ContractAnalysisAgent(fake_rag).analyze(contract_en).finding("annual_leave")
    assert finding.contract_fact.raw_value == "thirty (30) days"
    assert finding.regulatory_evidence[0].article_number == 109
    assert finding.interpretation.method.value == "none" and finding.interpretation.grounded is False
    assert finding.explanation.startswith("Agent note: ")


# --------------------------------------------------------------------------- LLM interpretation (mocked)
def test_grounded_llm_compliant_verdict_is_never_promoted_to_a_field_level_status(contract_en, fake_rag):
    """Stage 4 hardening: even a fully grounded, quote-verified LLM 'compliant' reading is demoted to
    REQUIRES_REVIEW at the field level. `interpretation.assessment`/`grounded` still record what the
    interpreter proposed and verified, for a human reviewer; only the deterministic clause-level
    ClauseLegalFinding may ever set status to COMPLIANT/NON_COMPLIANT (see agents/legal_rules.py)."""
    llm = FakeLLMClient(grounded_probation_verdict("compliant", "ninety (90) days"))
    result = ContractAnalysisAgent(fake_rag, LLMEvidenceInterpreter(llm)).analyze(contract_en)

    finding = result.finding("probation_period")
    assert finding.status is FindingStatus.REQUIRES_REVIEW
    assert finding.interpretation.assessment is FindingStatus.COMPLIANT  # the interpreter's own proposal, kept
    assert finding.interpretation.grounded and finding.interpretation.model == "fake-llm"
    assert finding.interpretation.evidence_quotes[0].verified and finding.interpretation.evidence_quotes[0].language == "ar"
    assert [e.article_number for e in finding.regulatory_evidence] == [53]
    assert finding.explanation.startswith("Agent interpretation (LLM")
    assert any("does not accept a compliance verdict at the field level" in note for note in finding.notes)
    assert result.interpreter == "llm:fake-llm"
    assert not {f.status for f in result.findings} & COMPLIANCE_LABELS


def test_grounded_llm_non_compliant_verdict_is_never_promoted_to_a_field_level_status(make_contract, fake_rag):
    """Same hardening as above, for a 'non_compliant' proposal."""
    llm = FakeLLMClient(grounded_probation_verdict("non_compliant", "200 days"))
    _, contract = make_contract(["Probation Period: 200 days"])
    result = ContractAnalysisAgent(fake_rag, LLMEvidenceInterpreter(llm)).analyze(contract)

    finding = result.finding("probation_period")
    assert finding.status is FindingStatus.REQUIRES_REVIEW
    assert finding.interpretation.assessment is FindingStatus.NON_COMPLIANT  # the interpreter's own proposal, kept
    assert finding.assessment.startswith("Requires human review")
    assert any("does not accept a compliance verdict at the field level" in note for note in finding.notes)
    assert result.status is AnalysisStatus.SUCCESS
    assert not {f.status for f in result.findings} & COMPLIANCE_LABELS


@pytest.mark.parametrize("change, reason", [
    ({"cited_evidence_ids": ["kb-999"], "evidence_quotes": [{"evidence_id": "kb-999", "quote": PROBATION_QUOTE_AR}]},
     "not supplied"),
    ({"evidence_quotes": [{"evidence_id": f"kb-{KB_PROBATION_ART_53}", "quote": "لا يجوز أن تزيد فترة التجربة على تسعين يوما"}]},
     "no exact quote"),
    ({"evidence_quotes": [{"evidence_id": f"kb-{KB_PROBATION_ART_53}",
                           "quote": "with the total duration in any event not exceeding one hundred and eighty days"}]},
     "no exact quote"),
    ({"contract_quote": "probation period of sixty (60) days"}, "contract quote"),
    ({"contract_quote": None}, "contract quote"),
    ({"explanation": "Article 80 of the Labor Law also applies."}, "article numbers not in the retrieved evidence"),
    ({"evidence_quotes": []}, "no exact quote"),
])
def test_ungrounded_llm_compliance_verdicts_are_downgraded(contract_en, fake_rag, change, reason):
    base = grounded_probation_verdict("non_compliant", "ninety (90) days")

    def respond(payload):
        response = base(payload)
        return {**response, **change} if payload["contract_fact"]["field"] == "probation_period" else response

    result = ContractAnalysisAgent(fake_rag, LLMEvidenceInterpreter(FakeLLMClient(respond))).analyze(contract_en)
    finding = result.finding("probation_period")
    assert finding.status is FindingStatus.REQUIRES_REVIEW
    assert finding.interpretation.grounded is False
    assert any(reason in note for note in finding.notes), finding.notes
    assert all(e.evidence_id.startswith("kb-") and e.evidence_id != "kb-999" for e in finding.regulatory_evidence)


@pytest.mark.parametrize("client, code", [
    (FakeLLMClient(error=TimeoutError("sk-test-secret timed out")), "llm_request_failed"),
    (FakeLLMClient("this is not json"), "llm_output_invalid"),
    (FakeLLMClient({"assessment": "maybe", "explanation": "x"}), "llm_output_invalid"),
])
def test_llm_failures_are_explicit_and_findings_fall_back_to_review(make_contract, fake_rag, client, code):
    _, contract = make_contract(["Probation Period: 90 days"])
    result = ContractAnalysisAgent(fake_rag, LLMEvidenceInterpreter(client)).analyze(contract)

    finding = result.finding("probation_period")
    assert finding.status is FindingStatus.REQUIRES_REVIEW
    assert [e.evidence_id for e in finding.regulatory_evidence] == [f"kb-{KB_PROBATION_ART_53}"]
    assert result.status is AnalysisStatus.PARTIAL
    assert [e.code for e in result.errors] == [code]
    assert "sk-test-secret" not in result.model_dump_json()


def test_llm_sees_only_the_extracted_fact_and_the_retrieved_evidence(contract_en, fake_rag, knowledge_base):
    llm = FakeLLMClient()
    ContractAnalysisAgent(fake_rag, LLMEvidenceInterpreter(llm)).analyze(contract_en)

    call = next(c for c in llm.calls if c["payload"]["contract_fact"]["field"] == "probation_period")
    payload = call["payload"]
    assert payload["contract_fact"]["raw_value"] == "ninety (90) days"
    assert [e["evidence_id"] for e in payload["evidence"]] == [f"kb-{KB_PROBATION_ART_53}"]
    assert payload["evidence"][0]["arabic_text"] == knowledge_base[KB_PROBATION_ART_53 - 1]["arabic_content"]
    assert "Use ONLY the contract facts and the regulatory evidence" in call["system"]
    assert "machine translation" in call["system"]
    # every LLM call happens after evidence retrieval and only for fields with relevant evidence
    assert set(llm.fields_called()) <= {f for c in ContractAnalysisAgent(fake_rag).analyze(contract_en).regulatory_checks
                                        if c.status == "evidence_found" for f in c.triggered_by_fields}


def test_notice_period_interpretation_receives_related_contract_facts(contract_en, fake_rag):
    llm = FakeLLMClient()
    ContractAnalysisAgent(fake_rag, LLMEvidenceInterpreter(llm)).analyze(contract_en)
    payload = next(c["payload"] for c in llm.calls if c["payload"]["contract_fact"]["field"] == "notice_period")
    assert {f["field"] for f in payload["related_contract_facts"]} <= {"contract_type", "salary"}
    assert payload["related_contract_facts"]


# --------------------------------------------------------------------------- model guarantees
def test_finding_model_rejects_compliance_labels_without_grounded_evidence(contract_en, fake_rag):
    finding = ContractAnalysisAgent(fake_rag).analyze(contract_en).finding("probation_period")
    data = finding.model_dump()
    for status in ("compliant", "non_compliant"):
        with pytest.raises(ValueError, match="compliance label"):
            AnalysisFinding.model_validate({**data, "status": status})
    not_found = ContractAnalysisAgent(fake_rag).analyze(contract_en).finding("total_salary").model_dump()
    with pytest.raises(ValueError, match="not_found"):
        AnalysisFinding.model_validate({**not_found, "status": "non_compliant"})
    with pytest.raises(ValueError, match="not_found"):
        AnalysisFinding.model_validate({**data, "status": "not_found"})


def test_logs_contain_no_contract_content_or_secrets(contract_en, fake_rag, caplog):
    llm = FakeLLMClient(error=RuntimeError("sk-live-SECRET"))
    with caplog.at_level(logging.DEBUG, logger="sanad"):
        ContractAnalysisAgent(fake_rag, LLMEvidenceInterpreter(llm)).analyze(contract_en)
    text = caplog.text
    assert "Jordan Sample" not in text and "12,000" not in text and "sk-live-SECRET" not in text


def test_default_interpreter_is_no_interpreter(fake_rag):
    assert isinstance(ContractAnalysisAgent(fake_rag).interpreter, NoInterpreter)
