"""The adapter drives the REAL rag/ code and knowledge base (infrastructure simulated).

Only the external infrastructure (Qdrant server, e5 model, OpenAI) is simulated; see tests/fakes.
Live infrastructure is covered by tests/live.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


from config import SanadSettings
from models import ResultStatus
from rag import RegulatoryRAGAdapter
from tests.contract.conftest import TEST_KEY, legacy_backend
from tests.fakes.infra_stubs import foreign_exception

ANNUAL_LEAVE = 111  # kb index of "المادة التاسعة بعد المائة" (Article 109)


def lexical_query(knowledge_base, kb_index, n_tokens=12):
    """Tokens copied from an article so the real BM25 component has lexical overlap."""
    return " ".join(knowledge_base[kb_index - 1]["arabic_content"].split()[:n_tokens])


# --------------------------------------------------------------------------- retrieval
def test_retrieve_evidence_returns_exactly_what_the_legacy_hybrid_retriever_returns(adapter, infra, knowledge_base, rag_dir):
    infra.dense_hits = [(ANNUAL_LEAVE, 0.87), (112, 0.85), (40, 0.81)]
    query = lexical_query(knowledge_base, ANNUAL_LEAVE)
    cwd, path = os.getcwd(), list(sys.path)

    result = adapter.retrieve_evidence(query)

    assert result.status is ResultStatus.SUCCESS, result.errors
    assert adapter.is_backend_loaded
    assert Path(legacy_backend().__file__).resolve().parent == rag_dir.resolve()
    assert os.getcwd() == cwd and sys.path == path, "loader must restore cwd and sys.path"
    # the legacy module was initialised with its own, unchanged configuration
    assert infra.embed_model_names == ["intfloat/multilingual-e5-base"]
    assert set(infra.collections) == {"saudi_labor_law"}
    assert infra.qdrant_urls == ["http://localhost:6333"]
    assert infra.dense_top_k == [3]
    assert infra.dense_queries == [query]

    direct = legacy_backend().get_retriever().retrieve(query)
    assert [(e.raw_metadata["index"], e.score) for e in result.evidence] == [
        (int(r["metadata"]["index"]), float(r["score"])) for r in direct
    ]
    assert [e.rank for e in result.evidence] == [1, 2, 3, 4, 5]
    assert result.legacy_entry_point.startswith("rag/retriever.py::HybridRetriever.retrieve")
    json.loads(result.model_dump_json())  # transportable between agents


def test_evidence_preserves_article_and_source_metadata(adapter, infra, knowledge_base):
    infra.dense_hits = [(ANNUAL_LEAVE, 0.9), (98, 0.8)]
    result = adapter.retrieve_evidence(lexical_query(knowledge_base, ANNUAL_LEAVE))
    assert result.status is ResultStatus.SUCCESS
    for item in result.evidence:
        article = knowledge_base[item.reference.kb_index - 1]
        ref = item.reference
        assert item.raw_metadata == article
        assert (ref.article_name_ar, ref.article_number, ref.article_number_ar, ref.english_number) == (
            article["arabic_name"], article["article_number"], article["number_ar"], article["english_number"])
        assert (ref.part_number, ref.part_number_ar, ref.part_title_ar) == (
            article["part_number"], article["part_number_ar"], article["part_title_ar"])
        assert (ref.chapter_number, ref.chapter_number_ar, ref.chapter_title_ar) == (
            article["chapter_number"], article["chapter_number_ar"], article["chapter_title_ar"])
        assert item.arabic_content == article["arabic_content"]
        assert item.english_content == article["english_content"]
        assert article["arabic_name"] in item.citation and article["english_number"] in item.citation
        assert item.metadata_complete and not item.score_is_rounded
        assert item.source.vector_collection == "saudi_labor_law"
        assert item.source.title_ar == "نظام العمل"
        assert item.source.knowledge_base_file.endswith("labor_law_parsed.json")
        assert item.source.english_content_is_machine_translation is True
    assert result.retrieval_config.collection == "saudi_labor_law"
    assert result.retrieval_config.alpha == 0.6 and result.retrieval_config.top_k == 5


def test_contract_clause_context_is_sent_to_the_existing_rag(adapter, infra):
    infra.dense_hits = [(ANNUAL_LEAVE, 0.9)]
    result = adapter.evidence_for_clause(
        question="Does this probation period comply with the Saudi Labor Law?",
        clause_text="The employee shall be subject to a probation period of 180 days.",
        clause_name="Probation period",
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.retrieval_query == (
        "Does this probation period comply with the Saudi Labor Law?\n\n"
        "Contract Clause (Probation period):\nThe employee shall be subject to a probation period of 180 days."
    )
    assert infra.dense_queries == [result.retrieval_query]
    assert result.query.clause_name == "Probation period"
    assert result.detected_language == "en"  # from the legacy detect_language()


def test_no_retrieval_signal_is_insufficient_evidence_not_success(adapter, infra):
    infra.dense_hits = []
    result = adapter.retrieve_evidence("zzqx no lexical or semantic overlap")
    assert result.status is ResultStatus.INSUFFICIENT_EVIDENCE
    assert all(item.score == 0 for item in result.evidence)
    assert any("no article with a positive score" in w for w in result.warnings)


# --------------------------------------------------------------------------- answer pipeline
def test_ask_uses_existing_answer_policy_question_and_returns_its_evidence(adapter, infra, knowledge_base):
    infra.dense_hits = [(ANNUAL_LEAVE, 0.9), (112, 0.86), (40, 0.8)]
    infra.llm_answer = "  Under Article 109 the worker is entitled to 21 days of annual leave.  "
    question = "How many days of annual leave is a worker entitled to?"

    result = adapter.ask(question)

    assert result.status is ResultStatus.SUCCESS, result.errors
    # the post-processing (strip + highlight_articles) came from rag.backend, not from the adapter
    assert result.answer == "Under **Article 109** the worker is entitled to 21 days of annual leave."
    assert result.answer_model == "gpt-4o-mini"
    assert result.legacy_entry_point == "rag/backend.py::answer_policy_question"
    [call] = infra.llm_calls
    assert call["model"] == "gpt-4o-mini" and infra.llm_api_keys == [TEST_KEY]
    prompt = call["messages"][0]["content"]
    assert prompt.startswith("You are an intelligent assistant specialized in Saudi Labor Law.")
    assert f"Question: {question}" in prompt
    for item in result.evidence:  # the LLM saw exactly the evidence we report
        assert item.arabic_content in prompt

    _, legacy_references = legacy_backend().answer_policy_question(question, None, api_key=TEST_KEY)
    assert [item.legacy_reference for item in result.evidence] == legacy_references
    for item in result.evidence:
        assert item.metadata_complete and item.score_is_rounded
        assert item.raw_metadata == knowledge_base[item.reference.kb_index - 1]


def test_ask_in_arabic_uses_the_existing_arabic_prompt(adapter, infra):
    infra.dense_hits = [(ANNUAL_LEAVE, 0.9)]
    result = adapter.ask("كم مدة الإجازة السنوية التي يستحقها العامل؟")
    assert result.status is ResultStatus.SUCCESS
    assert result.detected_language == "ar"
    assert infra.llm_calls[0]["messages"][0]["content"].startswith("أنت مساعد ذكي متخصص في نظام العمل السعودي.")


def test_ask_about_clause_passes_contract_context_and_employee_data_unchanged(adapter, infra):
    infra.dense_hits = [(ANNUAL_LEAVE, 0.9)]
    result = adapter.ask_about_clause(
        question="Is this annual leave clause compliant?",
        clause_text="The employee is entitled to 15 days of paid annual leave per year.",
        clause_name="Annual leave",
    )
    assert result.status is ResultStatus.SUCCESS
    prompt = infra.llm_calls[0]["messages"][0]["content"]
    assert "Contract Clause (Annual leave):\nThe employee is entitled to 15 days of paid annual leave per year." in prompt


# --------------------------------------------------------------------------- failures
def test_ask_without_api_key_is_explicit_and_never_touches_the_rag(rag_dir, infra, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    adapter = RegulatoryRAGAdapter(SanadSettings(rag_dir=rag_dir, openai_api_key=None))
    result = adapter.ask("What is the maximum probation period?")
    assert result.status is ResultStatus.ERROR
    assert [e.code for e in result.errors] == ["missing_api_key"]
    assert result.answer is None and result.evidence == []
    assert not adapter.is_backend_loaded and infra.llm_calls == []


def test_qdrant_unreachable_while_loading_existing_rag_is_explicit_and_recoverable(adapter, infra, real_qdrant_connection_error):
    infra.vector_store_init_error = real_qdrant_connection_error  # raised inside retriever.py at import

    result = adapter.retrieve_evidence("What is the maximum probation period?")

    assert result.status is ResultStatus.ERROR
    [error] = result.errors
    assert (error.code, error.stage) == ("vector_db_unreachable", "initialization")
    assert error.exception_type.endswith("ResponseHandlingException")
    assert result.evidence == [] and not adapter.is_backend_loaded
    assert "rag.backend" not in sys.modules and "rag.retriever" not in sys.modules

    answer = adapter.ask("What is the maximum probation period?")
    assert answer.status is ResultStatus.ERROR and answer.errors[0].code == "vector_db_unreachable"
    assert answer.answer is None and infra.llm_calls == []

    infra.vector_store_init_error = None  # Qdrant is back: the failure was not cached
    infra.dense_hits = [(ANNUAL_LEAVE, 0.9)]
    assert adapter.retrieve_evidence("What is the maximum probation period?").status is ResultStatus.SUCCESS


def test_qdrant_failure_during_a_query_is_explicit_for_both_entry_points(adapter, infra, real_qdrant_connection_error):
    infra.dense_hits = [(ANNUAL_LEAVE, 0.9)]
    assert adapter.retrieve_evidence("warm up").status is ResultStatus.SUCCESS
    infra.dense_retrieve_error = real_qdrant_connection_error

    evidence = adapter.retrieve_evidence("What is the maximum probation period?")
    assert evidence.status is ResultStatus.ERROR
    assert (evidence.errors[0].code, evidence.errors[0].stage) == ("vector_db_unreachable", "retrieval")

    answer = adapter.ask("What is the maximum probation period?")
    assert answer.status is ResultStatus.ERROR
    assert (answer.errors[0].code, answer.errors[0].stage) == ("vector_db_unreachable", "retrieval_and_answer")
    assert infra.llm_calls == []


def test_llm_failure_is_explicit(adapter, infra):
    infra.dense_hits = [(ANNUAL_LEAVE, 0.9)]
    infra.llm_error = foreign_exception("openai", "APIConnectionError", "Connection error.")
    result = adapter.ask("What is the annual leave entitlement?")
    assert result.status is ResultStatus.ERROR
    assert (result.errors[0].code, result.errors[0].stage) == ("llm_request_failed", "retrieval_and_answer")
    assert result.answer is None


def test_missing_knowledge_base_is_explicit(tmp_path, rag_dir, infra):
    for name in ("backend.py", "retriever.py"):
        shutil.copy(rag_dir / name, tmp_path / name)  # copies in a temp dir; originals untouched
    adapter = RegulatoryRAGAdapter(SanadSettings(rag_dir=tmp_path, openai_api_key=TEST_KEY))
    # The knowledge base now travels with the rag package, so a rag_dir whose data/ is missing is
    # reported by the read-only health check rather than at retrieval time.
    health = adapter.health(check_vector_db=False)
    assert health.status is ResultStatus.ERROR
    assert "knowledge_base_missing" in {e.code for e in health.errors}
    assert health.knowledge_base_article_count is None

