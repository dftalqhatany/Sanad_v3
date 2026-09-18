"""Raw failures of the existing RAG become explicit, specific error codes (original details kept)."""

from sanad.rag.errors import InvalidLegacyOutputError, LegacyInterfaceError, LegacyRagNotFoundError, classify_exception
from tests.fakes.legacy_stubs import foreign_exception

COLLECTION = "saudi_labor_law"
URL = "http://localhost:6333"


def classify(exc, stage="retrieval"):
    return classify_exception(exc, stage, collection=COLLECTION, qdrant_url=URL)


def test_real_qdrant_connection_failure_is_vector_db_unreachable(real_qdrant_connection_error):
    error = classify(real_qdrant_connection_error, "initialization")
    assert error.code == "vector_db_unreachable"
    assert error.stage == "initialization"
    assert error.exception_type.endswith("ResponseHandlingException")
    assert error.exception_chain, "the original exception chain must be preserved"
    assert "Connection refused" in error.message or "ConnectError" in " ".join(error.exception_chain)


def test_missing_collection_is_collection_not_found():
    exc = foreign_exception(
        "qdrant_client.http.exceptions", "UnexpectedResponse",
        "Unexpected Response: 404 (Not Found) Raw response content: Collection `saudi_labor_law` doesn't exist!",
    )
    assert classify(exc).code == "collection_not_found"


def test_other_qdrant_error_is_request_failed_not_unreachable():
    exc = foreign_exception("qdrant_client.http.exceptions", "UnexpectedResponse", "Unexpected Response: 400 (Bad Request)")
    assert classify(exc).code == "vector_db_request_failed"


def test_embedding_model_download_failure_is_not_mistaken_for_qdrant():
    exc = OSError("We couldn't connect to 'https://huggingface.co' to load the files for intfloat/multilingual-e5-base.")
    assert classify(exc, "initialization").code == "embedding_model_unavailable"
    hub = foreign_exception("huggingface_hub.errors", "LocalEntryNotFoundError", "cannot find the requested files",
                            cause=ConnectionError("proxy refused"))
    assert classify(hub, "initialization").code == "embedding_model_unavailable"


def test_openai_failure_is_llm_error_not_vector_db():
    exc = foreign_exception("openai", "APIConnectionError", "Connection error.",
                            cause=foreign_exception("httpx", "ConnectError", "[Errno 111] Connection refused"))
    assert classify(exc, "retrieval_and_answer").code == "llm_request_failed"


def test_missing_dependency_names_the_module():
    error = classify(ModuleNotFoundError("No module named 'llama_index'", name="llama_index"), "initialization")
    assert error.code == "dependency_missing"
    assert "llama_index" in error.message


def test_missing_knowledge_base_file():
    exc = FileNotFoundError(2, "No such file or directory", "data/labor_law/labor_law_parsed.json")
    assert classify(exc, "initialization").code == "knowledge_base_missing"


def test_adapter_specific_errors():
    assert classify(LegacyRagNotFoundError("gone"), "initialization").code == "legacy_rag_not_found"
    assert classify(LegacyInterfaceError("changed"), "initialization").code == "legacy_interface_changed"
    assert classify(InvalidLegacyOutputError("bad"), "retrieval").code == "invalid_legacy_output"


def test_unknown_error_falls_back_to_stage_code_and_keeps_details():
    error = classify(RuntimeError("something odd"), "retrieval")
    assert error.code == "retrieval_failed"
    assert "something odd" in error.message
    assert error.exception_type == "RuntimeError"
