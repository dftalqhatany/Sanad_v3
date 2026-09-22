"""Error codes and exception classification for the Regulatory RAG adapter.

The existing RAG connects to Qdrant and loads the embedding model at *import time*, and raises
raw third-party exceptions at call time. This module turns those exceptions into explicit,
machine-readable ErrorInfo objects. The original exception type and message are always kept.
"""

from __future__ import annotations

from enum import Enum
from urllib.parse import urlparse

from models.common import ErrorInfo


class RagErrorCode(str, Enum):
    LEGACY_RAG_NOT_FOUND = "legacy_rag_not_found"
    LEGACY_INTERFACE_CHANGED = "legacy_interface_changed"
    DEPENDENCY_MISSING = "dependency_missing"
    KNOWLEDGE_BASE_MISSING = "knowledge_base_missing"
    KNOWLEDGE_BASE_INVALID = "knowledge_base_invalid"
    EMBEDDING_MODEL_UNAVAILABLE = "embedding_model_unavailable"
    VECTOR_DB_UNREACHABLE = "vector_db_unreachable"
    VECTOR_DB_REQUEST_FAILED = "vector_db_request_failed"
    COLLECTION_NOT_FOUND = "collection_not_found"
    RETRIEVAL_CONFIG_UNREADABLE = "retrieval_config_unreadable"
    MISSING_API_KEY = "missing_api_key"
    LLM_REQUEST_FAILED = "llm_request_failed"
    INVALID_LEGACY_OUTPUT = "invalid_legacy_output"
    RAG_INITIALIZATION_FAILED = "rag_initialization_failed"
    RETRIEVAL_FAILED = "retrieval_failed"
    ANSWER_GENERATION_FAILED = "answer_generation_failed"
    HEALTH_CHECK_FAILED = "health_check_failed"


class LegacyRagNotFoundError(FileNotFoundError):
    """The existing RAG files are not where Sanad expects them."""


class LegacyInterfaceError(RuntimeError):
    """The existing RAG no longer exposes the functions the adapter relies on."""


class InvalidLegacyOutputError(ValueError):
    """The existing RAG returned data in an unexpected shape."""


_CONNECT_EXCEPTION_NAMES = {
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "PoolTimeout",
    "ResponseHandlingException",
    "NewConnectionError",
    "MaxRetryError",
    "ConnectionRefusedError",
}
_CONNECT_PHRASES = (
    "connection refused",
    "failed to establish a new connection",
    "timed out",
    "name or service not known",
    "nodename nor servname",
    "all connection attempts failed",
)
_EMBEDDING_MODULES = ("huggingface_hub", "transformers", "sentence_transformers", "tokenizers")

_MESSAGES = {
    RagErrorCode.LEGACY_RAG_NOT_FOUND: "The existing RAG project could not be found or loaded from the expected location",
    RagErrorCode.LEGACY_INTERFACE_CHANGED: "The existing RAG no longer exposes the interface the adapter depends on",
    RagErrorCode.DEPENDENCY_MISSING: "A Python dependency of the existing RAG is not installed",
    RagErrorCode.KNOWLEDGE_BASE_MISSING: "The regulatory knowledge-base file used by the existing RAG is missing",
    RagErrorCode.EMBEDDING_MODEL_UNAVAILABLE: "The embedding model used by the existing RAG could not be loaded",
    RagErrorCode.VECTOR_DB_UNREACHABLE: "Cannot reach the Qdrant vector database used by the existing RAG (is the Qdrant server running?)",
    RagErrorCode.VECTOR_DB_REQUEST_FAILED: "The Qdrant vector database rejected or failed a request",
    RagErrorCode.COLLECTION_NOT_FOUND: "The regulatory Qdrant collection used by the existing RAG does not exist",
    RagErrorCode.LLM_REQUEST_FAILED: "The existing RAG's LLM (OpenAI) request failed",
    RagErrorCode.INVALID_LEGACY_OUTPUT: "The existing RAG returned output in an unexpected format",
    RagErrorCode.RAG_INITIALIZATION_FAILED: "The existing RAG failed to initialise",
    RagErrorCode.RETRIEVAL_FAILED: "Regulatory retrieval failed",
    RagErrorCode.ANSWER_GENERATION_FAILED: "The existing RAG failed while retrieving or generating the answer",
    RagErrorCode.HEALTH_CHECK_FAILED: "The RAG health check failed",
}

_STAGE_DEFAULT = {
    "initialization": RagErrorCode.RAG_INITIALIZATION_FAILED,
    "retrieval": RagErrorCode.RETRIEVAL_FAILED,
    "retrieval_and_answer": RagErrorCode.ANSWER_GENERATION_FAILED,
    "health_check": RagErrorCode.HEALTH_CHECK_FAILED,
}


def iter_exception_chain(exc: BaseException, limit: int = 10) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(chain) < limit:
        chain.append(current)
        seen.add(id(current))
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return chain


def qualified_name(exc: BaseException) -> str:
    module = type(exc).__module__
    name = type(exc).__qualname__
    return name if module in (None, "builtins") else f"{module}.{name}"


def _from_modules(chain: list[BaseException], prefixes: tuple[str, ...]) -> bool:
    for exc in chain:
        module = type(exc).__module__ or ""
        if any(module == p or module.startswith(p + ".") for p in prefixes):
            return True
    return False


def _host_port(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    if not parsed.hostname:
        return None
    return f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname


def _classify(chain: list[BaseException], stage: str, collection: str | None, qdrant_url: str | None) -> RagErrorCode:
    text = " | ".join(str(e) for e in chain).lower()
    names = {type(e).__name__ for e in chain}

    def has(kind) -> bool:
        return any(isinstance(e, kind) for e in chain)

    if has(LegacyRagNotFoundError):
        return RagErrorCode.LEGACY_RAG_NOT_FOUND
    if has(LegacyInterfaceError):
        return RagErrorCode.LEGACY_INTERFACE_CHANGED
    if has(InvalidLegacyOutputError):
        return RagErrorCode.INVALID_LEGACY_OUTPUT
    if has(FileNotFoundError) and ".json" in text:
        return RagErrorCode.KNOWLEDGE_BASE_MISSING
    if _from_modules(chain, _EMBEDDING_MODULES) or "huggingface" in text:
        return RagErrorCode.EMBEDDING_MODEL_UNAVAILABLE
    if has(ModuleNotFoundError):
        return RagErrorCode.DEPENDENCY_MISSING
    if _from_modules(chain, ("openai",)):
        return RagErrorCode.LLM_REQUEST_FAILED
    if collection and collection.lower() in text and any(
        phrase in text for phrase in ("not found", "doesn't exist", "does not exist")
    ):
        return RagErrorCode.COLLECTION_NOT_FOUND
    # Embedding-model and OpenAI network failures were handled above, so the only remaining network
    # dependency of the existing RAG is Qdrant.
    host_port = _host_port(qdrant_url)
    connectivity = (
        bool(names & _CONNECT_EXCEPTION_NAMES)
        or has(ConnectionError)
        or has(TimeoutError)
        or any(phrase in text for phrase in _CONNECT_PHRASES)
        or (host_port is not None and host_port in text and _from_modules(chain, ("httpx", "httpcore", "grpc")))
    )
    if connectivity:
        return RagErrorCode.VECTOR_DB_UNREACHABLE
    if _from_modules(chain, ("qdrant_client",)):
        return RagErrorCode.VECTOR_DB_REQUEST_FAILED
    return _STAGE_DEFAULT.get(stage, RagErrorCode.RETRIEVAL_FAILED)


def classify_exception(
    exc: BaseException,
    stage: str,
    *,
    collection: str | None = None,
    qdrant_url: str | None = None,
) -> ErrorInfo:
    chain = iter_exception_chain(exc)
    code = _classify(chain, stage, collection, qdrant_url)
    detail = f"{qualified_name(exc)}: {exc}"
    if isinstance(exc, ModuleNotFoundError) and exc.name:
        detail = f"missing module '{exc.name}' ({detail})"
    return ErrorInfo(
        code=code.value,
        stage=stage,
        message=f"{_MESSAGES.get(code, code.value)}. {detail}"[:1500],
        exception_type=qualified_name(exc),
        exception_chain=[f"{qualified_name(e)}: {e}"[:500] for e in chain],
    )


def make_error(code: RagErrorCode, stage: str, message: str) -> ErrorInfo:
    return ErrorInfo(code=code.value, stage=stage, message=message)
