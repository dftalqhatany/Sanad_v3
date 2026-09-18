"""Programmatic interface (adapter) over the EXISTING hr_assistant RAG.

Deliberately NOT done here: re-implementing retrieval, embeddings, chunking, prompts or answer
generation; writing to Qdrant; re-ingesting; creating another knowledge base.

What it does:
  1. lazily imports the legacy `chatbot_backend` module (which builds the legacy HybridRetriever
     over the `saudi_labor_law` Qdrant collection),
  2. calls the legacy functions unchanged,
  3. converts their output into typed, traceable Pydantic results,
  4. reports every failure as an explicit status + ErrorInfo instead of crashing or hiding it.

Two entry points:
  * retrieve_evidence(): HybridRetriever.retrieve() only -> regulatory evidence, no LLM, no API key.
  * ask():               answer_policy_question()       -> legacy answer + the evidence it used.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from sanad.config import SanadSettings
from sanad.models.common import ErrorInfo, ResultStatus
from sanad.models.regulatory import (
    RagHealth,
    RegulatoryAnswerResult,
    RegulatoryEvidenceResult,
    RegulatoryQuery,
    RetrievalConfig,
    SourceDocument,
)
from sanad.rag.errors import RagErrorCode, classify_exception, make_error
from sanad.rag.legacy_config import LegacyConfigError, read_legacy_answer_model, read_legacy_retrieval_config
from sanad.rag.legacy_loader import LegacyRagLoader
from sanad.rag.mapping import evidence_from_legacy_reference, evidence_from_retriever_result

logger = logging.getLogger(__name__)

LEGACY_RETRIEVE_ENTRY_POINT = "hr_assistant/hybird_search.py::HybridRetriever.retrieve (via chatbot_backend.get_retriever)"
LEGACY_ANSWER_ENTRY_POINT = "hr_assistant/chatbot_backend.py::answer_policy_question"

_R = TypeVar("_R", bound=RegulatoryEvidenceResult)


class RegulatoryRAGAdapter:
    def __init__(
        self,
        settings: SanadSettings | None = None,
        loader: LegacyRagLoader | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.settings = settings or SanadSettings.from_env()
        self._loader = loader or LegacyRagLoader(self.settings.legacy_rag_dir, self.settings.legacy_backend_module)
        self._clock = clock
        self._config_warnings: list[str] = []
        try:
            self._retrieval_config: RetrievalConfig | None = read_legacy_retrieval_config(
                self.settings.retrieval_module_path
            )
        except LegacyConfigError as exc:
            self._retrieval_config = None
            self._config_warnings.append(f"{RagErrorCode.RETRIEVAL_CONFIG_UNREADABLE.value}: {exc}")
        self._answer_model = read_legacy_answer_model(self.settings.backend_module_path)
        self._source = SourceDocument(
            title_ar=self.settings.source_title_ar,
            title_en=self.settings.source_title_en,
            publisher=self.settings.source_publisher,
            url=self.settings.source_url,
            file_path=str(self.settings.source_document_path),
            knowledge_base_file=str(self.settings.knowledge_base_path),
            vector_collection=self._retrieval_config.collection if self._retrieval_config else None,
            english_content_is_machine_translation=self.settings.english_is_machine_translation,
        )

    # ------------------------------------------------------------------ properties
    @property
    def retrieval_config(self) -> RetrievalConfig | None:
        return self._retrieval_config

    @property
    def source_document(self) -> SourceDocument:
        return self._source

    @property
    def is_backend_loaded(self) -> bool:
        return self._loader.is_loaded

    # ------------------------------------------------------------------ query helpers
    @staticmethod
    def compose_retrieval_query(query: RegulatoryQuery) -> str:
        """Text sent to the existing RAG: the question, followed by the contract clause when given."""
        if not query.contract_context:
            return query.question
        label = f"Contract Clause ({query.clause_name})" if query.clause_name else "Contract Clause"
        return f"{query.question}\n\n{label}:\n{query.contract_context}"

    @staticmethod
    def _coerce(query: RegulatoryQuery | str) -> RegulatoryQuery:
        return query if isinstance(query, RegulatoryQuery) else RegulatoryQuery(question=query)

    # ------------------------------------------------------------------ public API
    def retrieve_evidence(self, query: RegulatoryQuery | str, top_k: int | None = None) -> RegulatoryEvidenceResult:
        """Regulatory evidence from the existing hybrid retriever (no LLM call, no API key needed)."""
        query = self._coerce(query)
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be >= 1")
        started = self._clock()
        warnings: list[str] = []
        if query.employee_data:
            warnings.append("employee_data is only applied by the legacy answer pipeline (ask()); it was not used for retrieval.")
        retrieval_query = self.compose_retrieval_query(query)
        common = dict(query=query, retrieval_query=retrieval_query, entry_point=LEGACY_RETRIEVE_ENTRY_POINT,
                      warnings=warnings, started=started)

        try:
            backend = self._loader.load()
        except Exception as exc:
            return self._failed(RegulatoryEvidenceResult, exc=exc, stage="initialization", **common)

        language = self._detect_language(backend, retrieval_query)
        try:
            retriever = backend.get_retriever()
            raw = retriever.retrieve(retrieval_query) if top_k is None else retriever.retrieve(retrieval_query, top_k=top_k)
            if not isinstance(raw, (list, tuple)):
                raise TypeError(f"HybridRetriever.retrieve returned {type(raw).__name__}, expected list")
            evidence = [evidence_from_retriever_result(item, rank=i, source=self._source) for i, item in enumerate(raw, 1)]
        except Exception as exc:
            return self._failed(RegulatoryEvidenceResult, exc=exc, stage="retrieval", language=language, **common)

        if any(item.score > 0 for item in evidence):
            status = ResultStatus.SUCCESS
        else:
            status = ResultStatus.INSUFFICIENT_EVIDENCE
            warnings.append("The existing retriever returned no article with a positive score for this query.")
        if status is ResultStatus.SUCCESS and not all(item.metadata_complete for item in evidence):
            status = ResultStatus.PARTIAL
            warnings.append("Some evidence items are missing article identity metadata.")
        return RegulatoryEvidenceResult(
            status=status,
            query=query,
            retrieval_query=retrieval_query,
            detected_language=language,
            evidence=evidence,
            retrieval_config=self._retrieval_config,
            legacy_entry_point=LEGACY_RETRIEVE_ENTRY_POINT,
            warnings=[*self._config_warnings, *warnings],
            duration_ms=self._elapsed(started),
        )

    def ask(self, query: RegulatoryQuery | str, api_key: str | None = None) -> RegulatoryAnswerResult:
        """Answer from the existing answer_policy_question(), plus the evidence it was based on."""
        query = self._coerce(query)
        started = self._clock()
        warnings: list[str] = []
        retrieval_query = self.compose_retrieval_query(query)
        common = dict(query=query, retrieval_query=retrieval_query, entry_point=LEGACY_ANSWER_ENTRY_POINT,
                      warnings=warnings, started=started)

        key = api_key or self.settings.openai_api_key
        if not key:
            return self._failed(
                RegulatoryAnswerResult,
                error=make_error(
                    RagErrorCode.MISSING_API_KEY,
                    "answer_generation",
                    "OPENAI_API_KEY is not set, so the existing RAG cannot generate an answer. "
                    "Use retrieve_evidence() for retrieval without the LLM.",
                ),
                stage="answer_generation",
                answer_model=self._answer_model,
                **common,
            )

        try:
            backend = self._loader.load()
        except Exception as exc:
            return self._failed(RegulatoryAnswerResult, exc=exc, stage="initialization",
                                answer_model=self._answer_model, **common)

        language = self._detect_language(backend, retrieval_query)
        try:
            output = backend.answer_policy_question(retrieval_query, query.employee_data, api_key=key)
            if not (isinstance(output, tuple) and len(output) == 2):
                raise TypeError(f"answer_policy_question returned {type(output).__name__}, expected (answer, references)")
            answer, references = output
            if not isinstance(answer, str) or not isinstance(references, list):
                raise TypeError("answer_policy_question returned unexpected types for (answer, references)")
            documents = getattr(backend, "documents", None)
            evidence = []
            for rank, reference in enumerate(references, 1):
                item, warning = evidence_from_legacy_reference(reference, rank=rank, source=self._source, documents=documents)
                evidence.append(item)
                if warning:
                    warnings.append(warning)
        except Exception as exc:
            return self._failed(RegulatoryAnswerResult, exc=exc, stage="retrieval_and_answer", language=language,
                                answer_model=self._answer_model, **common)

        message = None
        if not evidence:
            status, message, answer = ResultStatus.INSUFFICIENT_EVIDENCE, answer, None
        elif not any(item.score > 0 for item in evidence):
            status = ResultStatus.INSUFFICIENT_EVIDENCE
            warnings.append("The answer was generated from articles with no positive retrieval score; treat it as unsupported.")
        elif not all(item.metadata_complete for item in evidence):
            status = ResultStatus.PARTIAL
        else:
            status = ResultStatus.SUCCESS
        return RegulatoryAnswerResult(
            status=status,
            query=query,
            retrieval_query=retrieval_query,
            detected_language=language,
            evidence=evidence,
            retrieval_config=self._retrieval_config,
            legacy_entry_point=LEGACY_ANSWER_ENTRY_POINT,
            warnings=[*self._config_warnings, *warnings],
            duration_ms=self._elapsed(started),
            answer=answer,
            answer_model=self._answer_model,
            message=message,
        )

    def evidence_for_clause(
        self, question: str, clause_text: str, clause_name: str | None = None, top_k: int | None = None
    ) -> RegulatoryEvidenceResult:
        """Convenience for agents: regulatory evidence for one contract clause."""
        return self.retrieve_evidence(
            RegulatoryQuery(question=question, contract_context=clause_text, clause_name=clause_name), top_k=top_k
        )

    def ask_about_clause(
        self, question: str, clause_text: str, clause_name: str | None = None, api_key: str | None = None
    ) -> RegulatoryAnswerResult:
        """Convenience for agents: legacy answer + evidence for one contract clause."""
        return self.ask(
            RegulatoryQuery(question=question, contract_context=clause_text, clause_name=clause_name), api_key=api_key
        )

    def health(self, *, check_vector_db: bool = True, qdrant_url: str | None = None) -> RagHealth:
        """Read-only readiness check. Never imports the legacy module and never writes to Qdrant."""
        errors: list[ErrorInfo] = []
        warnings = list(self._config_warnings)
        settings = self.settings
        files_present = settings.backend_module_path.is_file() and settings.retrieval_module_path.is_file()
        if not files_present:
            errors.append(make_error(RagErrorCode.LEGACY_RAG_NOT_FOUND, "health_check",
                                     f"Existing RAG files not found under {settings.legacy_rag_dir}"))

        kb_count = None
        try:
            with open(settings.knowledge_base_path, encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                kb_count = len(data)
            else:
                errors.append(make_error(RagErrorCode.KNOWLEDGE_BASE_INVALID, "health_check",
                                         f"{settings.knowledge_base_path} does not contain a list of articles"))
        except FileNotFoundError as exc:
            errors.append(classify_exception(exc, "health_check"))
        except (OSError, ValueError) as exc:
            errors.append(make_error(RagErrorCode.KNOWLEDGE_BASE_INVALID, "health_check",
                                     f"Cannot read {settings.knowledge_base_path}: {exc}"))

        config = self._retrieval_config
        url = qdrant_url or (config.qdrant_url if config else None)
        collection = config.collection if config else None
        reachable = exists = points = size = distance = None
        if check_vector_db:
            if not url or not collection:
                errors.append(make_error(RagErrorCode.RETRIEVAL_CONFIG_UNREADABLE, "health_check",
                                         "Qdrant URL/collection could not be read from the existing RAG"))
            else:
                reachable, exists, points, size, distance = self._check_qdrant(url, collection, errors)
                if kb_count is not None and points is not None and points != kb_count:
                    warnings.append(
                        f"Collection '{collection}' has {points} points but the knowledge base has {kb_count} articles "
                        "(expected when long articles were split into several nodes; otherwise the collection changed)."
                    )

        status = ResultStatus.ERROR if errors else (ResultStatus.PARTIAL if warnings else ResultStatus.SUCCESS)
        return RagHealth(
            status=status,
            legacy_rag_dir=str(settings.legacy_rag_dir),
            legacy_files_present=files_present,
            legacy_backend_loaded=self._loader.is_loaded,
            knowledge_base_article_count=kb_count,
            retrieval_config=config,
            vector_db_url=url,
            vector_db_reachable=reachable,
            collection_exists=exists,
            collection_points_count=points,
            collection_vector_size=size,
            collection_distance=distance,
            errors=errors,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ internals
    def _check_qdrant(self, url: str, collection: str, errors: list[ErrorInfo]):
        reachable = exists = points = size = distance = None
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            errors.append(classify_exception(exc, "health_check"))
            return reachable, exists, points, size, distance
        client = None
        try:
            timeout = max(1, int(round(self.settings.health_timeout_s)))
            try:
                client = QdrantClient(url=url, timeout=timeout, check_compatibility=False)
            except TypeError:  # qdrant-client versions without check_compatibility
                client = QdrantClient(url=url, timeout=timeout)
            names = {c.name for c in client.get_collections().collections}  # read-only
            reachable = True
            exists = collection in names
            if not exists:
                errors.append(make_error(RagErrorCode.COLLECTION_NOT_FOUND, "health_check",
                                         f"Collection '{collection}' not found at {url}; available: {sorted(names)}"))
            else:
                info = client.get_collection(collection)  # read-only
                points = info.points_count
                vectors = info.config.params.vectors
                if hasattr(vectors, "size"):
                    size = int(vectors.size)
                    distance = str(getattr(vectors.distance, "value", vectors.distance))
        except Exception as exc:
            error = classify_exception(exc, "health_check", collection=collection, qdrant_url=url)
            if error.code == RagErrorCode.VECTOR_DB_UNREACHABLE.value:
                reachable = False
            errors.append(error)
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:  # closing a failed client must not mask the real error
                    pass
        return reachable, exists, points, size, distance

    @staticmethod
    def _detect_language(backend: Any, text: str):
        try:
            language = backend.detect_language(text)
        except Exception:
            return None
        return language if language in ("ar", "en") else None

    def _elapsed(self, started: float) -> float:
        return round((self._clock() - started) * 1000, 3)

    def _failed(
        self,
        result_cls: type[_R],
        *,
        query: RegulatoryQuery,
        retrieval_query: str,
        entry_point: str,
        warnings: list[str],
        started: float,
        stage: str,
        exc: BaseException | None = None,
        error: ErrorInfo | None = None,
        language: str | None = None,
        **extra: Any,
    ) -> _R:
        if error is None:
            config = self._retrieval_config
            error = classify_exception(
                exc, stage,
                collection=config.collection if config else None,
                qdrant_url=config.qdrant_url if config else None,
            )
        logger.error("Regulatory RAG failure [%s/%s]: %s", error.stage, error.code, error.message)
        return result_cls(
            status=ResultStatus.ERROR,
            query=query,
            retrieval_query=retrieval_query,
            detected_language=language,
            evidence=[],
            retrieval_config=self._retrieval_config,
            legacy_entry_point=entry_point,
            errors=[error],
            warnings=[*self._config_warnings, *warnings],
            duration_ms=self._elapsed(started),
            **extra,
        )
