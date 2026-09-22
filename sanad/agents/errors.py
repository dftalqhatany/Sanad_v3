"""Explicit error codes of the analysis agent. Messages never contain API keys, prompts or document content."""

from __future__ import annotations

from enum import Enum

from models.common import ErrorInfo


class AnalysisErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    EXTRACTION_NOT_USABLE = "extraction_not_usable"
    RAG_UNAVAILABLE = "rag_unavailable"
    RAG_QUERY_FAILED = "rag_query_failed"
    LLM_REQUEST_FAILED = "llm_request_failed"
    LLM_OUTPUT_INVALID = "llm_output_invalid"
    SALARY_BENCHMARK_FAILED = "salary_benchmark_failed"
    ANALYSIS_FAILED = "analysis_failed"


class ComparisonErrorCode(str, Enum):
    TOO_FEW_CONTRACTS = "too_few_contracts"
    TOO_MANY_CONTRACTS = "too_many_contracts"
    INVALID_INPUT = "invalid_input"
    INVALID_PRIORITY = "invalid_priority"
    CONTRACT_ANALYSIS_FAILED = "contract_analysis_failed"
    COMPARISON_FAILED = "comparison_failed"


class InterpretationError(Exception):
    """Raised by an interpreter; the agent turns it into an explicit error and a 'requires_review' finding."""

    def __init__(self, code: AnalysisErrorCode, message: str, exception_type: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exception_type = exception_type


def analysis_error(code: AnalysisErrorCode | ComparisonErrorCode, stage: str, message: str, exc: BaseException | None = None) -> ErrorInfo:
    return ErrorInfo(
        code=code.value,
        stage=stage,
        message=" ".join(message.split())[:500],
        exception_type=type(exc).__name__ if exc is not None else None,
    )
