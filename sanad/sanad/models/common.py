"""Status and error types shared by every Sanad component."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ResultStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ERROR = "error"


class ErrorInfo(BaseModel):
    """An explicit, non-hidden failure description."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(description="Machine-readable error code, e.g. 'vector_db_unreachable'.")
    stage: str = Field(description="Where it failed, e.g. 'initialization', 'retrieval', 'health_check'.")
    message: str
    exception_type: str | None = None
    exception_chain: list[str] = Field(default_factory=list)
