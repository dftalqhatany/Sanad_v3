"""Explicit error codes of the Orchestrator. Messages never contain document content or secrets."""

from __future__ import annotations

from enum import Enum

from sanad.agents.errors import analysis_error
from sanad.models.common import ErrorInfo


class OrchestratorErrorCode(str, Enum):
    NOTHING_TO_DO = "nothing_to_do"                    # no question and no usable document
    UNUSABLE_DOCUMENT = "unusable_document"            # unsupported / scanned / unreadable upload
    DOCUMENT_ROLE_UNDETERMINED = "document_role_undetermined"
    NO_CONTRACT_UPLOADED = "no_contract_uploaded"
    TOO_MANY_CONTRACTS = "too_many_contracts"
    QUESTION_MISSING = "question_missing"
    TASK_NOT_POSSIBLE = "task_not_possible"            # the requested task does not fit the uploaded documents
    ROUTED_AGENT_FAILED = "routed_agent_failed"
    ORCHESTRATION_FAILED = "orchestration_failed"


def orchestrator_error(code: OrchestratorErrorCode, stage: str, message: str,
                       exc: BaseException | None = None) -> ErrorInfo:
    return analysis_error(code, stage, message, exc)
