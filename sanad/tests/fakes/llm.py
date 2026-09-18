"""Scripted LLM client for the evidence interpreter. No network, no OpenAI package, no API key."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Responder = Callable[[dict[str, Any]], "dict[str, Any] | str"]


@dataclass
class FakeLLMClient:
    responder: Responder | dict[str, Any] | str | None = None
    error: BaseException | None = None
    model: str = "fake-llm"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete_json(self, *, system: str, user: str) -> str:
        payload = json.loads(user)
        self.calls.append({"system": system, "user": user, "payload": payload})
        if self.error is not None:
            raise self.error
        response = self.responder(payload) if callable(self.responder) else self.responder
        if response is None:
            response = {"assessment": "requires_review", "explanation": "Scripted default.",
                        "cited_evidence_ids": [e["evidence_id"] for e in payload["evidence"]]}
        return response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)

    def fields_called(self) -> list[str]:
        return [call["payload"]["contract_fact"]["field"] for call in self.calls]
