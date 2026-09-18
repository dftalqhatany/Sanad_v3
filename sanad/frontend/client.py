"""HTTP client for the Sanad API. The user interface talks to the API and to nothing else.

No Sanad agent, orchestrator, parser or RAG module is imported here on purpose: the frontend knows
only JSON over HTTP, so the routing and analysis rules stay in one place (the backend).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_S = 180.0


@dataclass
class ApiResponse:
    ok: bool
    status_code: int
    result: dict[str, Any] | None = None       # OrchestratorResult as JSON
    error: dict[str, Any] | None = None        # {"code", "message", "detail"}

    @property
    def message(self) -> str:
        if self.error:
            detail = f" ({self.error['detail']})" if self.error.get("detail") else ""
            return f"{self.error.get('message', 'The request failed.')}{detail}"
        return (self.result or {}).get("summary", "")


@dataclass
class SanadApiClient:
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = DEFAULT_TIMEOUT_S
    http_client: Any = field(default=None, repr=False, compare=False)  # tests inject one; production opens its own
    _client: httpx.Client | None = field(default=None, repr=False, compare=False)

    def _http(self) -> httpx.Client:
        if self.http_client is not None:
            return self.http_client
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url.rstrip("/"), timeout=self.timeout_s)
        return self._client

    # ------------------------------------------------------------------ calls
    def config(self) -> ApiResponse:
        return self._send("GET", "/api/config")

    def health(self) -> ApiResponse:
        return self._send("GET", "/api/health")

    def ask(self, question: str) -> ApiResponse:
        return self._send("POST", "/api/ask", json={"question": question})

    def analyze(self, uploads: list[tuple[str, bytes, str]], *, question: str | None = None, task: str = "auto",
                priorities: list[str] | None = None, target_job_json: str | None = None,
                labels: list[str] | None = None, salary: dict[str, str] | None = None) -> ApiResponse:
        """uploads: (filename, content, role). Everything else is sent as form fields."""
        files = [("files", (name, data, _content_type(name))) for name, data, _ in uploads]
        form: dict[str, Any] = {"task": task, "roles": [role for _, _, role in uploads]}
        if labels:
            form["labels"] = list(labels)
        if priorities:
            form["priorities"] = list(priorities)
        if question:
            form["question"] = question
        if target_job_json:
            form["target_job"] = target_job_json
        for key, value in (salary or {}).items():
            if value:
                form[key] = value
        return self._send("POST", "/api/analyze", files=files or None, data=form)

    # ------------------------------------------------------------------ internals
    def _send(self, method: str, path: str, **kwargs) -> ApiResponse:
        try:
            response = self._http().request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            return ApiResponse(ok=False, status_code=0,
                               error={"code": "api_unreachable",
                                      "message": f"The Sanad API at {self.base_url} could not be reached.",
                                      "detail": type(exc).__name__})
        try:
            payload = response.json()
        except ValueError:
            return ApiResponse(ok=False, status_code=response.status_code,
                               error={"code": "invalid_response", "message": "The API returned a non-JSON response."})
        if isinstance(payload, dict) and "error" in payload:
            return ApiResponse(ok=False, status_code=response.status_code, error=payload["error"])
        return ApiResponse(ok=response.status_code < 400, status_code=response.status_code, result=payload)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def _content_type(filename: str) -> str:
    return ("application/pdf" if filename.lower().endswith(".pdf")
            else "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
