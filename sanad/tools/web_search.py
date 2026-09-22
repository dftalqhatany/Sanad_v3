"""Web search and page retrieval for salary benchmarking.

    WebSearchClient  ->  SalaryBenchmarkProvider  ->  SalaryBenchmark

The client knows about HTTP, search providers, allowed domains, timeouts, rate limiting and caching.
It knows nothing about salaries; the provider knows nothing about HTTP. Tests inject a fake client.

Only domains on the configured allowlist are searched, fetched or returned, and every result carries
the URL, title, snippet, domain and retrieval date that the citation is later built from.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import urlparse

from config import SalarySettings

logger = logging.getLogger(f"sanad.{__name__}")  # one "sanad" logging namespace, as before the flattening

SEARCH_ENDPOINTS = {"tavily": "https://api.tavily.com/search", "brave": "https://api.search.brave.com/res/v1/web/search"}


@dataclass(frozen=True)
class WebSearchResult:
    """One page the client found or read. `content` is plain text (or raw JSON) when the page was fetched."""

    query: str
    url: str
    title: str = ""
    snippet: str = ""
    content: str = ""
    domain: str = ""
    retrieved_at: str = ""
    content_type: str = "text/html"

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.snippet}\n{self.content}".strip()


class WebSearchError(Exception):
    """The search itself failed (no network, timeout, provider error, bad key)."""

    def __init__(self, code: str, message: str, exception_type: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exception_type = exception_type


class WebSearchClient(Protocol):
    name: str

    def search(self, query: str, *, max_results: int | None = None) -> list[WebSearchResult]: ...


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def is_allowed(url: str, allowed_domains: tuple[str, ...]) -> bool:
    host = domain_of(url)
    return any(host == domain or host.endswith("." + domain) for domain in allowed_domains)


def _now() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class NullWebSearchClient:
    """Used when nothing is configured: no request is made and the provider reports 'not_configured'."""

    name = "none"

    def search(self, query: str, *, max_results: int | None = None) -> list[WebSearchResult]:
        return []


@dataclass
class HttpWebSearchClient:
    """Real runtime search: a search API when one is configured, plus the documented seed sources.

    Guardrails: allowlisted domains only, per-request timeout, a minimum delay between requests,
    a bounded number of results, a TTL cache per query and a cap on the text kept per page.
    """

    settings: SalarySettings
    http: Any = None                      # an httpx.Client; created lazily so tests can inject one
    clock: Any = time.monotonic
    today: Any = _now
    name: str = "http"
    calls: list[str] = field(default_factory=list, repr=False)
    _cache: dict[str, tuple[float, list[WebSearchResult]]] = field(default_factory=dict, repr=False)
    _last_request: float = field(default=0.0, repr=False)

    # ------------------------------------------------------------------ public
    def search(self, query: str, *, max_results: int | None = None) -> list[WebSearchResult]:
        limit = max_results or self.settings.max_results
        cached = self._cached(query)
        if cached is not None:
            return cached[:limit]

        urls: list[tuple[str, str, str]] = []  # (url, title, snippet)
        if self.settings.provider != "none" and self.settings.api_key:
            urls += self._provider_results(query, limit)
        urls += [(url, "", "") for url in self.settings.seed_urls]

        seen: set[str] = set()
        results: list[WebSearchResult] = []
        for url, title, snippet in urls:
            if url in seen or len(results) >= limit:
                continue
            seen.add(url)
            if not is_allowed(url, self.settings.allowed_domains):
                logger.info("Skipping %s: not on the salary source allowlist", domain_of(url))
                continue
            results.append(self._fetch(query, url, title, snippet))
        if self.settings.cache_ttl_s:
            self._cache[query] = (self.clock(), results)
        return results

    # ------------------------------------------------------------------ internals
    def _cached(self, query: str) -> list[WebSearchResult] | None:
        entry = self._cache.get(query)
        if entry is None:
            return None
        stored_at, results = entry
        if self.clock() - stored_at > self.settings.cache_ttl_s:
            self._cache.pop(query, None)
            return None
        return results

    def _client(self):
        if self.http is None:
            import httpx  # imported lazily: only the salary provider needs it

            self.http = httpx.Client(timeout=self.settings.timeout_s,
                                     headers={"User-Agent": self.settings.user_agent},
                                     follow_redirects=True)
        return self.http

    def _wait(self) -> None:
        gap = self.settings.min_request_interval_s - (self.clock() - self._last_request)
        if self._last_request and gap > 0:
            time.sleep(gap)
        self._last_request = self.clock()

    def _request(self, method: str, url: str, **kwargs):
        import httpx

        self._wait()
        self.calls.append(url)
        try:
            return self._client().request(method, url, timeout=self.settings.timeout_s, **kwargs)
        except httpx.TimeoutException as exc:
            raise WebSearchError("search_timeout", f"{domain_of(url) or url} did not answer within "
                                                   f"{self.settings.timeout_s:g}s.", type(exc).__name__) from None
        except httpx.HTTPError as exc:
            raise WebSearchError("search_unavailable", f"{domain_of(url) or url} could not be reached.",
                                 type(exc).__name__) from None

    def _provider_results(self, query: str, limit: int) -> list[tuple[str, str, str]]:
        provider, endpoint = self.settings.provider, SEARCH_ENDPOINTS[self.settings.provider]
        if provider == "tavily":
            response = self._request("POST", endpoint, json={
                "api_key": self.settings.api_key, "query": query, "max_results": limit,
                "include_domains": list(self.settings.allowed_domains), "include_raw_content": False})
            items = self._json(response, endpoint).get("results", [])
            return [(item.get("url", ""), item.get("title", ""), item.get("content", "")) for item in items
                    if item.get("url")]
        response = self._request("GET", endpoint, params={"q": query, "count": limit},
                                 headers={"X-Subscription-Token": self.settings.api_key or "",
                                          "Accept": "application/json"})
        items = self._json(response, endpoint).get("web", {}).get("results", [])
        return [(item.get("url", ""), item.get("title", ""), item.get("description", "")) for item in items
                if item.get("url")]

    def _json(self, response, endpoint: str) -> dict:
        if response.status_code >= 400:
            raise WebSearchError("search_provider_error",
                                 f"The {self.settings.provider} search API answered {response.status_code}.")
        try:
            payload = response.json()
        except ValueError:
            raise WebSearchError("search_provider_error",
                                 f"The {self.settings.provider} search API returned a non-JSON response.") from None
        return payload if isinstance(payload, dict) else {}

    def _fetch(self, query: str, url: str, title: str, snippet: str) -> WebSearchResult:
        """Reads the page; a page that cannot be read still comes back with its search snippet."""
        content, content_type = "", "text/html"
        try:
            response = self._request("GET", url)
            if response.status_code < 400:
                content_type = (response.headers.get("content-type") or "text/html").split(";")[0].strip()
                content = response.text[: self.settings.max_content_chars]
            else:
                logger.info("Salary source %s answered %s", domain_of(url), response.status_code)
        except WebSearchError as exc:
            logger.info("Salary source %s was not read: %s", domain_of(url), exc.code)
        return WebSearchResult(query=query, url=url, title=title, snippet=snippet, content=content,
                               domain=domain_of(url), retrieved_at=self.today(), content_type=content_type)

    def close(self) -> None:
        if self.http is not None:
            self.http.close()
            self.http = None
