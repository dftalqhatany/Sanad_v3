"""Deterministic stand-in for the web: no network, scripted pages, recorded queries."""

from __future__ import annotations

from dataclasses import dataclass, field

from tools.web_search import WebSearchError, WebSearchResult, domain_of


@dataclass
class FakePage:
    url: str
    title: str = ""
    snippet: str = ""
    content: str = ""
    content_type: str = "text/html"


@dataclass
class FakeWebSearchClient:
    """Returns the configured pages for every query (or raises the configured search failure)."""

    pages: list[FakePage] = field(default_factory=list)
    error: WebSearchError | None = None
    raises: BaseException | None = None
    retrieved_at: str = "2026-09-18"
    name: str = "fake"
    queries: list[str] = field(default_factory=list)

    def search(self, query: str, *, max_results: int | None = None) -> list[WebSearchResult]:
        self.queries.append(query)
        if self.raises is not None:
            raise self.raises
        if self.error is not None:
            raise self.error
        pages = self.pages[:max_results] if max_results else self.pages
        return [
            WebSearchResult(query=query, url=page.url, title=page.title, snippet=page.snippet, content=page.content,
                            domain=domain_of(page.url), retrieved_at=self.retrieved_at, content_type=page.content_type)
            for page in pages
        ]
