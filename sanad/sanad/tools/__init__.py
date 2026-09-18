"""Outward-facing tools Sanad agents use: currently web search and salary extraction.

Nothing here knows about the regulatory RAG, and nothing here decides anything: the tools retrieve
and parse, the agents interpret.
"""

from sanad.tools.web_search import (
    HttpWebSearchClient,
    NullWebSearchClient,
    WebSearchClient,
    WebSearchError,
    WebSearchResult,
)

__all__ = ["HttpWebSearchClient", "NullWebSearchClient", "WebSearchClient", "WebSearchError", "WebSearchResult"]
