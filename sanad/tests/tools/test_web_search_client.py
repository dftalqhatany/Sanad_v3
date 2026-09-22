"""The web search client: allowlist, providers, timeouts, rate limiting, caching, and real HTTP."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from config import SalarySettings
from tools.web_search import HttpWebSearchClient, NullWebSearchClient, WebSearchError, domain_of, is_allowed

SEED = "https://saudisalary.com/data-analyst-salary"
PAGE = "<html><body><p>Data Analyst salary 12,000 SAR per month.</p></body></html>"


def settings(**overrides) -> SalarySettings:
    base = dict(enabled=True, seed_urls=(SEED,), min_request_interval_s=0.0, cache_ttl_s=60.0)
    return SalarySettings(**{**base, **overrides})


def transport(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def page_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text=PAGE, headers={"content-type": "text/html; charset=utf-8"})


# --------------------------------------------------------------------------- basics
def test_the_null_client_makes_no_request():
    assert NullWebSearchClient().search("anything") == []


def test_domains_are_matched_by_suffix():
    assert is_allowed("https://saudiarabia.paylab.com/en/salaryinfo", ("paylab.com",))
    assert is_allowed("https://WWW.Stats.gov.sa/en/x", ("stats.gov.sa",))
    assert not is_allowed("https://paylab.com.evil.example/x", ("paylab.com",))
    assert domain_of("https://www.kaggle.com/datasets/x") == "kaggle.com"


def test_seed_sources_are_read_and_carry_their_provenance():
    client = HttpWebSearchClient(settings(), http=transport(page_handler), today=lambda: "2026-09-18")
    [result] = client.search("Data Analyst salary Riyadh Saudi Arabia 2026")
    assert result.url == SEED and result.domain == "saudisalary.com"
    assert result.retrieved_at == "2026-09-18" and "12,000 SAR" in result.content
    assert result.query == "Data Analyst salary Riyadh Saudi Arabia 2026"
    assert client.calls == [SEED]


def test_results_outside_the_allowlist_are_dropped():
    client = HttpWebSearchClient(settings(seed_urls=("https://example.com/salaries", SEED)),
                                 http=transport(page_handler))
    urls = [result.url for result in client.search("q")]
    assert urls == [SEED] and "https://example.com/salaries" not in client.calls


def test_a_page_that_fails_still_comes_back_with_its_snippet():
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="busy")

    [result] = HttpWebSearchClient(settings(), http=transport(failing)).search("q")
    assert result.url == SEED and result.content == ""


def test_a_timeout_while_searching_is_explicit():
    def timeout(request: httpx.Request):
        raise httpx.ConnectTimeout("too slow", request=request)

    client = HttpWebSearchClient(settings(provider="tavily", api_key="k"), http=transport(timeout))
    with pytest.raises(WebSearchError) as raised:
        client.search("q")
    assert raised.value.code == "search_timeout" and "did not answer" in raised.value.message


def test_a_page_timeout_does_not_fail_the_whole_search():
    def only_seed_times_out(request: httpx.Request):
        if request.url.host == "saudisalary.com":
            raise httpx.ReadTimeout("slow page", request=request)
        return httpx.Response(200, json={"results": [{"url": "https://www.kaggle.com/x", "title": "t", "content": "s"}]})

    client = HttpWebSearchClient(settings(provider="tavily", api_key="k"), http=transport(only_seed_times_out))
    results = {result.url: result for result in client.search("q")}
    assert results["https://www.kaggle.com/x"].snippet == "s"
    assert results[SEED].content == ""  # reported, not fatal


# --------------------------------------------------------------------------- search providers
def test_tavily_results_are_used_and_the_key_is_sent_once():
    seen: list[dict] = []

    def tavily(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.tavily.com":
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"results": [{"url": "https://saudiarabia.paylab.com/en/salaryinfo",
                                                          "title": "Paylab", "content": "range"}]})
        return page_handler(request)

    client = HttpWebSearchClient(settings(provider="tavily", api_key="secret-key"), http=transport(tavily))
    urls = [result.url for result in client.search("Data Analyst salary")]
    assert "https://saudiarabia.paylab.com/en/salaryinfo" in urls and SEED in urls
    assert seen[0]["query"] == "Data Analyst salary" and seen[0]["api_key"] == "secret-key"
    assert seen[0]["include_domains"] == list(client.settings.allowed_domains)


def test_brave_results_are_used_with_its_header():
    headers: list[str] = []

    def brave(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.search.brave.com":
            headers.append(request.headers.get("x-subscription-token", ""))
            return httpx.Response(200, json={"web": {"results": [{"url": "https://www.stats.gov.sa/x", "title": "G",
                                                                  "description": "d"}]}})
        return page_handler(request)

    client = HttpWebSearchClient(settings(provider="brave", api_key="brave-key"), http=transport(brave))
    assert "https://www.stats.gov.sa/x" in [result.url for result in client.search("q")]
    assert headers == ["brave-key"]


def test_a_provider_error_is_explicit():
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    client = HttpWebSearchClient(settings(provider="tavily", api_key="wrong"), http=transport(failing))
    with pytest.raises(WebSearchError) as raised:
        client.search("q")
    assert raised.value.code == "search_provider_error" and "401" in raised.value.message


# --------------------------------------------------------------------------- caching, limits, rate
def test_repeated_queries_are_served_from_the_cache():
    client = HttpWebSearchClient(settings(), http=transport(page_handler))
    client.search("same question")
    client.search("same question")
    assert client.calls == [SEED]
    client.search("another question")
    assert client.calls == [SEED, SEED]


def test_the_cache_expires():
    now = [0.0]
    client = HttpWebSearchClient(settings(cache_ttl_s=10), http=transport(page_handler), clock=lambda: now[0])
    client.search("q")
    now[0] = 11
    client.search("q")
    assert len(client.calls) == 2


def test_the_number_of_results_is_capped():
    seeds = tuple(f"https://saudisalary.com/{i}" for i in range(10))
    client = HttpWebSearchClient(settings(seed_urls=seeds, max_results=3), http=transport(page_handler))
    assert len(client.search("q")) == 3


def test_requests_are_spaced_out():
    now = [100.0]
    slept: list[float] = []
    seeds = ("https://saudisalary.com/a", "https://saudisalary.com/b")
    client = HttpWebSearchClient(settings(seed_urls=seeds, min_request_interval_s=0.5), http=transport(page_handler),
                                 clock=lambda: now[0])
    import tools.web_search as module

    original, module.time.sleep = module.time.sleep, lambda seconds: slept.append(seconds)
    try:
        client.search("q")
    finally:
        module.time.sleep = original
    assert slept == [0.5]  # the second request waited for the configured gap


def test_no_api_key_is_ever_logged(caplog):
    import logging

    client = HttpWebSearchClient(settings(provider="tavily", api_key="super-secret"), http=transport(page_handler))
    with caplog.at_level(logging.DEBUG, logger="sanad"):
        try:
            client.search("q")
        except WebSearchError:
            pass
    assert "super-secret" not in caplog.text and "super-secret" not in repr(client.settings)


# --------------------------------------------------------------------------- real HTTP against a local server
class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep the test output clean
        return


def test_the_client_really_speaks_http():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/data-analyst-salary"
        client = HttpWebSearchClient(settings(seed_urls=(url,), allowed_domains=("127.0.0.1",)))
        [result] = client.search("Data Analyst salary Riyadh Saudi Arabia 2026")
        assert "12,000 SAR" in result.content and result.content_type == "text/html"
        assert result.retrieved_at and result.domain == "127.0.0.1"
        client.close()
    finally:
        server.shutdown()
        server.server_close()
