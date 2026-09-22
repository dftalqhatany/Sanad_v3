"""Salary benchmarking boundaries: tools retrieve, the provider judges, nobody hard-codes a key."""

from __future__ import annotations

import ast
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS = sorted((PROJECT_ROOT / "tools").rglob("*.py"))
PROVIDER = PROJECT_ROOT / "agents" / "salary_web.py"
SALARY_FILES = [*TOOLS, PROVIDER, PROJECT_ROOT / "agents" / "salary.py"]
PACKAGE_DIRS = ("agents", "api", "extraction", "models", "orchestrator", "parsers", "rag", "tools")
PACKAGE_FILES = (sorted(f for d in PACKAGE_DIRS for f in (PROJECT_ROOT / d).rglob("*.py"))
                 + [PROJECT_ROOT / "config.py"]
                 + sorted((PROJECT_ROOT / "frontend").rglob("*.py")))


def _modules(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""


def test_the_tools_know_nothing_about_agents_the_rag_or_the_api():
    forbidden = ("agents", "rag", "orchestrator", "api",
                 "llama_index", "qdrant_client", "openai")
    offenders = [f"{path.name}: {module}" for path in TOOLS for module in _modules(path)
                 if any(module == f or module.startswith(f + ".") for f in forbidden)]
    assert len(TOOLS) >= 4
    assert not offenders, offenders


def test_the_provider_does_no_http_itself():
    modules = set(_modules(PROVIDER))
    assert not {"httpx", "requests", "urllib", "urllib.request", "http.client"} & modules
    assert "tools.web_search" in modules  # it goes through the client abstraction


def test_http_lives_only_in_the_search_client():
    users = {path.name for path in PACKAGE_FILES for module in _modules(path) if module in ("httpx", "requests")}
    assert users <= {"web_search.py", "client.py"}  # the search client and the UI's API client


def test_no_api_key_is_hard_coded_anywhere():
    assignment = re.compile(r"(\w*(?:api_key|token|secret)\w*)\s*[:=]\s*[\"']([A-Za-z0-9_\-]{12,})[\"']",
                            re.IGNORECASE)
    offenders = []
    for path in PACKAGE_FILES:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = assignment.search(line)
            # an enum member such as MISSING_API_KEY = "missing_api_key" names itself; a credential does not
            if match and match.group(2).lower() != match.group(1).lower() and "env" not in line.lower():
                offenders.append(f"{path.name}:{number} {match.group(1)}")
    assert not offenders, offenders


def test_keys_and_limits_come_from_settings():
    config = (PROJECT_ROOT / "config.py").read_text(encoding="utf-8")
    for variable in ("SANAD_SEARCH_PROVIDER", "SANAD_SEARCH_API_KEY", "SANAD_SALARY_DOMAINS", "SANAD_SALARY_TIMEOUT_S",
                     "SANAD_SALARY_MAX_RESULTS", "SANAD_SALARY_CACHE_TTL_S", "SANAD_SALARY_MIN_INTERVAL_S",
                     "SANAD_SALARY_FX_RATES"):
        assert variable in config, variable
    assert "TAVILY_API_KEY" in config or "_API_KEY" in config
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "SANAD_SEARCH_API_KEY" in readme and "SANAD_SALARY_ENABLED" in readme


def test_the_documented_sources_and_their_ranking_are_declared():
    from config import SALARY_ALLOWED_DOMAINS, SALARY_SEED_SOURCES
    from tools.salary_sources import profile_for

    assert {"stats.gov.sa", "open.data.gov.sa", "saudisalary.com", "paylab.com", "kaggle.com"} <= set(SALARY_ALLOWED_DOMAINS)
    assert any("stats.gov.sa" in url for url in SALARY_SEED_SOURCES)
    assert profile_for("https://www.stats.gov.sa/en/w/x").tier == "official"
    assert profile_for("https://open.data.gov.sa/en/datasets/x").tier == "official"
    assert profile_for("https://saudisalary.com/x").tier == "market"
    assert profile_for("https://saudiarabia.paylab.com/en/x").tier == "market"
    assert profile_for("https://www.kaggle.com/datasets/x").tier == "supplementary"
    assert profile_for("https://lnkd.in/p/x").tier == "lead_only"
    assert profile_for("https://www.linkedin.com/posts/x").tier == "lead_only"


def test_the_market_range_is_built_only_by_the_provider():
    setters = [path.name for path in PACKAGE_FILES
               if re.search(r"market_(min|max)\s*=", path.read_text(encoding="utf-8"))]
    assert set(setters) <= {"salary_web.py", "analysis.py"}  # the provider, and the model that declares the field


def test_the_comparison_reads_the_benchmark_rather_than_recomputing_it():
    dimensions = (PROJECT_ROOT / "agents" / "comparison_dimensions.py").read_text(encoding="utf-8")
    assert "salary_benchmark" in dimensions and "WebSearchSalaryProvider" not in dimensions
    assert "extract_observations" not in dimensions
