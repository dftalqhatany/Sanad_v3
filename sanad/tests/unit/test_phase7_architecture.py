"""Phase 7 boundaries: a thin API over the Orchestrator, and a frontend that only speaks HTTP."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
API_FILES = sorted((PROJECT_ROOT / "sanad" / "api").rglob("*.py"))
FRONTEND_FILES = sorted((PROJECT_ROOT / "frontend").rglob("*.py"))

# The API may use the Orchestrator, the shared models and the settings - nothing deeper.
API_FORBIDDEN = ("sanad.agents", "sanad.rag", "sanad.parsers", "sanad.extraction", "chatbot_backend", "hybird_search",
                 "llama_index", "qdrant_client", "openai", "rank_bm25")
API_FORBIDDEN_NAMES = ("AnalysisAgent", "ContractComparisonAgent", "ContractAnalysisAgent", "CvAnalysisAgent",
                       "RegulatoryRAGAdapter", "DocumentProcessor", "extract_contract", "extract_cv",
                       "retrieve_evidence", "compare", "analyze", "benchmark_salary", "complete_json")


def _modules(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""


def _names(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Name):
            yield node.id, node.lineno
        elif isinstance(node, ast.Attribute):
            yield node.attr, node.lineno


def test_the_api_never_imports_an_agent_the_rag_a_parser_or_an_llm():
    offenders = [f"{path.name}: {module}" for path in API_FILES for module in _modules(path)
                 if any(module == f or module.startswith(f + ".") for f in API_FORBIDDEN)]
    assert len(API_FILES) >= 4
    assert not offenders, offenders


def test_the_api_names_no_agent_parser_or_retrieval_call():
    offenders = [f"{path.name}:{line} {name}" for path in API_FILES for name, line in _names(path)
                 if name in API_FORBIDDEN_NAMES]
    assert not offenders, offenders


def test_the_only_sanad_call_the_api_makes_is_orchestrator_handle():
    calls = []
    for path in API_FILES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                target = ast.unparse(node.func)
                if "orchestrator" in target:
                    calls.append(target)
    assert calls == ["orchestrator.handle"]


def test_the_api_imports_the_orchestrator_and_the_shared_models_only():
    modules = {module for path in API_FILES for module in _modules(path) if module.startswith("sanad.")}
    assert modules <= {"sanad.orchestrator", "sanad.config", "sanad.models.analysis", "sanad.models.orchestration",
                       "sanad.api.app", "sanad.api.schemas"}, modules


def test_the_frontend_talks_to_the_api_and_imports_no_backend_code():
    offenders = [f"{path.name}: {module}" for path in FRONTEND_FILES for module in _modules(path)
                 if module == "sanad" or module.startswith("sanad.")]
    assert len(FRONTEND_FILES) >= 3
    assert not offenders, offenders
    http_users = {path.name for path in FRONTEND_FILES for module in _modules(path) if module in ("httpx", "requests")}
    assert http_users == {"client.py"}  # the UI reaches the backend through one client


def test_the_frontend_has_no_routing_or_analysis_logic():
    text = "\n".join(path.read_text(encoding="utf-8") for path in FRONTEND_FILES)
    for term in ("RegulatoryQuery", "FindingStatus", "route_request", "TargetJob(", "ContractExtraction"):
        assert term not in text, term
    app = (PROJECT_ROOT / "frontend" / "app.py").read_text(encoding="utf-8")
    assert "client.analyze(" in app and "client.ask(" in app  # every request goes through the API client


def test_the_api_is_runnable_as_a_module():
    main = (PROJECT_ROOT / "sanad" / "api" / "__main__.py").read_text(encoding="utf-8")
    assert "uvicorn.run" in main and "create_app()" in main


def test_no_uploaded_file_is_written_to_disk():
    text = "\n".join(path.read_text(encoding="utf-8") for path in API_FILES)
    for term in ("open(", "NamedTemporaryFile", "mkdtemp", "shutil.copy", "write_bytes", "os.remove"):
        assert term not in text, term
