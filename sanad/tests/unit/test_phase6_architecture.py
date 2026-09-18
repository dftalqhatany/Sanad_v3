"""Phase 6 boundaries: the Orchestrator only coordinates the existing components."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR = PROJECT_ROOT / "sanad" / "orchestrator"
ORCHESTRATOR_FILES = sorted(ORCHESTRATOR.rglob("*.py")) + [PROJECT_ROOT / "sanad" / "models" / "orchestration.py"]
AGENT_FILES = sorted((PROJECT_ROOT / "sanad" / "agents").rglob("*.py"))

# The Orchestrator must never reach past the abstractions built in Phases 2-5.
FORBIDDEN_MODULES = ("chatbot_backend", "hybird_search", "llama_index", "qdrant_client", "openai", "rank_bm25",
                     "sanad.agents.contract_analysis", "sanad.agents.cv_analysis", "sanad.agents.regulatory",
                     "sanad.agents.interpretation", "sanad.agents.salary", "sanad.agents.comparison_dimensions",
                     "sanad.agents.recommendation")
FORBIDDEN_NAMES = ("HybridRetriever", "QdrantClient", "answer_policy_question", "get_retriever", "complete_json",
                   "OpenAIChatClient", "LLMEvidenceInterpreter", "RegulatoryEvidenceCollector", "build_dimensions",
                   "recommend")
# Only these methods of the existing components may be called.
ALLOWED_METHODS = {"analysis_agent": {"analyze", "benchmark_salary"}, "comparison_agent": {"compare"},
                   "regulatory_source": {"ask", "retrieve_evidence"}, "intake": {"load"}}


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_level = {id(node) for node in tree.body}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, id(node) in top_level
        elif isinstance(node, ast.ImportFrom):
            yield node.module or "", id(node) in top_level


def test_orchestrator_does_not_import_the_legacy_rag_an_llm_or_agent_internals():
    offenders = [f"{path.name}: {module}" for path in ORCHESTRATOR_FILES for module, _ in _imports(path)
                 if any(module == f or module.startswith(f + ".") for f in FORBIDDEN_MODULES)]
    assert len(ORCHESTRATOR_FILES) >= 5
    assert not offenders, offenders


def test_the_only_rag_import_is_the_existing_adapter_and_it_is_lazy():
    rag_imports = [(path.name, module, top) for path in ORCHESTRATOR_FILES for module, top in _imports(path)
                   if module == "sanad.rag" or module.startswith("sanad.rag.")]
    assert rag_imports == [("orchestrator.py", "sanad.rag.adapter", False)]


def test_orchestrator_names_no_retriever_llm_or_agent_internal():
    offenders = []
    for path in ORCHESTRATOR_FILES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
            if name in FORBIDDEN_NAMES:
                offenders.append(f"{path.name}:{node.lineno} {name}")
    assert not offenders, offenders


def test_only_public_component_methods_are_called():
    called: dict[str, set[str]] = {}
    for path in ORCHESTRATOR_FILES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner = node.func.value
                if isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name) and owner.value.id == "self":
                    called.setdefault(owner.attr, set()).add(node.func.attr)
    for component, allowed in ALLOWED_METHODS.items():
        assert called.get(component, set()) <= allowed, (component, called.get(component))
    assert called.get("analysis_agent") and called.get("comparison_agent")  # both are actually used


def test_parsing_happens_only_in_the_intake_step():
    users = [path.name for path in ORCHESTRATOR_FILES for module, _ in _imports(path)
             if module.startswith("sanad.parsers") or module.startswith("sanad.extraction")]
    assert set(users) == {"intake.py"}
    agent_users = [path.name for path in AGENT_FILES for module, _ in _imports(path)
                   if module.startswith("sanad.parsers")]
    assert not agent_users, agent_users  # agents never parse documents themselves


def test_agents_do_not_depend_on_the_orchestrator():
    offenders = [path.name for path in AGENT_FILES for module, _ in _imports(path)
                 if module.startswith("sanad.orchestrator")]
    assert not offenders, offenders


def test_routing_is_deterministic_and_model_free():
    text = (ORCHESTRATOR / "routing.py").read_text(encoding="utf-8")
    assert "llm" not in text.lower() and "prompt" not in text.lower()
    tree = ast.parse(text)
    assert [node.name for node in tree.body if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")] \
        == ["route_request"]


def test_the_orchestrator_knows_nothing_about_the_api_or_the_user_interface():
    text = "\n".join(path.read_text(encoding="utf-8") for path in ORCHESTRATOR_FILES)
    for web in ("fastapi", "streamlit", "uvicorn", "flask", "starlette"):
        assert web not in text, web
    offenders = [path.name for path in ORCHESTRATOR_FILES for module, _ in _imports(path)
                 if module.startswith(("sanad.api", "frontend"))]
    assert not offenders, offenders  # the API depends on the orchestrator, never the other way round


def test_running_the_orchestrator_loads_no_legacy_rag_or_openai():
    code = (
        "import sys, json\n"
        "from sanad.orchestrator import SanadOrchestrator\n"
        "assert not [m for m in sys.modules if m.startswith('sanad.rag')], 'importing the orchestrator loaded sanad.rag'\n"
        "from sanad.agents import AnalysisAgent, ContractAnalysisAgent, ContractComparisonAgent\n"
        "from sanad.models.orchestration import SanadRequest, UploadedDocument\n"
        "from tests.fakes.regulatory_adapter import FakeRegulatoryAdapter\n"
        "from tests.fixtures.documents.builders import build_all\n"
        "kb = json.load(open('../hr_assistant/data/labor_law/labor_law_parsed.json', encoding='utf-8'))\n"
        "rag = FakeRegulatoryAdapter(kb)\n"
        "analysis = AnalysisAgent(ContractAnalysisAgent(rag))\n"
        "orchestrator = SanadOrchestrator(analysis, ContractComparisonAgent(analysis), rag)\n"
        "docs = build_all()\n"
        "uploads = [UploadedDocument(filename=n, content=docs[n]) for n in ('sample_contract_en.docx', 'sample_cv_en.docx')]\n"
        "result = orchestrator.handle(SanadRequest(documents=uploads))\n"
        "assert result.routing.route.value == 'contract_analysis', result.routing.route\n"
        "assert result.analysis.contract_analysis.evidence\n"
        "roots = ('chatbot_backend', 'hybird_search', 'llama_index', 'qdrant_client', 'openai', 'rank_bm25')\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in roots]\n"
        "assert not loaded, loaded\n"
        "print('clean')\n"
    )
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip() == "clean"
