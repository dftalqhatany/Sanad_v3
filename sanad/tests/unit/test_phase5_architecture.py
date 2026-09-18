"""Phase 5 boundaries: the comparison agent only consumes Analysis Agent results."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENTS = PROJECT_ROOT / "sanad" / "agents"
COMPARISON_FILES = [AGENTS / "comparison.py", AGENTS / "comparison_dimensions.py", AGENTS / "recommendation.py",
                    PROJECT_ROOT / "sanad" / "models" / "comparison.py"]
# Everything below is the Analysis Agent's (or an earlier layer's) job; the comparison must not redo it.
FORBIDDEN_MODULES = ("sanad.rag", "sanad.parsers", "sanad.extraction", "sanad.agents.contract_analysis",
                     "sanad.agents.cv_analysis", "sanad.agents.regulatory", "openai", "qdrant_client", "llama_index",
                     "hybird_search", "chatbot_backend")
FORBIDDEN_NAMES = ("retrieve_evidence", "RegulatoryEvidenceCollector", "ContractAnalysisAgent", "CvAnalysisAgent",
                   "extract_contract", "extract_cv", "DocumentProcessor", "complete_json", "LLMEvidenceInterpreter")


def _modules(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""


def test_comparison_modules_do_not_reach_past_the_analysis_agent():
    offenders = []
    for path in COMPARISON_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders += [f"{path.name}: import {m}" for m in _modules(tree)
                      if any(m == f or m.startswith(f + ".") for f in FORBIDDEN_MODULES)]
        for node in ast.walk(tree):
            name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
            if name in FORBIDDEN_NAMES:
                offenders.append(f"{path.name}:{node.lineno} {name}")
    assert not offenders, offenders


def test_the_comparison_agent_calls_the_analysis_agent():
    tree = ast.parse((AGENTS / "comparison.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "analyze"]
    assert len(calls) == 1
    assert ast.unparse(calls[0].func) == "self.analysis_agent.analyze"


def test_no_llm_is_used_to_choose_a_contract():
    for path in COMPARISON_FILES:
        text = path.read_text(encoding="utf-8")
        assert "LLMClient(" not in text and "interpret(" not in text, path.name
