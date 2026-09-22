"""Phase 4 boundaries: agents reach regulatory knowledge only through sanad/rag/adapter.py."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_FILES = sorted((PROJECT_ROOT / "agents").rglob("*.py"))
PHASE4_FILES = [*AGENT_FILES, PROJECT_ROOT / "models" / "analysis.py"]

# Retrieval / vector / embedding / BM25 / RAG internals must never be used directly by an agent.
FORBIDDEN_EVERYWHERE = ("qdrant_client", "llama_index", "rag.retriever", "rag.backend", "rank_bm25",
                        "sentence_transformers", "transformers", "torch", "sklearn", "langchain", "chromadb", "faiss",
                        "rag.loader", "rag.retrieval_config", "parsers")
FORBIDDEN_NAMES = ("HybridRetriever", "QdrantClient", "QdrantVectorStore", "VectorStoreIndex", "BM25Okapi",
                   "HuggingFaceEmbedding", "VectorIndexRetriever")


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_level = set(map(id, tree.body))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node, alias.name, id(node) in top_level
        elif isinstance(node, ast.ImportFrom):
            yield node, node.module or "", id(node) in top_level


def test_agents_do_not_import_retrieval_vector_embedding_or_rag_internals():
    offenders = [f"{path.name}:{node.lineno} {name}" for path in PHASE4_FILES for node, name, _ in _imports(path)
                 if any(name == f or name.startswith(f + ".") for f in FORBIDDEN_EVERYWHERE)]
    assert len(AGENT_FILES) >= 8
    assert not offenders, offenders


def test_the_only_rag_import_is_the_existing_adapter_and_it_is_lazy():
    rag_imports = [(path.name, name, top) for path in PHASE4_FILES for _, name, top in _imports(path)
                   if name == "rag" or name.startswith("rag.")]
    assert rag_imports == [("contract_analysis.py", "rag.adapter", False)]


def test_openai_is_only_imported_lazily_inside_the_client():
    openai_imports = [(path.name, top) for path in PHASE4_FILES for _, name, top in _imports(path)
                      if name == "openai" or name.startswith("openai.")]
    assert openai_imports == [("interpretation.py", False)]


def test_agents_do_not_reference_retriever_or_vector_store_classes():
    offenders = []
    for path in PHASE4_FILES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
            if name in FORBIDDEN_NAMES:
                offenders.append(f"{path.name}:{node.lineno} {name}")
    assert not offenders, offenders


def test_agents_stay_agents():
    # Later phases live in their own packages: sanad/orchestrator (Phase 6) and sanad/api (Phase 7).
    agents = PROJECT_ROOT / "agents"
    names = {path.stem for path in agents.glob("*.py")}
    assert not names & {"orchestrator", "supervisor", "router", "api", "app", "server", "frontend"}
    agent_text = "\n".join(path.read_text(encoding="utf-8") for path in AGENT_FILES)
    for web in ("fastapi", "streamlit", "uvicorn", "flask"):
        assert web not in agent_text, web  # no web framework inside an agent


def test_running_the_agents_loads_no_rag_infrastructure_or_openai():
    # The fake adapter uses the real rag.mapping; rag.backend and rag.retriever must stay unloaded.
    code = (
        "import sys, json\n"
        "import agents\n"
        "assert not [m for m in sys.modules if m.startswith('rag')], 'importing agents loaded a rag module'\n"
        "from agents import AnalysisAgent, ContractAnalysisAgent\n"
        "from config import DocumentProcessingSettings\n"
        "from parsers import DocumentProcessor\n"
        "from extraction import extract_contract, extract_cv\n"
        "from models.analysis import TargetJob\n"
        "from tests.fixtures.documents.builders import build_all\n"
        "from tests.fakes.regulatory_adapter import FakeRegulatoryAdapter\n"
        "from tests.fakes.llm import FakeLLMClient\n"
        "from agents import LLMEvidenceInterpreter\n"
        "kb = json.load(open('rag/data/labor_law/labor_law_parsed.json', encoding='utf-8'))\n"
        "processor = DocumentProcessor(DocumentProcessingSettings())\n"
        "docs = build_all()\n"
        "contract = extract_contract(processor.parse_bytes(docs['sample_contract_ar.docx'], 'c.docx'))\n"
        "cv = extract_cv(processor.parse_bytes(docs['sample_cv_en.docx'], 'cv.docx'))\n"
        "agent = AnalysisAgent(ContractAnalysisAgent(FakeRegulatoryAdapter(kb), LLMEvidenceInterpreter(FakeLLMClient())))\n"
        "bundle = agent.analyze(contract=contract, cv=cv, target_job=TargetJob(title='Analyst'))\n"
        "assert bundle.contract_analysis.evidence\n"
        "roots = ('llama_index', 'qdrant_client', 'openai', 'rank_bm25',\n"
        "         'sentence_transformers', 'torch', 'transformers')\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in roots]\n"
        "loaded += [m for m in ('rag.backend', 'rag.retriever') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
        "print('clean')\n"
    )
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip() == "clean"
