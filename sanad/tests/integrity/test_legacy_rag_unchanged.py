"""Proof that Sanad did not modify the existing RAG, its knowledge base or its vector database.

Compared against tests/fixtures/legacy_rag_baseline.json, recorded at the start of Phase 2 before
any Sanad code existed, and against git HEAD. The only permitted differences are the changes the
project owner approved, listed in tests/fixtures/approved_legacy_changes.json.

Qdrant WAL/segment files are excluded from hashing on purpose: a running Qdrant server rewrites
them during normal operation (12 segment.json files already differed from git before Phase 2).
The live test tests/live checks the collection's contents through the Qdrant API instead.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.conftest import BASELINE_PATH, LEGACY_DIR, PROJECT_ROOT, REPO_ROOT

APPROVED_CHANGES = json.loads(
    (BASELINE_PATH.parent / "approved_legacy_changes.json").read_text(encoding="utf-8")
)["changes"]

FORBIDDEN_VECTOR_DB_CALLS = {
    "create_collection", "recreate_collection", "delete_collection", "update_collection",
    "upsert", "upload_points", "upload_collection", "delete", "delete_vectors", "set_payload",
    "overwrite_payload", "clear_payload", "from_documents", "insert_nodes",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=REPO_ROOT, env=env, capture_output=True, text=True)


def test_existing_rag_code_notebooks_and_documents_are_byte_identical(baseline):
    protected = baseline["protected_files_sha256"]
    assert len(protected) == 26
    for must_include in (
        "hr_assistant/app.py", "hr_assistant/chatbot_backend.py", "hr_assistant/hybird_search.py",
        "hr_assistant/process_labor_pdf.ipynb", "hr_assistant/process_data_vectors.ipynb",
        "hr_assistant/data/labor_law/labor_law_ar.pdf", "hr_assistant/data/labor_law/labor_law_parsed.json",
    ):
        assert must_include in protected
    assert set(APPROVED_CHANGES) == {"hr_assistant/hybird_search.py"}
    changed = {rel for rel, digest in protected.items() if rel not in APPROVED_CHANGES and sha256(REPO_ROOT / rel) != digest}
    assert not changed, f"existing RAG files were modified: {sorted(changed)}"


def test_approved_fix_is_the_only_change_to_hybird_search(baseline):
    for rel, change in APPROVED_CHANGES.items():
        data = (REPO_ROOT / rel).read_bytes()
        fixed_line, original_line = (change["fixed_line"] + "\n").encode(), (change["original_line"] + "\n").encode()
        assert data.count(fixed_line) == 1, f"approved fix not present in {rel}"
        assert data.decode("utf-8").splitlines()[change["line_number"] - 1] == change["fixed_line"]
        assert hashlib.sha256(data).hexdigest() == change["sha256_after"], f"{rel} has changes beyond the approved line"
        reverted = data.replace(fixed_line, original_line)
        assert hashlib.sha256(reverted).hexdigest() == change["sha256_before"] == baseline["protected_files_sha256"][rel]


def test_no_files_were_added_to_the_existing_rag_project(baseline):
    current = {
        p.relative_to(REPO_ROOT).as_posix()
        for p in LEGACY_DIR.rglob("*")
        if p.is_file() and "qdrant_storage" not in p.parts and "__pycache__" not in p.parts and p.name != ".DS_Store"
    }
    assert current == set(baseline["protected_files_sha256"])


def test_qdrant_collections_are_still_present_with_unchanged_configuration(baseline):
    collections_dir = LEGACY_DIR / "qdrant_storage" / "collections"
    assert sorted(p.name for p in collections_dir.iterdir() if p.is_dir()) == baseline["qdrant_collections_on_disk"]
    for rel, digest in baseline["qdrant_collection_configs_sha256"].items():
        assert sha256(REPO_ROOT / rel) == digest, f"{rel} changed"
    config = json.loads((collections_dir / "saudi_labor_law" / "config.json").read_text(encoding="utf-8"))
    assert config["params"]["vectors"] == {"size": 768, "distance": "Cosine"}


def test_knowledge_base_still_has_all_articles(baseline, knowledge_base):
    assert len(knowledge_base) == baseline["knowledge_base"]["article_count"] == 249
    assert [a["index"] for a in knowledge_base] == list(range(1, 250))


def test_protected_files_match_git_head(baseline):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    unchanged = [rel for rel in baseline["protected_files_sha256"] if rel not in APPROVED_CHANGES]
    completed = git("diff", "--quiet", "HEAD", "--", *unchanged)
    if completed.returncode not in (0, 1):
        pytest.skip(f"git diff unavailable here: {completed.stderr.strip()}")
    assert completed.returncode == 0, "existing RAG files differ from git HEAD"
    for rel in APPROVED_CHANGES:
        numstat = git("diff", "--numstat", "HEAD", "--", rel).stdout.split()
        # uncommitted: exactly one line removed and one added; committed: no diff (hash test still guards content)
        assert numstat in ([], ["1", "1", rel]), f"unexpected git diff for {rel}: {numstat}"


def test_existing_rag_interface_used_by_adapter_and_streamlit_is_intact():
    backend = ast.parse((LEGACY_DIR / "chatbot_backend.py").read_text(encoding="utf-8"))
    functions = {n.name: n for n in backend.body if isinstance(n, ast.FunctionDef)}
    assert [a.arg for a in functions["answer_policy_question"].args.args] == ["query", "employee_data", "api_key"]
    assert {"get_retriever", "detect_language", "generate_answer", "highlight_articles"} <= set(functions)
    assert any(isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "documents" for n in backend.body)

    search = ast.parse((LEGACY_DIR / "hybird_search.py").read_text(encoding="utf-8"))
    [retriever_cls] = [n for n in search.body if isinstance(n, ast.ClassDef) and n.name == "HybridRetriever"]
    assert "retrieve" in {n.name for n in retriever_cls.body if isinstance(n, ast.FunctionDef)}

    app = ast.parse((LEGACY_DIR / "app.py").read_text(encoding="utf-8"))
    assert any(
        isinstance(n, ast.ImportFrom) and n.module == "chatbot_backend" and any(a.name == "answer_policy_question" for a in n.names)
        for n in app.body
    )


def test_sanad_code_contains_no_vector_db_write_or_ingestion_calls():
    offenders = []
    for path in (PROJECT_ROOT / "sanad").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_VECTOR_DB_CALLS:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} .{node.func.attr}()")
    assert not offenders, offenders
