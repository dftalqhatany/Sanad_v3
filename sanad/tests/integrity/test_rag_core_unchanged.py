"""Proof that the migration did not change Sanad's RAG core, its knowledge base or its vector database.

Before Phase 2 these tests proved that Sanad never modified an external hr_assistant/ project. That
project is now part of Sanad (rag/ and eval/), so the guarantee is restated rather than dropped:

    the migrated RAG core and regulatory source data must still be byte-identical to what they were
    before any Sanad code existed, except for differences the project owner approved explicitly.

Nothing is re-baselined from the filesystem. Every hash in tests/fixtures/rag_core_baseline.json is
COPIED from legacy_rag_baseline.json (recorded at the start of Phase 2, before Sanad existed) or from
approved_legacy_changes.json, and both of those files are kept unchanged as the historical record.

The approved differences live in three places, and a difference with no record still fails:
  * approved_legacy_changes.json          the one-line retrieval fix in rag/retriever.py
  * rag_core_baseline.json -> changed_during_migration   the two structural edits in rag/backend.py
  * rag_core_baseline.json -> changed_in_phase3          the notebooks' repointed relative paths
  * approved_legacy_removals.json         files removed with the owner's approval

Every recorded change is checked by REVERTING it: undoing the listed replacements must reproduce the
original file byte for byte, so a recorded change cannot hide an unrecorded one.

Retired with the migration, and why:
  * the git-HEAD comparison: every protected file moved, so a diff against HEAD describes the
    migration itself rather than any drift. The content hashes above are the stronger guarantee.
  * the hr_assistant/app.py interface check: the legacy Streamlit UI was superseded by frontend/ +
    api/ + the orchestrator's regulatory route. The backend interface it used is still checked below.

Qdrant WAL/segment files are excluded from hashing on purpose: a running Qdrant rewrites them.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from tests.conftest import BASELINE_PATH, PROJECT_ROOT, RAG_DIR

FIXTURES = BASELINE_PATH.parent
CORE = json.loads((FIXTURES / "rag_core_baseline.json").read_text(encoding="utf-8"))
APPROVED_CHANGES = json.loads((FIXTURES / "approved_legacy_changes.json").read_text(encoding="utf-8"))["changes"]
APPROVED_REMOVALS = json.loads((FIXTURES / "approved_legacy_removals.json").read_text(encoding="utf-8"))["removals"]

UNCHANGED = CORE["unchanged_files_sha256"]
MIGRATION_CHANGES = {**CORE["changed_during_migration"], **CORE.get("changed_in_phase3", {})}
REPLACED = CORE.get("replaced_in_phase3", {})

# Sanad's own packages, scanned for forbidden vector-database writes. rag/ is included: the RAG
# retrieves and never ingests, and that must stay true now that it lives in the tree.
SANAD_PACKAGE_DIRS = ("agents", "api", "extraction", "models", "orchestrator", "parsers", "rag", "tools")

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


# --------------------------------------------------------------------------- the migrated core
def test_the_migrated_rag_core_and_source_data_are_byte_identical(baseline):
    """Every carried-over file still hashes to the value recorded before Sanad existed."""
    historical = baseline["protected_files_sha256"]
    assert len(UNCHANGED) == 6
    for must_include in ("rag/retriever.py", "rag/data/labor_law/labor_law_ar.pdf",
                         "rag/data/labor_law/labor_law_parsed.json", "eval/data/ground-truth-data.csv"):
        assert must_include in UNCHANGED, f"{must_include} must stay byte-identical"

    changed = {rel for rel, entry in UNCHANGED.items() if sha256(PROJECT_ROOT / rel) != entry["sha256"]}
    assert not changed, f"migrated RAG files were modified: {sorted(changed)}"

    # the recorded hashes must be the historical ones, not something re-baselined after the move
    for rel, entry in UNCHANGED.items():
        origin = entry["migrated_from"]
        expected = (APPROVED_CHANGES[rel]["sha256_after"] if entry.get("carries_approved_fix")
                    else historical[origin])
        assert entry["sha256"] == expected, f"{rel}: hash does not come from the historical record"


def test_the_approved_fix_is_still_the_only_change_to_the_retriever(baseline):
    for rel, change in APPROVED_CHANGES.items():
        data = (PROJECT_ROOT / rel).read_bytes()
        fixed_line, original_line = (change["fixed_line"] + "\n").encode(), (change["original_line"] + "\n").encode()
        assert data.count(fixed_line) == 1, f"approved fix not present in {rel}"
        assert data.decode("utf-8").splitlines()[change["line_number"] - 1] == change["fixed_line"]
        assert hashlib.sha256(data).hexdigest() == change["sha256_after"], f"{rel} has changes beyond the approved line"
        reverted = data.replace(fixed_line, original_line)
        assert (hashlib.sha256(reverted).hexdigest() == change["sha256_before"]
                == baseline["protected_files_sha256"][change["migrated_from"]])


def test_every_recorded_change_reverts_to_the_original_file(baseline):
    """rag/backend.py and the five notebooks: undoing the recorded replacements must reproduce the original."""
    for rel, change in MIGRATION_CHANGES.items():
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == change["sha256_after"], \
            f"{rel} has changes beyond the approved migration edits"
        for replacement in change["replacements"]:
            assert text.count(replacement["after"]) == 1, f"{rel}: recorded replacement not found"
            text = text.replace(replacement["after"], replacement["before"])
        reverted = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert reverted == change["sha256_before"], f"{rel} cannot be reverted: something else changed"
        # a file changed only in Phase 3 reverts to its migrated bytes, which are the historical ones;
        # rag/backend.py reverts to the pre-Sanad file itself
        historical = baseline["protected_files_sha256"][change["migrated_from"]]
        assert reverted == historical, f"{rel}: the pre-change bytes are not the historical ones"


def test_a_replaced_file_still_matches_its_recorded_hash():
    """rag/README.md was rewritten in Phase 3; it is pinned by its new hash so it cannot drift silently."""
    assert set(REPLACED) == {"rag/README.md"}
    for rel, entry in REPLACED.items():
        assert sha256(PROJECT_ROOT / rel) == entry["sha256_after"], f"{rel} changed since it was replaced"
        assert entry["sha256_before"] != entry["sha256_after"]


def test_no_rag_file_was_added_or_removed_without_a_record():
    """rag/ holds exactly the recorded modules; data/ and qdrant_storage/ are checked separately."""
    present = {
        p.relative_to(PROJECT_ROOT).as_posix()
        for p in RAG_DIR.rglob("*.py")
        if "__pycache__" not in p.parts
    }
    recorded_core = {"rag/retriever.py", "rag/backend.py"}
    sanad_own = {"rag/__init__.py", "rag/adapter.py", "rag/errors.py", "rag/mapping.py",
                 "rag/loader.py", "rag/retrieval_config.py"}
    assert present == recorded_core | sanad_own, (
        f"unexpected in rag/: {sorted(present - recorded_core - sanad_own)}; "
        f"missing from rag/: {sorted((recorded_core | sanad_own) - present)}"
    )


def test_the_historical_record_of_the_migration_is_intact():
    """The pre-Sanad baseline and the approved-removals record are kept, unchanged, as provenance."""
    historical = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert len(historical["protected_files_sha256"]) == 26
    assert CORE["migrated_from_baseline"] == "tests/fixtures/legacy_rag_baseline.json"
    assert CORE["historical_git_head"] == historical["git_head"]

    # every one of the 26 historical files is accounted for: migrated, changed, or explicitly not migrated
    accounted = ({e["migrated_from"] for e in UNCHANGED.values()}
                 | {c["migrated_from"] for c in MIGRATION_CHANGES.values()}
                 | {r["migrated_from"] for r in REPLACED.values()}
                 | set(CORE["not_migrated"]))
    assert accounted == set(historical["protected_files_sha256"]), \
        f"unaccounted for: {sorted(set(historical['protected_files_sha256']) - accounted)}"

    # the approved removals are still recorded against their historical hashes
    for rel, removal in APPROVED_REMOVALS.items():
        assert removal["sha256_at_removal"] == historical["protected_files_sha256"][rel]
        assert rel in CORE["not_migrated"]


# --------------------------------------------------------------------------- the vector database
def test_qdrant_collections_are_still_present_with_unchanged_configuration():
    collections_dir = RAG_DIR / "qdrant_storage" / "collections"
    assert sorted(p.name for p in collections_dir.iterdir() if p.is_dir()) == CORE["qdrant"]["collections"]
    for rel, digest in CORE["qdrant"]["collection_configs_sha256"].items():
        assert sha256(PROJECT_ROOT / rel) == digest, f"{rel} changed"
    config = json.loads((collections_dir / "saudi_labor_law" / "config.json").read_text(encoding="utf-8"))
    assert config["params"]["vectors"] == {"size": 768, "distance": "Cosine"}


def test_there_is_exactly_one_qdrant_storage_directory():
    found = sorted(p.relative_to(PROJECT_ROOT).as_posix()
                   for p in PROJECT_ROOT.rglob("qdrant_storage") if p.is_dir() and ".venv" not in p.parts)
    assert found == ["rag/qdrant_storage"], found


def test_knowledge_base_still_has_all_articles(knowledge_base):
    assert len(knowledge_base) == CORE["knowledge_base"]["article_count"] == 249
    assert [a["index"] for a in knowledge_base] == list(range(1, 250))


# --------------------------------------------------------------------------- the interface and the writes
def test_the_rag_interface_used_by_the_adapter_is_intact():
    backend = ast.parse((RAG_DIR / "backend.py").read_text(encoding="utf-8"))
    functions = {n.name: n for n in backend.body if isinstance(n, ast.FunctionDef)}
    assert [a.arg for a in functions["answer_policy_question"].args.args] == ["query", "employee_data", "api_key"]
    assert {"get_retriever", "detect_language", "generate_answer", "highlight_articles"} <= set(functions)
    assert any(isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "documents" for n in backend.body)

    search = ast.parse((RAG_DIR / "retriever.py").read_text(encoding="utf-8"))
    [retriever_cls] = [n for n in search.body if isinstance(n, ast.ClassDef) and n.name == "HybridRetriever"]
    assert "retrieve" in {n.name for n in retriever_cls.body if isinstance(n, ast.FunctionDef)}

    # the backend reaches the retriever through the package, and nothing else
    imports = {n.module for n in ast.walk(backend) if isinstance(n, ast.ImportFrom)}
    assert "rag.retriever" in imports


def test_sanad_code_contains_no_vector_db_write_or_ingestion_calls():
    offenders = []
    paths = [f for d in SANAD_PACKAGE_DIRS for f in (PROJECT_ROOT / d).rglob("*.py")] + [PROJECT_ROOT / "config.py"]
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_VECTOR_DB_CALLS:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} .{node.func.attr}()")
    assert not offenders, offenders
