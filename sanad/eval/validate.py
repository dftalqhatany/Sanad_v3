"""Validate the Sanad Golden Dataset. Deterministic, offline, no model and no network.

This proves the dataset itself is sound before anything is measured with it:

  * every case parses against the typed schema (eval/schema.py)
  * case ids are unique across every file
  * a regulatory case's article number agrees with labor_law_parsed.json at its index
  * the immutable source files still hash to what the dataset was built from
  * no case claims to be 'reviewed' without naming a reviewer
  * no compliance conclusion was invented: a case asserting findings is 'needs_review' with no articles
  * duplicate question strings are grouped explicitly instead of silently deduplicated

Run:      python eval/validate.py
Freeze:   python eval/validate.py --freeze     (records current source hashes into config/dataset.json)
Exit code 0 = valid, 1 = invalid.
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from eval.schema import GoldenCase, load_cases  # noqa: E402

REPO_ROOT = EVAL_DIR.parent  # the project root
CONFIG = EVAL_DIR / "config" / "dataset.json"
GOLDEN = EVAL_DIR / "golden"

# Paths moved in the Phase 2 restructure; the file CONTENTS, and therefore every sha256 in
# config/dataset.json, are unchanged. The dataset was not rebuilt.
IMMUTABLE_SOURCES = {
    "ground_truth_csv": "eval/data/ground-truth-data.csv",
    "knowledge_base": "rag/data/labor_law/labor_law_parsed.json",
    "results_gpt35": "eval/data/results-gpt35.csv",
    "results_gpt4o": "eval/data/results-gpt4o.csv",
}
FIXTURE_SOURCES = {
    "synthetic_content": "tests/fixtures/documents/synthetic_content.py",
    "builders": "tests/fixtures/documents/builders.py",
    "salary_pages": "tests/fixtures/salary_pages.py",
}
VALID_STATUSES = {"source_derived", "generated_from_fixture", "needs_review", "reviewed", "quarantined"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_hashes() -> dict:
    return {name: sha256(REPO_ROOT / rel) for name, rel in {**IMMUTABLE_SOURCES, **FIXTURE_SOURCES}.items()}


def freeze() -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps({
        "dataset_version": "v1",
        "built_against": {**IMMUTABLE_SOURCES, **FIXTURE_SOURCES},
        "sha256": current_hashes(),
        "law_version_metadata": None,
        "law_version_note": "The repository contains no law-version, amendment or effective-date metadata. "
                            "These fields are recorded as unavailable rather than invented.",
        "never_ground_truth": ["eval/data/results-gpt35.csv", "eval/data/results-gpt4o.csv"],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"frozen: {CONFIG.relative_to(REPO_ROOT)}")


def main() -> int:
    problems: list[str] = []
    notes: list[str] = []

    if not CONFIG.exists():
        print("FAIL: config/dataset.json is missing; run with --freeze first")
        return 1
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    # 1. the immutable sources must still be what the dataset was built from
    for name, digest in current_hashes().items():
        expected = config["sha256"].get(name)
        if expected is None:
            problems.append(f"source '{name}' is not recorded in config/dataset.json")
        elif expected != digest:
            problems.append(f"source '{name}' changed since the dataset was built "
                            f"(recorded {expected[:12]}…, now {digest[:12]}…)")

    # 2. load and schema-validate every case file
    files = sorted(GOLDEN.rglob("*.jsonl"))
    if not files:
        problems.append("no .jsonl case files found under eval/golden/")
    cases: list[GoldenCase] = []
    per_file: dict[str, list[GoldenCase]] = {}
    for path in files:
        try:
            loaded = load_cases(path)
        except ValueError as exc:
            problems.append(f"schema error: {exc}")
            continue
        per_file[str(path.relative_to(EVAL_DIR))] = loaded
        cases.extend(loaded)

    # 3. unique case ids across the whole dataset
    ids = collections.Counter(c.case_id for c in cases)
    for case_id, count in ids.items():
        if count > 1:
            problems.append(f"duplicate case_id '{case_id}' appears {count} times")

    # 4. regulatory gold must agree with the knowledge base
    kb = {d["index"]: d for d in json.loads((REPO_ROOT / IMMUTABLE_SOURCES["knowledge_base"]).read_text("utf-8"))}
    checked = 0
    for case in cases:
        if case.task != "regulatory_question" or case.gold is None:
            continue
        for index, number in zip(case.gold.article_indices, case.gold.article_numbers):
            entry = kb.get(index)
            if entry is None:
                problems.append(f"{case.case_id}: index {index} is not in the knowledge base")
            elif str(entry.get("article_number")) != str(number):
                problems.append(f"{case.case_id}: article_number {number} disagrees with the knowledge base "
                                f"({entry.get('article_number')}) at index {index}")
            else:
                checked += 1

    # 5. review honesty
    for case in cases:
        if case.review.status not in VALID_STATUSES:
            problems.append(f"{case.case_id}: unknown review status '{case.review.status}'")
        if case.review.status == "reviewed" and not case.review.reviewed_by:
            problems.append(f"{case.case_id}: claims 'reviewed' without a reviewer")

    # 6. no invented legal conclusion
    for case in cases:
        findings = case.expected.get("findings") or []
        for finding in findings:
            if case.review.status != "needs_review":
                problems.append(f"{case.case_id}: asserts a compliance finding without 'needs_review'")
            if finding.get("expected_articles"):
                problems.append(f"{case.case_id}: an article number was filled in without human review")
            if finding.get("status") not in (None, "TO_BE_REVIEWED"):
                problems.append(f"{case.case_id}: a compliance status was filled in without human review")

    # 7. duplicate question strings must be grouped, not dropped.
    #    Checked WITHIN a split: two cases in the same file asking the same question are the leakage
    #    risk. The same question appearing in source_all and in a subset of it is a selection, not a
    #    duplicate, and is accounted for separately in check 9.
    for name, loaded in per_file.items():
        by_question: dict[str, list[str]] = collections.defaultdict(list)
        for case in loaded:
            question = (case.input.get("question") or "").strip()
            if question:
                by_question[question].append(case.case_id)
        for question, group in by_question.items():
            if len(group) > 1 and any(c.duplicate_group is None for c in loaded if c.case_id in group):
                problems.append(f"{name}: duplicate question shared by {group} is not marked with duplicate_group")

    # 8. provenance present
    for case in cases:
        if not case.provenance.source:
            problems.append(f"{case.case_id}: provenance.source is empty")

    # 9. a subset must be a view over the source, not a second copy with its own labels
    source_rows = {c.provenance.source_row: c for c in cases if c.case_id.startswith("reg_src_")}
    subset_overlap = 0
    for case in cases:
        if not case.case_id.startswith("reg_v1_"):
            continue
        origin = source_rows.get(case.provenance.source_row)
        if origin is None:
            problems.append(f"{case.case_id}: subset case has no matching source case for row "
                            f"{case.provenance.source_row}")
            continue
        subset_overlap += 1
        if case.gold and origin.gold and case.gold.article_indices != origin.gold.article_indices:
            problems.append(f"{case.case_id}: subset gold disagrees with source case {origin.case_id}")
    notes.append(f"subset_v1 re-issues {subset_overlap} source rows under selection ids; they are the same "
                 f"underlying rows and must be counted once, not twice")

    # ---------------------------------------------------------------- report
    print("=" * 72)
    print("SANAD GOLDEN DATASET — VALIDATION")
    print("=" * 72)
    print(f"dataset version : {config['dataset_version']}")
    print(f"case files      : {len(per_file)}")
    print(f"total cases     : {len(cases)}")
    print()
    print("cases per file:")
    for name, loaded in sorted(per_file.items()):
        print(f"  {name:44s} {len(loaded):5d}")
    print()
    by_task = collections.Counter(c.task for c in cases)
    by_status = collections.Counter(c.review.status for c in cases)
    by_language = collections.Counter(c.language for c in cases)
    print("by task:      ", dict(by_task))
    print("by review:    ", dict(by_status))
    print("by language:  ", dict(by_language))
    print(f"duplicate groups: {len({c.duplicate_group for c in cases if c.duplicate_group})}"
          f" covering {sum(1 for c in cases if c.duplicate_group)} cases")
    print(f"gold article numbers cross-checked against the knowledge base: {checked}")
    scorable = [c for c in cases if c.review.status != "quarantined"]
    print(f"scorable cases (not quarantined): {len(scorable)}")
    print()
    if notes:
        for note in notes:
            print("note:", note)
    if problems:
        print(f"RESULT: INVALID — {len(problems)} problem(s)")
        for problem in problems[:40]:
            print("  -", problem)
        return 1
    print("RESULT: VALID — every check passed")
    return 0


if __name__ == "__main__":
    if "--freeze" in sys.argv:
        freeze()
        sys.exit(0)
    sys.exit(main())
