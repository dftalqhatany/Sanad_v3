"""Build the regulatory portion of the Golden Dataset from the existing ground-truth CSV.

    eval/data/ground-truth-data.csv   (immutable source, never written to)
        -> golden/regulatory/source_all.jsonl    every usable row, normalised
        -> golden/regulatory/quarantined.jsonl   rows that carry no real question
        -> golden/regulatory/subset_v1.jsonl     a deterministic, stratified selection

Nothing here decides which article is correct: the gold label is inherited from the source dataset,
whose article_number was verified to agree with labor_law_parsed.json at every one of its 1,245 rows.
Every emitted case says so in `review.status = "source_derived"`.

Run:  python eval/build_regulatory.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent  # the project root
SOURCE_CSV = REPO_ROOT / "eval" / "data" / "ground-truth-data.csv"
KB_PATH = REPO_ROOT / "rag" / "data" / "labor_law" / "labor_law_parsed.json"
OUT = EVAL_DIR / "golden" / "regulatory"

# Rows whose "question" is a generator placeholder rather than a question.
PLACEHOLDER = {f"question{n}" for n in range(1, 10)}
SUBSET_TARGET = 40


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def language_of(text: str) -> str:
    arabic = sum(1 for c in text if "؀" <= c <= "ۿ")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    if arabic and latin > arabic:
        return "mixed"
    return "ar" if arabic else ("en" if latin else "not_applicable")


def overlap(question: str, article: str) -> float:
    """Token overlap the way the BM25 in rag/retriever.py sees it: whitespace split, no normalisation."""
    q, a = set(question.split()), set(article.split())
    return len(q & a) / len(q) if q else 0.0


def build() -> dict:
    csv_hash, kb_hash = sha256(SOURCE_CSV), sha256(KB_PATH)
    kb = {d["index"]: d for d in json.loads(KB_PATH.read_text(encoding="utf-8"))}
    rows = list(csv.DictReader(SOURCE_CSV.open(encoding="utf-8")))

    # group identical question strings so duplicates stay visible instead of being dropped
    seen: dict[str, list[int]] = {}
    for number, row in enumerate(rows, 2):
        seen.setdefault(row["question"].strip(), []).append(number)

    usable, quarantined = [], []
    for number, row in enumerate(rows, 2):
        question = row["question"].strip()
        index, article_number = int(row["index"]), row["article_number"].strip()
        placeholder = question.lower() in PLACEHOLDER
        group = f"dup_{hashlib.sha1(question.encode()).hexdigest()[:8]}" if len(seen[question]) > 1 else None

        provenance = {
            "source": "eval/data/ground-truth-data.csv",
            "source_row": number,
            "source_sha256": csv_hash,
            "authority": "Saudi Labor Law (nizam al-amal), as parsed into labor_law_parsed.json",
            "law_snapshot": kb_hash,
            "law_version": None,
            "effective_date": None,
            "unavailable_fields": ["law_version", "effective_date", "amendment_reference"],
        }

        if placeholder:
            quarantined.append({
                "case_id": f"reg_quar_{number:04d}",
                "task": "regulatory_question",
                "title": f"Placeholder row {number} (article index {index})",
                "language": "not_applicable",
                "input": {"question": question},
                "expected": {},
                "gold": {"article_indices": [index], "article_numbers": [article_number]},
                "provenance": provenance,
                "review": {"status": "quarantined", "reason": "the source row carries the generator placeholder "
                                                              "'questionN' instead of a question; it has no retrieval "
                                                              "signal and must not be scored"},
                "duplicate_group": group,
                "tags": ["placeholder", "excluded_from_metrics"],
                "notes": "Kept for provenance and to make the source dataset's 40 failed generations visible.",
            })
            continue

        article_text = kb[index]["arabic_content"]
        usable.append({
            "case_id": f"reg_src_{number:04d}",
            "task": "regulatory_question",
            "title": f"Q{number} → article {article_number} (kb index {index})",
            "language": language_of(question),
            "input": {"question": question},
            "expected": {"retrieval": {"must_retrieve_article_index": index, "at_k": [3, 5]}},
            "gold": {"article_indices": [index], "article_numbers": [article_number]},
            "provenance": provenance,
            "review": {"status": "source_derived", "method": "inherited_from_source_dataset"},
            "duplicate_group": group,
            "tags": ["retrieval"],
            "notes": "",
            "_overlap": round(overlap(question, article_text), 4),   # stripped before writing
            "_index": index,
        })

    subset = select_subset(usable, kb)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "source_all.jsonl", [strip(c) for c in usable])
    write(OUT / "quarantined.jsonl", quarantined)
    write(OUT / "subset_v1.jsonl", subset)
    return {"rows": len(rows), "usable": len(usable), "quarantined": len(quarantined), "subset": len(subset),
            "csv_sha256": csv_hash, "kb_sha256": kb_hash}


def select_subset(usable: list[dict], kb: dict) -> list[dict]:
    """A deliberately mixed selection. Deterministic: no randomness, no seed, stable ordering."""
    by_overlap = sorted(usable, key=lambda c: (c["_overlap"], c["case_id"]))
    picked: dict[str, dict] = {}

    def take(cases, tag, count):
        for case in cases:
            if len(picked) >= SUBSET_TARGET or count <= 0:
                return
            if case["case_id"] in picked:
                continue
            chosen = dict(case)
            chosen["tags"] = [*case["tags"], tag]
            picked[case["case_id"]] = chosen
            count -= 1

    # 1. articles whose legal number is shared by two knowledge-base entries ("مكرر" / bis articles):
    #    the hardest case for article-number accuracy.
    shared: dict[str, list[int]] = {}
    for index, article in kb.items():
        shared.setdefault(str(article.get("article_number")), []).append(index)
    bis = {i for indices in shared.values() if len(indices) > 1 for i in indices}
    take([c for c in usable if c["_index"] in bis], "bis_article", 8)

    # 2. high lexical overlap - BM25 should find these easily; a miss here is a real problem
    take(list(reversed(by_overlap)), "high_lexical_overlap", 8)

    # 3. low lexical overlap - paraphrase-like; these separate dense retrieval from BM25
    take(by_overlap, "low_lexical_overlap", 8)

    # 4. adjacent-article confusion risk: short articles with a neighbour in the same chapter
    neighbours = [c for c in usable
                  if len(kb[c["_index"]]["arabic_content"]) < 400
                  and (c["_index"] + 1 in kb or c["_index"] - 1 in kb)]
    take(sorted(neighbours, key=lambda c: c["case_id"]), "near_neighbour", 8)

    # 5. spread across the whole law so the subset is not clustered
    spread = sorted(usable, key=lambda c: (c["_index"], c["case_id"]))
    step = max(1, len(spread) // max(1, SUBSET_TARGET - len(picked)))
    take(spread[::step], "coverage_spread", SUBSET_TARGET - len(picked))

    out = []
    for case in sorted(picked.values(), key=lambda c: c["case_id"]):
        entry = strip(case)
        entry["case_id"] = case["case_id"].replace("reg_src_", "reg_v1_")
        entry["notes"] = (f"Selected into subset v1 from source row {case['provenance']['source_row']} "
                          f"(source case {case['case_id']}). Label inherited, not re-reviewed.")
        out.append(entry)
    return out


def strip(case: dict) -> dict:
    return {k: v for k, v in case.items() if not k.startswith("_")}


def write(path: Path, cases: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")


if __name__ == "__main__":
    stats = build()
    print(json.dumps(stats, indent=2))
