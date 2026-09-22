# Sanad Golden Dataset

Evaluation assets for Sanad. Nothing in this folder is imported by `sanad/` or `frontend/`, and
nothing here changes product behaviour. The test suite under `sanad/tests/` is unaffected: pytest
collects `testpaths = ["tests"]`, so this folder is never picked up by a normal test run.

---

## 1. Why this exists

The 506 tests under `tests/` answer one question: *does the code do what it was written to do?*
They do not answer the question this dataset exists for: *is the output correct, grounded and
reliable?*

Before this dataset there was no way to say whether a change to retrieval, a prompt, a parser or a
salary source made Sanad better or worse. The only reference material in the repository was an
unused ground-truth CSV and two files of historical model output. This dataset turns the first of
those into something Sanad's evaluation can actually consume, and adds the cases the CSV cannot
supply.

---

## 2. Three different things, kept apart

| | What it is | Where | May it define "correct"? |
|---|---|---|---|
| **Source dataset** | `ground-truth-data.csv` — 1,245 rows of question → article, produced by `evaluation_data.ipynb` | `eval/data/` | **Yes, for retrieval** — its article numbers agree with the knowledge base at 100% of rows |
| **Golden dataset** | This folder — normalised, typed, provenance-tracked cases | `sanad/eval/golden/` | Yes, within the review status each case declares |
| **Historical model output** | `results-gpt35.csv`, `results-gpt4o.csv` — 1,245 answers each from older models | `eval/data/` | **No. Never.** |

---

## 3. Why `ground-truth-data.csv` is used for retrieval evaluation

It is the only labelled set in the repository, and it was verified before use:

* 1,245 rows, 4 columns (`question`, `article_number`, `article_orig`, `index`), **zero missing values**
* 249 distinct `index` values, exactly **5 questions per article**
* `article_number` **agrees with `labor_law_parsed.json` at every single row** (1,245/1,245)
* `article_orig` is **byte-identical** to the knowledge base's `arabic_content` at that index (1,245/1,245)
* `index` is the same 1-based join key the retriever uses (`rag/retriever.py:76`)

Two caveats travel with every metric computed from it:

1. **The questions were generated from the article text**, so they share vocabulary with their gold
   article. Retrieval scores on this set are an **upper bound**, not a field estimate.
2. **40 rows are unusable.** For eight articles (indices 31, 38, 47, 72, 113, 143, 207, 238) the
   generator wrote the literal placeholders `question1`…`question5`. Those rows are quarantined, not
   deleted — see `golden/regulatory/quarantined.jsonl`.

After quarantine: **1,205 usable questions, all Arabic, covering 241 of 249 articles.**
The source dataset contains **no English questions at all.**

---

## 4. Why the GPT result files are not ground truth

`results-gpt35.csv` and `results-gpt4o.csv` are what two models answered in 2026, under a
configuration that no longer matches the product. They are model output, not law. Using them as
truth would measure how well Sanad imitates an older model rather than whether Sanad is right.
They may be used for one purpose only: as optional comparative material when discussing drift.
The `cosine` column that `rag_evaluation.ipynb` computes is **not present in the saved files**.

---

## 5. Structure

```
eval/
├── README.md                     this file
├── schema.py                     typed case schema; invalid cases fail at load time
├── build_regulatory.py           ground-truth-data.csv  ->  regulatory cases
├── build_fixture_cases.py        repository fixtures    ->  contract/comparison/salary/robustness
├── validate.py                   deterministic validator (also `--freeze`)
├── config/
│   └── dataset.json              dataset version + sha256 of every source it was built from
├── golden/
│   ├── regulatory/
│   │   ├── source_all.jsonl      every usable source row, normalised
│   │   ├── subset_v1.jsonl       a stratified selection for fast runs
│   │   └── quarantined.jsonl     rows with no real question, kept for provenance
│   ├── contract/cases.jsonl
│   ├── comparison/cases.jsonl
│   ├── salary/cases.jsonl
│   └── robustness/cases.jsonl
├── results/                      Checkpoint 10 writes run output here
└── reports/                      Checkpoint 10 writes rendered reports here
```

---

## 6. Case schema

One JSON object per line. Enforced by `eval/schema.py` (pydantic, `extra="forbid"`).

```jsonc
{
  "case_id": "reg_src_0152",                  // unique, lower_snake_case
  "task": "regulatory_question",              // regulatory_question | contract_analysis |
                                              // contract_comparison | salary_benchmark | robustness
  "title": "...",
  "language": "ar",                           // ar | en | mixed | not_applicable
  "input":  { "question": "..." },            // or documents / salary_query / scenario
  "expected": { "retrieval": { "must_retrieve_article_index": 53, "at_k": [3, 5] } },
  "gold":   { "article_indices": [53], "article_numbers": ["52"] },
  "provenance": {
    "source": "hr_assistant/data/ground-truth-data.csv",   // historical: where the row came from when
                                                           // the dataset was built, kept verbatim in every case
    "source_row": 152,
    "source_sha256": "a8757fd1…",
    "authority": "Saudi Labor Law …",
    "law_snapshot": "972cd578…",
    "law_version": null,                      // not available in this repository
    "effective_date": null,                   // not available in this repository
    "unavailable_fields": ["law_version", "effective_date", "amendment_reference"]
  },
  "review": { "status": "source_derived", "method": "inherited_from_source_dataset" },
  "duplicate_group": null,
  "tags": ["retrieval"],
  "notes": ""
}
```

**`article_indices` and `article_numbers` are both required and are not interchangeable.** In the
source data they differ in 95.6% of rows (offsets 0–4), and four legal article numbers are shared by
two knowledge-base entries each — `11 → [11, 12]`, `79 → [80, 81]`, `131 → [133, 134]`,
`229 → [232, 233]` (the "مكرر" / bis articles). Matching on the article number alone would score
those as correct when the wrong text was retrieved.

---

## 7. Review status — what a case is actually worth

| Status | Meaning |
|---|---|
| `source_derived` | The label was inherited from `ground-truth-data.csv`. Nobody re-read the law. |
| `generated_from_fixture` | Derived mechanically from a fixture whose content is known, e.g. "this document says 90 days, so extraction must return 90 days". |
| `needs_review` | A human must supply or confirm the expectation. Carries the exact question to answer. |
| `reviewed` | A named person verified it against the primary source. Requires `reviewed_by`, `reviewed_at`, `method`. |
| `quarantined` | Kept for provenance, excluded from every metric. Carries a reason. |

**No case in v1 is `reviewed`.** Nothing here has been checked by a person against the Arabic text of
the law, and the dataset says so rather than implying otherwise. `source_derived` is a real label —
it is as good as the CSV, which was verified against the knowledge base — but it is not human review.

---

## 8. Duplicate handling

Seven question strings appear more than once in the source, covering 44 rows. **All of them are the
`questionN` placeholders** — there are no genuine duplicate questions in the usable data. Every
colliding case carries a `duplicate_group` id so the collision stays visible.

Rule for metrics: **a duplicate group counts once.** Five near-identical questions about one article
are not five independent pieces of evidence, and scoring them separately would inflate whichever
retrieval mode happens to suit that article. The same applies to the five questions per article: when
reporting, give both the per-question number and the per-article number.

---

## 9. Legal provenance

Each case records the authority, the knowledge-base snapshot hash, and — explicitly — the fields the
repository cannot supply. `labor_law_parsed.json` has **no version, amendment or effective-date
field**, so `law_version` and `effective_date` are `null` and listed in `unavailable_fields`. They are
not guessed. Until that metadata exists, a case cannot be shown to be current; it can only be shown
to match the snapshot it was built from.

---

## 10. Adding a case

1. Add it to the right builder (`build_regulatory.py` or `build_fixture_cases.py`) so it is
   reproducible, or append a line to the relevant `.jsonl` if it is genuinely one-off.
2. Give it honest provenance and the weakest review status that is true.
3. If it asserts a legal conclusion, leave `expected.findings[].status` as `TO_BE_REVIEWED` and
   `expected_articles` empty, and set `review.status = "needs_review"` with the question for the
   reviewer. The validator rejects a filled-in legal conclusion that nobody reviewed.
4. Run `python eval/validate.py` and make it pass.
5. If a source file changed, re-run `python eval/validate.py --freeze` and say why in the commit.

---

## 11. What must never be ground truth

* `results-gpt35.csv`, `results-gpt4o.csv`, or any other model output
* an LLM's opinion about which article applies
* a compliance status nobody verified against the Arabic text
* a single "correct salary" — salary is evaluated on methodology, never against an invented figure
* a retrieved article, used to justify the expectation that it should have been retrieved

---

## 12. How Checkpoint 10 consumes this

The evaluation runner will:

1. load every `.jsonl` through `eval/schema.py`, skipping `quarantined`;
2. for `regulatory_question`, call `RegulatoryRAGAdapter.retrieve_evidence()` and compute Hit Rate@3,
   Hit Rate@5, MRR@5 and article-number accuracy against `gold.article_indices`, reporting
   per-question and per-article figures separately;
3. for `contract_analysis`, materialise the fixture recipe with the existing builders and score the
   `generated_from_fixture` extraction expectations — the `needs_review` compliance halves are
   reported as "awaiting review", never as failures;
4. for `contract_comparison`, check the declared dimension, comparability and fallback expectations;
5. for `salary_benchmark`, inject the named fixture pages through `FakeWebSearchClient` and check
   methodology, not amounts;
6. for `robustness`, run the scenario and check behaviour;
7. write raw output to `results/` and a rendered scorecard to `reports/`, with the dataset version
   and every source hash from `config/dataset.json` stamped on the run.
