# Sanad

Sanad analyses Saudi employment contracts and CVs on top of the **existing** `hr_assistant` Saudi Labor Law RAG,
which it reuses unchanged through an adapter. Nothing in `hr_assistant/` is modified by this project.

```
frontend (Streamlit)  ->  HTTP  ->  sanad.api  ->  SanadOrchestrator.handle()
                                                        |- AnalysisAgent           (contract + CV, salary interface)
                                                        |- ContractComparisonAgent (2-5 contracts, one analysis each)
                                                        `- RegulatoryRAGAdapter    -> existing hr_assistant RAG -> Qdrant
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # add ../hr_assistant/requirements.txt to run the real RAG
```

## Run

```bash
python -m sanad.api                      # API on http://127.0.0.1:8000  (docs at /docs)
streamlit run frontend/app.py            # UI  on http://localhost:8501
```

Environment variables: `OPENAI_API_KEY` (optional; enables written answers and evidence-grounded interpretation),
`SANAD_LEGACY_RAG_DIR`, `SANAD_MAX_UPLOAD_MB`, `SANAD_API_HOST`, `SANAD_API_PORT`, `SANAD_API_MAX_FILES`,
`SANAD_API_CORS_ORIGINS`, `SANAD_ANALYSIS_LLM_MODEL`. See `sanad/config.py`.

## Salary benchmarking (real web sources)

Benchmarking is off until it is configured; until then it reports `not_configured` and no market figure is shown.

```bash
export SANAD_SALARY_ENABLED=1                 # read the documented sources directly (no API key needed)
# or, to search the web as well:
export SANAD_SEARCH_PROVIDER=tavily           # or 'brave'
export SANAD_SEARCH_API_KEY=...               # also read from TAVILY_API_KEY / BRAVE_API_KEY; never hard-coded
```

Optional: `SANAD_SALARY_DOMAINS` (allowlist; only these domains are searched, fetched and cited),
`SANAD_SALARY_SEEDS`, `SANAD_SALARY_TIMEOUT_S` (default 10), `SANAD_SALARY_MAX_RESULTS` (default 8),
`SANAD_SALARY_CACHE_TTL_S` (default 3600), `SANAD_SALARY_MIN_INTERVAL_S` (default 0.5),
`SANAD_SALARY_FX_RATES` (e.g. `USD:3.75,EUR:4.05`; a currency without a rate is reported but never converted).

Source ranking: official Saudi statistics (GASTAT, Saudi Open Data) outrank market sites (SaudiSalary, Paylab),
which outrank supplementary datasets (Kaggle). LinkedIn is a discovery lead only and never sets a range.
A result is always a range with its sources, or `insufficient_data` with the reason; sources that disagree are
reported separately instead of averaged, and base salary is never merged with total compensation.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | liveness only |
| `GET /api/config` | upload limits, tasks, roles, comparison priorities |
| `POST /api/analyze` | multipart upload of PDF/DOCX contracts and CV (+ question, task, priorities, labels, target_job) |
| `POST /api/ask` | a regulatory question with no documents |

Every routed request is answered with an `OrchestratorResult` JSON body. HTTP status follows its `status`:
200 (success / partial / insufficient_evidence), 422 (invalid_input), 502 (rag_error), 500 (analysis_error);
uploads rejected at the edge return 400/413/415 with `{"error": {...}}`.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -v -rxXs
```

`tests/integrity` proves the existing RAG is untouched; `tests/live` (skipped by default) needs a running Qdrant,
the e5 embedding model and the legacy dependencies.
