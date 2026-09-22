# Sanad (سند)

## Team Project

**Sanad** is an AI-powered employment contract assistant for the Saudi labor market.

The project was developed as part of the **LLM Zoomcamp** and extends an existing Saudi labor-law RAG system.

---

## Team Members

| Name | Role |
|------|------|
| Team Member 1 | |
| Team Member 2 | |
| Team Member 3 | |
| Team Member 4 | |

> Add each team member's name and primary responsibility above.

---

## Project Goal

Sanad helps employees and HR teams understand employment contracts and relevant Saudi labor regulations.

The system can:

- Analyze employment contracts
- Retrieve relevant Saudi labor regulations
- Explain contract clauses
- Compare multiple contracts
- Analyze CVs
- Benchmark salaries
- Provide structured recommendations

---

## Architecture

Sanad is a modular multi-agent system built around one hybrid RAG, which lives in `rag/` inside this
project. Each layer depends only on the one below it, and the boundaries are enforced by tests:

```
frontend/  -> HTTP ->  api/  ->  orchestrator/  ->  agents/  ->  rag/  ->  Qdrant + the labor law
```

### Main Components

1. **Orchestrator**
   - Receives the user request
   - Determines the required workflow
   - Routes the request to the appropriate components

2. **Regulatory Agent**
   - Retrieves relevant Saudi labor regulations
   - Uses the existing RAG knowledge base
   - Provides regulatory context for answers

3. **Contract Analysis Agent**
   - Extracts information from contracts
   - Analyzes contract terms
   - Identifies relevant regulatory considerations

4. **CV Analysis Agent**
   - Extracts information from CVs
   - Analyzes candidate information
   - Supports salary benchmarking workflows

5. **Comparison Agent**
   - Compares multiple contracts
   - Identifies differences between contract terms
   - Produces structured comparison results

6. **Salary Agent**
   - Extracts salary-related information
   - Uses available salary sources
   - Supports market salary benchmarking

---

## RAG System

There is exactly one RAG implementation, in `rag/`:

| File | What it does |
|---|---|
| `retriever.py` | `HybridRetriever`: dense retrieval (`intfloat/multilingual-e5-base` via Qdrant) fused with BM25. Both score vectors are MinMax-scaled and combined as `0.6 · dense + 0.4 · BM25`; the top 5 articles are returned. |
| `backend.py` | `answer_policy_question()`: retrieval plus grounded answer generation, with the article references it used. |
| `adapter.py` | `RegulatoryRAGAdapter`: the only way the agents reach the RAG. Returns typed, traceable results and reports every failure explicitly instead of raising. |
| `errors.py` | The RAG error taxonomy: a missing dependency, an unreachable Qdrant, a missing collection and an embedding-model failure are distinct, machine-readable codes. |
| `mapping.py` | Converts retriever and backend output into Sanad's models. Missing fields become `None`; nothing is invented. |
| `loader.py` | Imports `rag.backend` lazily, once, under a lock — importing it connects to Qdrant and loads the embedding model, so nothing may import it eagerly. |
| `retrieval_config.py` | Reads the Qdrant URL, collection, embedding model, top-k and fusion weight out of `retriever.py` by parsing it, without importing it. `retriever.py` stays the single source of truth. |
| `data/labor_law/` | The source PDF and the 249 parsed articles the retriever searches. |
| `qdrant_storage/` | The Qdrant volume: collections `saudi_labor_law` (live) and `labor_law_ar`. |

Importing `rag` is cheap and has no side effects: no Qdrant connection, no embedding model, no OpenAI
call happens until an adapter call needs one.

---

## Running Sanad

From the project root, with Qdrant running:

```bash
python -m api                                   # the service, on http://127.0.0.1:8000
PYTHONPATH=. streamlit run frontend/app.py      # the interface, on http://localhost:8501
```

---

## Running Qdrant

The regulatory RAG reads the `saudi_labor_law` collection from a Qdrant server at the URL declared in
`rag/retriever.py` (`http://localhost:6333`). The storage lives inside the project, so run this
**from the project root**:

```bash
docker run -p 6333:6333 -p 6334:6334 \
   -v "$(pwd)/rag/qdrant_storage:/qdrant/storage:z" \
   qdrant/qdrant
```

This is the only `qdrant_storage` in the project. The collections are not rebuilt by running Sanad;
to regenerate them from the source PDF, see `eval/notebooks/`.

---

## Configuration

Sanad reads its settings from environment variables (`sanad/config.py`). No key is hard-coded, and no key is
logged or included in a `repr`. Retrieval settings — the Qdrant URL, collection, embedding model, top-k and
fusion weight — are deliberately **not** configured here: they are read from the existing
`rag/retriever.py`, which stays the single source of truth.

### Salary benchmarking

| Variable | Default | Effect |
|---|---|---|
| `SANAD_SEARCH_PROVIDER` | `none` | Search provider used to find salary sources: `none`, `tavily` or `brave`. |
| `SANAD_SEARCH_API_KEY` | unset | API key for that provider. Also read from `TAVILY_API_KEY` / `BRAVE_API_KEY`. |
| `SANAD_SALARY_ENABLED` | unset (off) | Set to `1`, `true` or `yes` to benchmark from the documented seed sources without a search provider. |

Salary benchmarking runs when `SANAD_SALARY_ENABLED` is set, **or** when `SANAD_SEARCH_PROVIDER` is not `none`
and an API key is present. With neither, Sanad returns a salary benchmark whose status is `not_configured`:
no market data is used, and none is estimated or invented.

### Answer generation

| Variable | Default | Effect |
|---|---|---|
| `OPENAI_API_KEY` | unset | Passed to the existing RAG's answer generation. Without it, no LLM interpretation runs and regulatory findings stay `requires_review`. |
