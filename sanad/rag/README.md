# Sanad RAG

The one retrieval-augmented generation system in Sanad. Everything in this directory serves a single
purpose: answering questions about Saudi Labor Law from the official text, with the articles used
traceable in every result.

This package grew out of the standalone `hr_assistant` project. That project no longer exists — its
retriever, answer backend, knowledge base and Qdrant volume were migrated here unchanged. The
original README is preserved in git history, and every migrated file's hash is pinned in
`tests/fixtures/rag_core_baseline.json`.

---

## Data source

The Arabic text of نظام العمل (the Saudi Labor Law), published by the Ministry of Human Resources
and Social Development:

<https://www.hrsd.gov.sa/sites/default/files/2025-07/nzam-al-ml----wfq-alhwyt-aljdydt-2.pdf>

`data/labor_law/labor_law_ar.pdf` is that document. `data/labor_law/labor_law_parsed.json` is the
249 articles parsed out of it, each with its part, chapter, Arabic name and number, Arabic content,
and an English translation. **The Arabic text is the legal reference**; the English was produced by
machine translation (`Helsinki-NLP/opus-mt-ar-en`) and is a reading aid, not an authority.

The parsing and embedding pipeline is not part of the runtime. It lives in `../eval/notebooks/`:
`process_labor_pdf.ipynb` builds the parsed JSON from the PDF, and `process_data_vectors.ipynb`
embeds the articles and writes the Qdrant collection. Sanad itself never writes to Qdrant.

---

## Retrieval

`retriever.py` holds the configuration constants and `HybridRetriever`, and it is the single source
of truth for all of them — `retrieval_config.py` reads them out of it by parsing the file rather
than importing it, because importing connects to Qdrant and loads the embedding model.

| Setting | Value |
|---|---|
| Qdrant | `http://localhost:6333`, collection `saudi_labor_law` |
| Vectors | 768 dimensions, Cosine distance |
| Embeddings | `intfloat/multilingual-e5-base` |
| Dense retrieval | LlamaIndex `VectorIndexRetriever`, `similarity_top_k = 3` |
| Lexical retrieval | `BM25Okapi` over the Arabic content |
| Fusion | both score vectors MinMax-scaled, then `0.6 · dense + 0.4 · BM25` |
| Results | top 5 articles, each with its index, score, content and full metadata |

`backend.py` puts an answer on top of that: it detects the question's language, retrieves, and asks
the configured OpenAI model to answer **only** from the retrieved articles, returning the answer
together with the references it used. The API key is supplied per call and is never stored or logged.

---

## Files

| File | Role |
|---|---|
| `retriever.py` | `HybridRetriever` and the retrieval configuration |
| `backend.py` | `answer_policy_question()` and its helpers |
| `adapter.py` | `RegulatoryRAGAdapter` — the only door the agents use; typed results, explicit failures, a read-only health check |
| `errors.py` | the RAG error taxonomy, kept distinct so a missing dependency never looks like an empty answer |
| `mapping.py` | retriever/backend output → Sanad models; missing fields become `None` |
| `loader.py` | lazy, locked, verified import of `rag.backend` |
| `retrieval_config.py` | reads the retrieval settings out of `retriever.py` without importing it |
| `data/labor_law/` | the source PDF and the 249 parsed articles |
| `qdrant_storage/` | the Qdrant volume — the only one in the project |

Importing this package is cheap: `rag.backend` and `rag.retriever` are loaded on first use, so no
Qdrant connection, embedding model or OpenAI import happens at import time.

---

## Running Qdrant

From the project root, with the collections already built:

```bash
docker run -p 6333:6333 -p 6334:6334 \
   -v "$(pwd)/rag/qdrant_storage:/qdrant/storage:z" \
   qdrant/qdrant
```

`saudi_labor_law` is the collection the retriever uses. `labor_law_ar` is an earlier collection kept
alongside it; nothing in Sanad reads it.
