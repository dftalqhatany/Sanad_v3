# Sanad (سند)

Sanad is an AI-powered employment contract assistant for the Saudi labor market: it analyzes
employment contracts and CVs, compares offers, benchmarks salaries, and answers questions about
Saudi Labor Law from a hybrid RAG over the official text.

The project lives in one directory:

```
sanad/
├── rag/          the retriever, the answer backend, the adapter, the knowledge base, qdrant_storage/
├── agents/       contract analysis, CV analysis, comparison, recommendation, salary
├── orchestrator/ the single entry point: deterministic routing, no analysis of its own
├── api/          a thin HTTP layer over SanadOrchestrator.handle()
├── frontend/     the Streamlit interface; talks to the API over HTTP only
├── extraction/  parsers/  models/  tools/
├── tests/        the test suite
└── eval/         the Golden Dataset, the evaluation design, and the development notebooks
```

See [`sanad/README.md`](sanad/README.md) for the architecture, configuration and how to run it.
