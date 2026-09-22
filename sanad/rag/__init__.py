"""Sanad's RAG: the retriever, the answer backend, and the typed adapter the agents use.

    retriever.py  HybridRetriever: dense (e5-base via Qdrant) + BM25, MinMax-scaled, fused 0.6/0.4, top-5
    backend.py    answer_policy_question() and the helpers around it
    adapter.py    RegulatoryRAGAdapter: typed, traceable results and explicit errors
    errors.py     the RAG error taxonomy   mapping.py  legacy output -> Sanad models
    data/         the regulatory knowledge base    qdrant_storage/  the Qdrant volume

Importing this package does not connect to Qdrant or load the embedding model: rag.backend and
rag.retriever are imported lazily, on first use, by rag.loader.
"""

from rag.adapter import RegulatoryRAGAdapter
from rag.errors import RagErrorCode

__all__ = ["RagErrorCode", "RegulatoryRAGAdapter"]
