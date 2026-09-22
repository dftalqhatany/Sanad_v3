"""Doubles for the *third-party infrastructure* the legacy RAG touches at import/call time.

Only llama_index (embedding model + Qdrant vector store), qdrant_client and openai are replaced.
Everything else is REAL: rag/backend.py, rag/retriever.py
(HybridRetriever, BM25 fusion, prompts, highlight_articles), the labor_law_parsed.json knowledge
base, rank_bm25, scikit-learn and numpy.

The fake dense retriever returns nodes whose metadata is the full article dict, exactly as
process_data_vectors.ipynb stored it in Qdrant (Document(text=..., metadata=art)).
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from types import SimpleNamespace


@dataclass
class StubState:
    documents: list[dict]
    dense_hits: list[tuple[int, float]] = field(default_factory=list)  # (kb index, cosine score)
    vector_store_init_error: BaseException | None = None
    dense_retrieve_error: BaseException | None = None
    llm_answer: str = "stub answer"
    llm_error: BaseException | None = None
    embed_model_names: list[str] = field(default_factory=list)
    qdrant_urls: list[str] = field(default_factory=list)
    collections: list[str] = field(default_factory=list)
    dense_top_k: list[int] = field(default_factory=list)
    dense_queries: list[str] = field(default_factory=list)
    llm_calls: list[dict] = field(default_factory=list)
    llm_api_keys: list[str] = field(default_factory=list)


def foreign_exception(module: str, name: str, message: str, cause: BaseException | None = None) -> Exception:
    """An exception whose class appears to come from `module` (e.g. 'openai')."""
    cls = type(name, (Exception,), {"__module__": module})
    exc = cls(message)
    exc.__cause__ = cause
    return exc


def build_stub_modules(state: StubState) -> dict[str, types.ModuleType]:
    class HuggingFaceEmbedding:
        def __init__(self, model_name: str, **_: object) -> None:
            state.embed_model_names.append(model_name)
            self.model_name = model_name

    class Settings:
        embed_model = None

    class QdrantClient:
        def __init__(self, url: str | None = None, **_: object) -> None:
            state.qdrant_urls.append(url)
            self.url = url

    class QdrantVectorStore:
        def __init__(self, client, collection_name: str, **_: object) -> None:
            if state.vector_store_init_error is not None:
                raise state.vector_store_init_error
            state.collections.append(collection_name)
            self.client = client
            self.collection_name = collection_name

    class StorageContext:
        @classmethod
        def from_defaults(cls, vector_store=None, **_: object):
            ctx = cls()
            ctx.vector_store = vector_store
            return ctx

    class VectorStoreIndex:
        @classmethod
        def from_vector_store(cls, vector_store, storage_context=None, embed_model=None, **_: object):
            index = cls()
            index.vector_store = vector_store
            return index

    class VectorIndexRetriever:
        def __init__(self, index, similarity_top_k: int = 10, **_: object) -> None:
            self.similarity_top_k = similarity_top_k
            state.dense_top_k.append(similarity_top_k)

        def retrieve(self, query: str):
            state.dense_queries.append(query)
            if state.dense_retrieve_error is not None:
                raise state.dense_retrieve_error
            nodes = []
            for kb_index, score in state.dense_hits[: self.similarity_top_k]:
                article = state.documents[kb_index - 1]
                assert article["index"] == kb_index
                nodes.append(SimpleNamespace(node=SimpleNamespace(metadata=dict(article)), score=score))
            return nodes

    class OpenAI:
        def __init__(self, api_key: str | None = None, **_: object) -> None:
            state.llm_api_keys.append(api_key)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, model: str, messages: list[dict], **_: object):
            state.llm_calls.append({"model": model, "messages": messages})
            if state.llm_error is not None:
                raise state.llm_error
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=state.llm_answer))])

    def module(name: str, **attrs: object) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__path__ = []  # behave like a package for dotted imports
        mod.__dict__.update(attrs)
        return mod

    return {
        "llama_index": module("llama_index"),
        "llama_index.core": module(
            "llama_index.core", Settings=Settings, StorageContext=StorageContext, VectorStoreIndex=VectorStoreIndex
        ),
        "llama_index.core.retrievers": module("llama_index.core.retrievers", VectorIndexRetriever=VectorIndexRetriever),
        "llama_index.vector_stores": module("llama_index.vector_stores"),
        "llama_index.vector_stores.qdrant": module("llama_index.vector_stores.qdrant", QdrantVectorStore=QdrantVectorStore),
        "llama_index.embeddings": module("llama_index.embeddings"),
        "llama_index.embeddings.huggingface": module(
            "llama_index.embeddings.huggingface", HuggingFaceEmbedding=HuggingFaceEmbedding
        ),
        "qdrant_client": module("qdrant_client", QdrantClient=QdrantClient),
        "openai": module("openai", OpenAI=OpenAI),
    }
