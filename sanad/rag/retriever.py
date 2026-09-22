from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler


# ---------- CONFIG ----------
QDRANT_URL = "http://localhost:6333"
COLLECTION = "saudi_labor_law"
EMBED_MODEL = "intfloat/multilingual-e5-base"
TOP_K = 5   # number of results to retrieve
ALPHA = 0.6 # weight for semantic scores in hybrid fusion


# ---------- Embedding Model ----------
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
Settings.embed_model = embed_model

# ---------- Qdrant Setup ----------
qdrant = QdrantClient(url=QDRANT_URL)
store = QdrantVectorStore(client=qdrant, collection_name=COLLECTION)
vector_store = QdrantVectorStore(client=qdrant, collection_name=COLLECTION)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    embed_model=embed_model
)

retriever = VectorIndexRetriever(
    index=index,
    vector_store=store,
    embed_model=embed_model,
    similarity_top_k=3
)


# ---------- Hybrid Retriever ----------
class HybridRetriever:
    """
    Combines dense (semantic) and lexical (BM25) retrieval into a hybrid retriever.
    Returns structured results compatible with chat backend.
    """

    def __init__(self, documents, alpha=ALPHA, dense_retriever=retriever):
        """
        Args:
            documents (list[dict]): Parsed labor law articles with metadata.
            alpha (float): Weight for semantic vs lexical scores.
            dense_retriever: A VectorIndexRetriever instance.
        """
        self.docs = documents
        self.alpha = alpha
        self.dense = dense_retriever
        self.bm25 = BM25Okapi([d["arabic_content"].split() for d in documents])

    def retrieve(self, query, top_k=TOP_K):
        """
        Perform hybrid retrieval using BM25 and dense similarity.
        Returns list of dicts: { index, score, content, metadata }.
        """
        # ---------- BM25 Retrieval ----------
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_scores = MinMaxScaler().fit_transform(bm25_scores.reshape(-1, 1)).flatten()

        # ---------- Dense Retrieval ----------
        dense_results = self.dense.retrieve(query)
        dense_scores = np.zeros(len(self.docs))

        for r in dense_results:
            idx = (r.node.metadata.get("index") or 0) - 1  # stored "index" is 1-based; dense_scores is 0-based
            if idx is not None and 0 <= idx < len(self.docs):
                dense_scores[idx] = r.score

        dense_scores = MinMaxScaler().fit_transform(dense_scores.reshape(-1, 1)).flatten()

        # ---------- Hybrid Fusion ----------
        hybrid_scores = self.alpha * dense_scores + (1 - self.alpha) * bm25_scores

        # ---------- Top-K Selection ----------
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        # ---------- Build Structured Results ----------
        results = []
        for i in top_indices:
            doc = self.docs[i]
            results.append({
                "index": i,
                "score": float(hybrid_scores[i]),
                "content": doc.get("arabic_content", ""),
                "metadata": doc
            })

        return results
