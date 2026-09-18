"""Integration layer between Sanad agents and the existing hr_assistant RAG."""

from sanad.rag.adapter import RegulatoryRAGAdapter
from sanad.rag.errors import RagErrorCode

__all__ = ["RagErrorCode", "RegulatoryRAGAdapter"]
