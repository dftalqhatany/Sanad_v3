"""Sanad: multi-agent HR/legal assistant built around the existing hr_assistant RAG.

Importing this package is cheap and has no side effects: the legacy RAG (Qdrant
connection, embedding model) is only loaded when an adapter call needs it.
"""

__version__ = "0.1.0"
