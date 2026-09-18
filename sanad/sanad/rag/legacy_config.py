"""Read the existing RAG's configuration *statically* (AST), without importing it.

Importing hybird_search.py connects to Qdrant and loads the embedding model, so the constants
are parsed from the source instead. This keeps hybird_search.py the single source of truth for
the Qdrant URL, collection, embedding model, top-k and fusion weight.
"""

from __future__ import annotations

import ast
from pathlib import Path

from sanad.models.regulatory import RetrievalConfig

_REQUIRED_CONSTANTS = ("QDRANT_URL", "COLLECTION", "EMBED_MODEL", "TOP_K", "ALPHA")


class LegacyConfigError(RuntimeError):
    pass


def _parse(path: Path) -> ast.Module:
    try:
        return ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise LegacyConfigError(f"Cannot read the existing RAG configuration from {path}: {exc}") from exc


def read_legacy_retrieval_config(path: Path) -> RetrievalConfig:
    tree = _parse(path)
    constants: dict[str, object] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in _REQUIRED_CONSTANTS
        ):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    missing = [name for name in _REQUIRED_CONSTANTS if name not in constants]
    if missing:
        raise LegacyConfigError(f"{path} no longer defines literal constants: {', '.join(missing)}")

    dense_top_k = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "VectorIndexRetriever":
            for keyword in node.keywords:
                if keyword.arg == "similarity_top_k":
                    try:
                        dense_top_k = int(ast.literal_eval(keyword.value))
                    except (ValueError, TypeError):
                        dense_top_k = None
    return RetrievalConfig(
        qdrant_url=str(constants["QDRANT_URL"]),
        collection=str(constants["COLLECTION"]),
        embedding_model=str(constants["EMBED_MODEL"]),
        top_k=int(constants["TOP_K"]),
        alpha=float(constants["ALPHA"]),
        dense_similarity_top_k=dense_top_k,
        source_file=str(path),
    )


def read_legacy_answer_model(path: Path) -> str | None:
    """Model name passed to chat.completions.create(...) in chatbot_backend.py (informational)."""
    try:
        tree = _parse(path)
    except LegacyConfigError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "create":
            for keyword in node.keywords:
                if keyword.arg == "model":
                    try:
                        return str(ast.literal_eval(keyword.value))
                    except ValueError:
                        return None
    return None
