"""Lazy, explicit loading of the RAG backend module (rag.backend).

Why this still exists after the RAG moved in-tree: importing rag.backend connects to Qdrant and
loads the embedding model at import time, so nothing may import it eagerly. It is imported only on
first use, under a lock, and the module is checked for the callables the adapter relies on.
Failures are re-raised (never swallowed) and are not cached, so a later call can succeed once
Qdrant is up.

The sys.path and os.chdir juggling this module used to need is gone: rag.backend now imports
rag.retriever normally and locates its knowledge base relative to its own file.
"""

from __future__ import annotations

import importlib
import sys
import threading
from pathlib import Path
from types import ModuleType

from rag.errors import LegacyInterfaceError, LegacyRagNotFoundError

PACKAGE_DIR = Path(__file__).resolve().parent  # rag.backend always resolves from here
REQUIRED_FILES = ("backend.py", "retriever.py")
REQUIRED_CALLABLES = ("answer_policy_question", "get_retriever", "detect_language")


class RagLoader:
    def __init__(self, rag_dir: Path, module_name: str = "rag.backend") -> None:
        self.rag_dir = Path(rag_dir).expanduser().resolve()
        self.module_name = module_name
        self._module: ModuleType | None = None
        self._lock = threading.RLock()
        self.last_error: BaseException | None = None

    @property
    def is_loaded(self) -> bool:
        return self._module is not None

    def load(self) -> ModuleType:
        module = self._module
        if module is not None:
            return module
        with self._lock:
            if self._module is not None:
                return self._module
            try:
                self._check_files()
                module = sys.modules.get(self.module_name)
                if module is None:
                    importlib.invalidate_caches()
                    module = importlib.import_module(self.module_name)
                self._verify(module)
            except BaseException as exc:
                self.last_error = exc
                raise
            self._module = module
            self.last_error = None
            return module

    def _check_files(self) -> None:
        if not self.rag_dir.is_dir():
            raise LegacyRagNotFoundError(f"RAG directory not found: {self.rag_dir}")
        missing = [name for name in REQUIRED_FILES if not (self.rag_dir / name).is_file()]
        if missing:
            raise LegacyRagNotFoundError(f"RAG files missing in {self.rag_dir}: {', '.join(missing)}")

    def _verify(self, module: ModuleType) -> None:
        origin = Path(getattr(module, "__file__", "") or "").resolve()
        if origin.parent != PACKAGE_DIR:
            raise LegacyRagNotFoundError(
                f"A module named '{self.module_name}' is already loaded from {origin}, not from {PACKAGE_DIR}"
            )
        missing = [name for name in REQUIRED_CALLABLES if not callable(getattr(module, name, None))]
        if missing:
            raise LegacyInterfaceError(f"{origin} no longer defines: {', '.join(missing)}")
