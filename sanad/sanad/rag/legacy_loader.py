"""Lazy, explicit loading of the existing RAG backend module.

Why this exists (all constraints come from the unmodified legacy code):
  * chatbot_backend.py uses flat imports (`from hybird_search import ...`), so its directory must be
    on sys.path while importing;
  * it opens "data/labor_law/labor_law_parsed.json" relative to the *current working directory*;
  * importing it connects to Qdrant and loads the embedding model.

The module is therefore imported only on first use, under a lock, with the legacy directory
temporarily on sys.path and as the working directory. Failures are re-raised (never swallowed)
and are not cached, so a later call can succeed once Qdrant is up.
Note: os.chdir is process-wide; it is held only for the duration of the one-time import.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import threading
from pathlib import Path
from types import ModuleType

from sanad.rag.errors import LegacyInterfaceError, LegacyRagNotFoundError

REQUIRED_FILES = ("chatbot_backend.py", "hybird_search.py")
REQUIRED_CALLABLES = ("answer_policy_question", "get_retriever", "detect_language")


class LegacyRagLoader:
    def __init__(self, legacy_dir: Path, module_name: str = "chatbot_backend") -> None:
        self.legacy_dir = Path(legacy_dir).expanduser().resolve()
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
                    module = self._import_in_legacy_context()
                self._verify(module)
            except BaseException as exc:
                self.last_error = exc
                raise
            self._module = module
            self.last_error = None
            return module

    def _check_files(self) -> None:
        if not self.legacy_dir.is_dir():
            raise LegacyRagNotFoundError(f"Existing RAG directory not found: {self.legacy_dir}")
        missing = [name for name in REQUIRED_FILES if not (self.legacy_dir / name).is_file()]
        if missing:
            raise LegacyRagNotFoundError(f"Existing RAG files missing in {self.legacy_dir}: {', '.join(missing)}")

    def _import_in_legacy_context(self) -> ModuleType:
        legacy = str(self.legacy_dir)
        added_to_path = legacy not in sys.path
        if added_to_path:
            sys.path.insert(0, legacy)
        previous_cwd = os.getcwd()
        os.chdir(legacy)
        try:
            importlib.invalidate_caches()
            return importlib.import_module(self.module_name)
        finally:
            os.chdir(previous_cwd)
            if added_to_path:
                with contextlib.suppress(ValueError):
                    sys.path.remove(legacy)

    def _verify(self, module: ModuleType) -> None:
        origin = Path(getattr(module, "__file__", "") or "").resolve()
        if origin.parent != self.legacy_dir:
            raise LegacyRagNotFoundError(
                f"A module named '{self.module_name}' is already loaded from {origin}, not from {self.legacy_dir}"
            )
        missing = [name for name in REQUIRED_CALLABLES if not callable(getattr(module, name, None))]
        if missing:
            raise LegacyInterfaceError(f"{origin} no longer defines: {', '.join(missing)}")
