"""Phase 3 boundaries: the document layers are independent of the regulatory RAG and its vector database."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENT_LAYER_FILES = [
    *sorted((PROJECT_ROOT / "sanad" / "parsers").rglob("*.py")),
    *sorted((PROJECT_ROOT / "sanad" / "extraction").rglob("*.py")),
    PROJECT_ROOT / "sanad" / "models" / "documents.py",
    PROJECT_ROOT / "sanad" / "models" / "extraction.py",
]
FORBIDDEN = ("sanad.rag", "chatbot_backend", "hybird_search", "llama_index", "qdrant_client", "openai",
             "PIL", "pytesseract", "easyocr", "paddleocr", "cv2")  # the MVP has no image uploads and no OCR


def test_document_layers_do_not_import_the_rag_or_vector_database():
    offenders = []
    for path in DOCUMENT_LAYER_FILES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            offenders += [f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> {name}" for name in names
                          if any(name == f or name.startswith(f + ".") for f in FORBIDDEN)]
    assert len(DOCUMENT_LAYER_FILES) >= 16
    assert not offenders, offenders


def test_parsing_and_extracting_loads_no_rag_module():
    code = (
        "import sys\n"
        "sys.modules['PIL'] = None  # Pillow must not be needed\n"
        "from sanad.config import DocumentProcessingSettings\n"
        "from sanad.parsers import DocumentProcessor\n"
        "from sanad.extraction import extract_contract, extract_cv\n"
        "from tests.fixtures.documents.builders import build_all\n"
        "processor = DocumentProcessor(DocumentProcessingSettings())\n"
        "for name, data in build_all().items():\n"
        "    document = processor.parse_bytes(data, name)\n"
        "    extract_cv(document) if 'cv' in name else extract_contract(document)\n"
        "roots = ('chatbot_backend', 'hybird_search', 'llama_index', 'qdrant_client', 'openai', 'pytesseract')\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in roots or m.startswith('sanad.rag')]\n"
        "assert not loaded, loaded\n"
        "print('clean')\n"
    )
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip() == "clean"


def test_mvp_has_no_standalone_image_parser_ocr_engine_or_image_dependencies():
    parsers = PROJECT_ROOT / "sanad" / "parsers"
    assert not (parsers / "image_parser.py").exists() and not (parsers / "ocr.py").exists()
    for name in ("requirements.txt", "pyproject.toml"):
        path = PROJECT_ROOT / name
        if path.exists():
            text = path.read_text(encoding="utf-8").lower()
            for dependency in ("pillow", "pytesseract", "easyocr", "paddleocr", "opencv"):
                assert dependency not in text, (name, dependency)
