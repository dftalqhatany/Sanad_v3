"""Centralised Sanad settings.

Environment variables
---------------------
OPENAI_API_KEY         Key passed to the RAG's answer generation (never hard-coded, never logged).
SANAD_RAG_DIR          Location of the RAG package (retriever, backend, knowledge base, qdrant_storage).
                       Defaults to ./rag inside this project.
SANAD_LEGACY_RAG_DIR   Deprecated alias for SANAD_RAG_DIR, from when the RAG lived in a separate
                       hr_assistant project. Still honoured; SANAD_RAG_DIR wins if both are set.

The Qdrant URL, collection name, embedding model, top-k and fusion weight are NOT configured
here: they are read from rag/retriever.py so the retriever stays the single source of truth
(see rag.retrieval_config).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent  # the Sanad project root: this file sits in it
DEFAULT_RAG_DIR = PROJECT_ROOT / "rag"  # retriever, backend, knowledge base and qdrant_storage


@dataclass(frozen=True)
class SanadSettings:
    rag_dir: Path = DEFAULT_RAG_DIR
    openai_api_key: str | None = field(default=None, repr=False)

    # Names of the RAG modules inside rag_dir.
    backend_module: str = "rag.backend"
    backend_file: str = "backend.py"
    retrieval_file: str = "retriever.py"
    knowledge_base_relpath: str = "data/labor_law/labor_law_parsed.json"
    source_document_relpath: str = "data/labor_law/labor_law_ar.pdf"

    # Provenance of the regulatory source, as documented in rag/README.md.
    source_title_ar: str = "نظام العمل"
    source_title_en: str = "Saudi Labor Law"
    source_publisher: str = "Ministry of Human Resources and Social Development (MHRSD)"
    source_url: str | None = (
        "https://www.hrsd.gov.sa/sites/default/files/2025-07/nzam-al-ml----wfq-alhwyt-aljdydt-2.pdf"
    )
    # The Arabic text is the legal reference; English content was machine-translated (Helsinki-NLP/opus-mt-ar-en).
    english_is_machine_translation: bool = True

    health_timeout_s: float = 3.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "rag_dir", Path(self.rag_dir).expanduser().resolve())

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "SanadSettings":
        env = os.environ if env is None else env
        return cls(
            rag_dir=Path(env.get("SANAD_RAG_DIR") or env.get("SANAD_LEGACY_RAG_DIR") or DEFAULT_RAG_DIR),
            openai_api_key=(env.get("OPENAI_API_KEY") or None),
        )

    @property
    def backend_module_path(self) -> Path:
        return self.rag_dir / self.backend_file

    @property
    def retrieval_module_path(self) -> Path:
        return self.rag_dir / self.retrieval_file

    @property
    def knowledge_base_path(self) -> Path:
        return self.rag_dir / self.knowledge_base_relpath

    @property
    def source_document_path(self) -> Path:
        return self.rag_dir / self.source_document_relpath


# --------------------------------------------------------------------------- Phase 3: document processing
@dataclass(frozen=True)
class DocumentProcessingSettings:
    """Limits for uploaded PDF and DOCX documents (parsers never execute files and never write temporary files).

    Environment variables
    ---------------------
    SANAD_MAX_UPLOAD_MB      Maximum accepted file size in megabytes (default 20).
    SANAD_MAX_PDF_PAGES      Maximum number of PDF pages processed (default 100).
    SANAD_UPLOAD_ROOT        When set, parse_path() only reads files located under this directory.
    """

    max_file_size_bytes: int = 20 * 1024 * 1024
    max_pdf_pages: int = 100
    min_page_text_chars: int = 20
    max_docx_entries: int = 2000
    max_docx_uncompressed_bytes: int = 100 * 1024 * 1024
    allowed_roots: tuple[Path, ...] | None = None

    def __post_init__(self) -> None:
        if self.allowed_roots is not None:
            roots = tuple(Path(root).expanduser().resolve() for root in self.allowed_roots)
            object.__setattr__(self, "allowed_roots", roots)
        for name in ("max_file_size_bytes", "max_pdf_pages", "max_docx_entries", "max_docx_uncompressed_bytes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "DocumentProcessingSettings":
        env = os.environ if env is None else env
        kwargs: dict = {}
        if env.get("SANAD_MAX_UPLOAD_MB"):
            kwargs["max_file_size_bytes"] = int(float(env["SANAD_MAX_UPLOAD_MB"]) * 1024 * 1024)
        if env.get("SANAD_MAX_PDF_PAGES"):
            kwargs["max_pdf_pages"] = int(env["SANAD_MAX_PDF_PAGES"])
        if env.get("SANAD_UPLOAD_ROOT"):
            kwargs["allowed_roots"] = (Path(env["SANAD_UPLOAD_ROOT"]),)
        return cls(**kwargs)


# --------------------------------------------------------------------------- Phase 4: analysis agent
@dataclass(frozen=True)
class AnalysisSettings:
    """Settings for the Contract & CV Analysis Agent.

    The OpenAI key is SanadSettings.openai_api_key (OPENAI_API_KEY); without it no LLM interpretation runs and
    regulatory findings stay 'requires_review'.

    Environment variables
    ---------------------
    SANAD_ANALYSIS_LLM_MODEL        Model used for evidence-grounded interpretation (default gpt-4o-mini, the model
                                    the existing RAG already uses).
    SANAD_ANALYSIS_LLM_TIMEOUT_S    Request timeout in seconds (default 60).
    """

    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 60.0
    max_evidence_per_topic: int = 6

    def __post_init__(self) -> None:
        if self.llm_timeout_s <= 0 or self.max_evidence_per_topic <= 0:
            raise ValueError("llm_timeout_s and max_evidence_per_topic must be positive")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AnalysisSettings":
        env = os.environ if env is None else env
        kwargs: dict = {}
        if env.get("SANAD_ANALYSIS_LLM_MODEL"):
            kwargs["llm_model"] = env["SANAD_ANALYSIS_LLM_MODEL"]
        if env.get("SANAD_ANALYSIS_LLM_TIMEOUT_S"):
            kwargs["llm_timeout_s"] = float(env["SANAD_ANALYSIS_LLM_TIMEOUT_S"])
        return cls(**kwargs)


# --------------------------------------------------------------------------- Phase 7: HTTP API
@dataclass(frozen=True)
class ApiSettings:
    """Edge limits of the thin HTTP API (the parsers keep their own per-file limits).

    Environment variables
    ---------------------
    SANAD_API_MAX_FILES      Maximum number of uploaded files per request (default 6: 5 contracts + 1 CV).
    SANAD_API_MAX_TOTAL_MB   Maximum total upload size per request (default 60).
    SANAD_API_CORS_ORIGINS   Comma-separated browser origins allowed to call the API (default the local UI).
    SANAD_API_HOST           Host for `python -m api` (default 127.0.0.1).
    SANAD_API_PORT           Port for `python -m api` (default 8000).
    """

    max_files: int = 6
    max_total_bytes: int = 60 * 1024 * 1024
    allowed_suffixes: tuple[str, ...] = (".pdf", ".docx")
    cors_origins: tuple[str, ...] = ("http://localhost:8501", "http://127.0.0.1:8501")
    host: str = "127.0.0.1"
    port: int = 8000

    def __post_init__(self) -> None:
        if self.max_files <= 0 or self.max_total_bytes <= 0:
            raise ValueError("max_files and max_total_bytes must be positive")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ApiSettings":
        env = os.environ if env is None else env
        kwargs: dict = {}
        if env.get("SANAD_API_MAX_FILES"):
            kwargs["max_files"] = int(env["SANAD_API_MAX_FILES"])
        if env.get("SANAD_API_MAX_TOTAL_MB"):
            kwargs["max_total_bytes"] = int(float(env["SANAD_API_MAX_TOTAL_MB"]) * 1024 * 1024)
        if env.get("SANAD_API_CORS_ORIGINS"):
            kwargs["cors_origins"] = tuple(o.strip() for o in env["SANAD_API_CORS_ORIGINS"].split(",") if o.strip())
        if env.get("SANAD_API_HOST"):
            kwargs["host"] = env["SANAD_API_HOST"]
        if env.get("SANAD_API_PORT"):
            kwargs["port"] = int(env["SANAD_API_PORT"])
        return cls(**kwargs)


# --------------------------------------------------------------------------- Phase 8: salary benchmarking (web search)
SALARY_SEED_SOURCES: tuple[str, ...] = (
    "https://www.stats.gov.sa/en/w/gastat-saudi-workers-monthly-average-wage-in-four-sectors-10.238-sar",
    "https://open.data.gov.sa/en/datasets/view/9801e8a2-2442-48f9-bf84-9666febd3cce/preview/parsed/"
    "Average%20Salary%20per%20Department%202026.json",
    "https://saudisalary.com/investigator-salary",
    "https://saudiarabia.paylab.com/en/salaryinfo",
    "https://www.kaggle.com/datasets/amirmahdiabbootalebi/salary-by-job-title-and-country",
)
SALARY_ALLOWED_DOMAINS: tuple[str, ...] = (
    "stats.gov.sa", "open.data.gov.sa", "saudisalary.com", "paylab.com", "kaggle.com", "linkedin.com", "lnkd.in",
)


@dataclass(frozen=True)
class SalarySettings:
    """Real salary benchmarking over the web. No key is hard-coded and no rate is assumed beyond the SAR/USD peg.

    Environment variables
    ---------------------
    SANAD_SEARCH_PROVIDER     'tavily', 'brave' or 'none' (default 'none': only the seed sources are read).
    SANAD_SEARCH_API_KEY      API key for that provider (also read from TAVILY_API_KEY / BRAVE_API_KEY).
    SANAD_SALARY_ENABLED      '1' to enable benchmarking without a search provider (seed sources only).
    SANAD_SALARY_DOMAINS      Comma-separated allowlist; only these domains are searched, fetched and cited.
    SANAD_SALARY_SEEDS        Comma-separated seed URLs read on every query (defaults to the documented source set).
    SANAD_SALARY_TIMEOUT_S    Per-request timeout in seconds (default 10).
    SANAD_SALARY_MAX_RESULTS  Maximum search results fetched per query (default 8).
    SANAD_SALARY_CACHE_TTL_S  Cache lifetime for a query's results (default 3600; 0 disables the cache).
    SANAD_SALARY_MIN_INTERVAL_S  Minimum delay between outgoing requests (default 0.5).
    SANAD_SALARY_FX_RATES     Explicit rates to SAR, e.g. 'USD:3.75,EUR:4.05'. Currencies without a rate are never
                              converted; they are reported and left out of the range.
    """

    provider: Literal["none", "tavily", "brave"] = "none"
    api_key: str | None = field(default=None, repr=False)
    enabled: bool = False
    allowed_domains: tuple[str, ...] = SALARY_ALLOWED_DOMAINS
    seed_urls: tuple[str, ...] = SALARY_SEED_SOURCES
    timeout_s: float = 10.0
    max_results: int = 8
    cache_ttl_s: float = 3600.0
    min_request_interval_s: float = 0.5
    max_content_chars: int = 200_000
    fx_rates_to_sar: Mapping[str, float] = field(default_factory=lambda: {"USD": 3.75})
    user_agent: str = "Sanad/0.1 (salary benchmarking; +https://github.com/)"

    def __post_init__(self) -> None:
        if self.timeout_s <= 0 or self.max_results <= 0:
            raise ValueError("timeout_s and max_results must be positive")
        if self.cache_ttl_s < 0 or self.min_request_interval_s < 0:
            raise ValueError("cache_ttl_s and min_request_interval_s must not be negative")
        object.__setattr__(self, "fx_rates_to_sar", dict(self.fx_rates_to_sar))

    @property
    def is_configured(self) -> bool:
        """Benchmarking runs when a search provider has a key, or when the seed sources are explicitly enabled."""
        return bool(self.enabled or (self.provider != "none" and self.api_key))

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "SalarySettings":
        env = os.environ if env is None else env
        provider = (env.get("SANAD_SEARCH_PROVIDER") or "none").strip().lower()
        if provider not in ("none", "tavily", "brave"):
            raise ValueError(f"unknown SANAD_SEARCH_PROVIDER '{provider}'")
        key = env.get("SANAD_SEARCH_API_KEY") or env.get(f"{provider.upper()}_API_KEY") or None
        kwargs: dict = {"provider": provider, "api_key": key or None,
                        "enabled": (env.get("SANAD_SALARY_ENABLED") or "").strip() in ("1", "true", "yes")}
        if env.get("SANAD_SALARY_DOMAINS"):
            kwargs["allowed_domains"] = tuple(d.strip().lower() for d in env["SANAD_SALARY_DOMAINS"].split(",") if d.strip())
        if env.get("SANAD_SALARY_SEEDS"):
            kwargs["seed_urls"] = tuple(u.strip() for u in env["SANAD_SALARY_SEEDS"].split(",") if u.strip())
        for name, key_name, cast in (("timeout_s", "SANAD_SALARY_TIMEOUT_S", float),
                                     ("max_results", "SANAD_SALARY_MAX_RESULTS", int),
                                     ("cache_ttl_s", "SANAD_SALARY_CACHE_TTL_S", float),
                                     ("min_request_interval_s", "SANAD_SALARY_MIN_INTERVAL_S", float)):
            if env.get(key_name):
                kwargs[name] = cast(env[key_name])
        if env.get("SANAD_SALARY_FX_RATES"):
            rates = {}
            for pair in env["SANAD_SALARY_FX_RATES"].split(","):
                if ":" in pair:
                    code, rate = pair.split(":", 1)
                    rates[code.strip().upper()] = float(rate)
            kwargs["fx_rates_to_sar"] = rates
        return cls(**kwargs)
