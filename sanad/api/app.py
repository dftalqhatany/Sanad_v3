"""Thin HTTP API (Phase 7).

    HTTP request  ->  edge validation (file type, size, count)
                  ->  SanadRequest
                  ->  SanadOrchestrator.handle()          <- the only Sanad call the API makes
                  ->  OrchestratorResult as JSON

The API contains no routing, analysis, retrieval, parsing or extraction logic, and never imports an
agent, the RAG, a parser or an LLM client.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    HTTP_STATUS,
    ApiConfig,
    ApiErrorResponse,
    ApiHealth,
    ApiValidationError,
    AskBody,
    UploadLimits,
    build_request,
    build_salary_query,
    safe_filename,
)
from config import AnalysisSettings, ApiSettings, DocumentProcessingSettings, SalarySettings, SanadSettings
from models.orchestration import DocumentRole, SanadRequest, TaskHint
from orchestrator import SanadOrchestrator

logger = logging.getLogger(f"sanad.{__name__}")  # one "sanad" logging namespace, as before the flattening
COMPARISON_PRIORITIES = ("basic_salary", "stated_pay", "compliance", "cv_compatibility", "weekly_working_hours",
                         "daily_working_hours", "working_days_per_week", "annual_leave", "probation_period")


def create_app(orchestrator: SanadOrchestrator | None = None, *, settings: SanadSettings | None = None,
               api_settings: ApiSettings | None = None, document_settings: DocumentProcessingSettings | None = None,
               analysis_settings: AnalysisSettings | None = None,
               salary_settings: SalarySettings | None = None) -> FastAPI:
    settings = settings or SanadSettings.from_env()
    api_settings = api_settings or ApiSettings.from_env()
    document_settings = document_settings or DocumentProcessingSettings.from_env()
    orchestrator = orchestrator or SanadOrchestrator.from_settings(settings, analysis_settings, document_settings)

    app = FastAPI(title="Sanad API", version="0.1.0",
                  description="Single entry point for Sanad: uploads and questions are routed by the Orchestrator.")
    app.state.orchestrator = orchestrator
    app.state.limits = UploadLimits(api=api_settings, documents=document_settings)
    app.state.answer_generation = bool(settings.openai_api_key)
    app.state.salary_benchmarking = salary_settings.is_configured if salary_settings else SalarySettings.from_env().is_configured
    if api_settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(api_settings.cors_origins), allow_methods=["*"],
                           allow_headers=["*"])
    app.include_router(router)

    @app.exception_handler(ApiValidationError)
    async def _rejected(request: Request, exc: ApiValidationError) -> JSONResponse:  # noqa: ARG001
        logger.info("Rejected request: %s (%s)", exc.code, exc.status_code)
        return JSONResponse(status_code=exc.status_code,
                            content=ApiErrorResponse.model_validate(
                                {"error": {"code": exc.code, "message": exc.message, "detail": exc.detail}}
                            ).model_dump())

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
        logger.error("Unhandled API error: %s", type(exc).__name__)
        return JSONResponse(status_code=500, content={"error": {"code": "internal_error",
                                                                "message": "The request could not be processed.",
                                                                "detail": type(exc).__name__}})

    return app


router = APIRouter(prefix="/api")


@router.get("/health", response_model=ApiHealth)
def health() -> ApiHealth:
    """Liveness only: no agent, RAG or document work is done here."""
    return ApiHealth()


@router.get("/config", response_model=ApiConfig)
def config(request: Request) -> ApiConfig:
    limits: UploadLimits = request.app.state.limits
    orchestrator: SanadOrchestrator = request.app.state.orchestrator
    return ApiConfig(
        allowed_file_types=list(limits.api.allowed_suffixes),
        max_file_size_mb=round(limits.max_file_bytes / (1024 * 1024), 2),
        max_total_size_mb=round(limits.api.max_total_bytes / (1024 * 1024), 2),
        max_files=limits.api.max_files,
        max_contracts=orchestrator.comparison_agent.max_contracts,
        tasks=[task.value for task in TaskHint],
        document_roles=[role.value for role in DocumentRole],
        comparison_priorities=list(COMPARISON_PRIORITIES),
        answer_generation_enabled=request.app.state.answer_generation,
        salary_benchmarking_enabled=request.app.state.salary_benchmarking,
    )


@router.post("/analyze")
async def analyze(
    request: Request,
    files: list[UploadFile] = File(default_factory=list),
    roles: list[str] = Form(default_factory=list),
    labels: list[str] = Form(default_factory=list),
    question: str | None = Form(default=None),
    task: str = Form(default=TaskHint.AUTO.value),
    priorities: list[str] = Form(default_factory=list),
    target_job: str | None = Form(default=None),
    salary_job_title: str | None = Form(default=None),
    salary_location: str | None = Form(default=None),
    years_experience: str | None = Form(default=None),
    seniority: str | None = Form(default=None),
) -> JSONResponse:
    """Uploaded PDF/DOCX files (+ optional question and hints) -> the Orchestrator's structured result."""
    limits: UploadLimits = request.app.state.limits
    uploads = [(safe_filename(item.filename, position), await item.read())
               for position, item in enumerate(files, 1)]
    salary_query = build_salary_query(salary_job_title, salary_location, years_experience, seniority)
    sanad_request = build_request(uploads, roles=roles, labels=labels, question=question, task=task,
                                  priorities=priorities, target_job=target_job, limits=limits,
                                  salary_query=salary_query)
    return _handled(request, sanad_request)


@router.post("/ask")
def ask(request: Request, body: AskBody) -> JSONResponse:
    """A regulatory question with no documents; the Orchestrator sends it to the existing RAG adapter."""
    return _handled(request, SanadRequest(question=body.question))


def _handled(request: Request, sanad_request: SanadRequest) -> JSONResponse:
    orchestrator: SanadOrchestrator = request.app.state.orchestrator
    result = orchestrator.handle(sanad_request)
    logger.info("API %s -> route=%s status=%s", request.url.path, result.routing.route.value, result.status.value)
    return JSONResponse(status_code=HTTP_STATUS[result.status], content=result.model_dump(mode="json"))
