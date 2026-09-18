"""Thin HTTP API over SanadOrchestrator.handle() (Phase 7). Requires fastapi; see requirements.txt."""

from sanad.api.app import create_app
from sanad.api.schemas import ApiValidationError

__all__ = ["ApiValidationError", "create_app"]
