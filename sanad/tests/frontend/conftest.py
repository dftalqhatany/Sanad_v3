"""The frontend tests reuse the API fixtures: the UI is exercised against the real API app."""

from tests.api.test_api_salary import salary_client  # noqa: F401
from tests.api.conftest import (  # noqa: F401
    client,
    parsed,
    processor,
    real_orchestrator,
    sample_bytes,
    sample_paths,
    stub,
    stub_client,
)
