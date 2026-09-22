"""Run the API locally:  python -m api   (host/port from SANAD_API_HOST / SANAD_API_PORT)."""

from __future__ import annotations

import logging

from api.app import create_app
from config import ApiSettings


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = ApiSettings.from_env()
    uvicorn.run(create_app(), host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
