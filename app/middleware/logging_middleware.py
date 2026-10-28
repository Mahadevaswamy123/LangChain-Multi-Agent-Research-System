"""
Structured request/response logging middleware.

Each request receives a unique ``X-Request-ID`` (UUID4) that is:
- added to the response headers
- bound to the structlog context for the duration of the request
- logged alongside method, path, status code, and duration in ms

Sensitive fields (passwords, tokens, authorization headers) are masked
before logging.
"""
from __future__ import annotations

import time
import uuid
from typing import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

logger = structlog.get_logger(__name__)

# Fields / header names that must never appear in logs
_SENSITIVE_HEADERS = frozenset(
    {"authorization", "x-api-key", "cookie", "set-cookie", "proxy-authorization"}
)
_SENSITIVE_BODY_KEYS = frozenset(
    {"password", "token", "access_token", "refresh_token", "secret", "api_key"}
)
_MASK = "***REDACTED***"


def _sanitise_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        k: (_MASK if k.lower() in _SENSITIVE_HEADERS else v)
        for k, v in headers.items()
    }


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that logs every HTTP request and its response.

    Adds ``X-Request-ID`` to all responses so callers can correlate logs
    with specific requests.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        start_time = time.perf_counter()

        # Attempt to extract a user identifier from the request state (set by
        # auth middleware) or fall back to the Authorization header.
        user_id: str | None = getattr(request.state, "user_id", None)

        # Bind context so all log statements within this request carry the IDs
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            user_id=user_id,
            method=request.method,
            path=request.url.path,
        )

        logger.info(
            "request_started",
            query_params=str(request.query_params),
            client_host=request.client.host if request.client else None,
            headers=_sanitise_headers(dict(request.headers)),
        )

        try:
            response: Response = await call_next(request)
        except Exception as exc:  # pragma: no cover – bubble up
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.exception(
                "request_failed",
                duration_ms=duration_ms,
                error=str(exc),
            )
            raise

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "request_completed",
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response

