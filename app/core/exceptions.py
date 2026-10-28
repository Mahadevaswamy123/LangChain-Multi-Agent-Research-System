"""
Custom exception hierarchy and FastAPI exception handlers.

All handlers return a standardised JSON body::

    {
        "error":     "<exception class name>",
        "detail":    "<human-readable message>",
        "timestamp": "<ISO-8601 UTC timestamp>"
    }
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------

class AgentException(Exception):
    """Base exception for all application-level errors."""

    def __init__(self, message: str = "An unexpected agent error occurred", status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(message={self.message!r}, status_code={self.status_code})"


# ---------------------------------------------------------------------------
# Derived exceptions
# ---------------------------------------------------------------------------

class RateLimitException(AgentException):
    """Raised when a client exceeds the configured rate limit."""

    def __init__(self, message: str = "Rate limit exceeded. Please slow down.") -> None:
        super().__init__(message=message, status_code=429)


class VectorStoreException(AgentException):
    """Raised when an operation against the vector store fails."""

    def __init__(self, message: str = "Vector store operation failed.") -> None:
        super().__init__(message=message, status_code=503)


class EvaluationException(AgentException):
    """Raised when the RAGAS evaluation pipeline encounters an error."""

    def __init__(self, message: str = "Evaluation pipeline failed.") -> None:
        super().__init__(message=message, status_code=500)


class InvalidTopicException(AgentException):
    """Raised when the supplied research topic is invalid or empty."""

    def __init__(self, message: str = "The provided research topic is invalid.") -> None:
        super().__init__(message=message, status_code=422)


class PipelineTimeoutException(AgentException):
    """Raised when a research pipeline run exceeds its time budget."""

    def __init__(self, message: str = "The research pipeline timed out.") -> None:
        super().__init__(message=message, status_code=504)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _error_body(exc_name: str, detail: str) -> dict[str, str]:
    return {
        "error": exc_name,
        "detail": detail,
        "timestamp": _utc_now_iso(),
    }


# ---------------------------------------------------------------------------
# FastAPI exception handlers
# ---------------------------------------------------------------------------

def _agent_exception_handler(request: Request, exc: AgentException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(exc.__class__.__name__, exc.message),
    )


def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=_error_body("InternalServerError", "An unexpected internal error occurred."),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all custom exception handlers to *app*."""
    app.add_exception_handler(AgentException, _agent_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RateLimitException, _agent_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(VectorStoreException, _agent_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(EvaluationException, _agent_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(InvalidTopicException, _agent_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(PipelineTimeoutException, _agent_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unhandled_exception_handler)  # type: ignore[arg-type]

