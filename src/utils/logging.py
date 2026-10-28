"""
Structured logging setup using structlog.

Usage
-----
::

    from src.utils.logging import get_logger, bind_context

    logger = get_logger(__name__)
    bind_context(user_id="u_123", session_id="sess_abc")
    logger.info("processing_started", topic="AI safety")

The logger is configured once at import time based on ``LOG_LEVEL`` and
``DEBUG`` from application settings.  In production (``DEBUG=False``) JSON
output is emitted on stdout; in development a colourful human-readable
format is used instead.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    unbind_contextvars,
)

__all__ = [
    "get_logger",
    "bind_context",
    "unbind_context",
    "clear_context",
    "setup_logging",
]


def setup_logging(log_level: str = "INFO", debug: bool = False) -> None:
    """
    Configure structlog and the stdlib ``logging`` root logger.

    Call this once at application startup (e.g. in ``main.py``).

    Parameters
    ----------
    log_level:
        One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.
    debug:
        When ``True`` a colourful human-readable console renderer is used.
        When ``False`` (production) JSON is emitted.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if debug:
        renderer: Any = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(log_level.upper()),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Return a structlog ``BoundLogger`` for *name*.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
    """
    return structlog.get_logger(name)  # type: ignore[return-value]


def bind_context(**kwargs: Any) -> None:
    """
    Bind key-value pairs to the current context-var context.

    All subsequent log calls within the same async task / thread will
    automatically include these fields.
    """
    bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """Remove specific keys from the current context-var context."""
    unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear **all** keys from the current context-var context."""
    clear_contextvars()


def _auto_configure() -> None:
    try:
        from app.core.config import settings  # noqa: PLC0415

        setup_logging(log_level=settings.LOG_LEVEL, debug=settings.DEBUG)
    except Exception:  # pragma: no cover
        setup_logging(log_level="INFO", debug=False)


_auto_configure()
