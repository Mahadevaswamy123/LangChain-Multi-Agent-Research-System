"""
Production FastAPI application entry point.

Features:
- Lifespan context manager for startup/shutdown lifecycle
- CORS middleware for cross-origin requests
- Redis-backed sliding-window rate limiting middleware
- Structured request logging middleware
- Global exception handlers
- Prometheus metrics endpoint at /metrics
- Health check endpoint at /health
- Router registration for research, auth, health APIs
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import nltk
import redis.asyncio as aioredis
import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import settings
from app.core.exceptions import AgentException, register_exception_handlers
from app.middleware.logging_middleware import LoggingMiddleware
from app.middleware.rate_limit import RateLimitMiddleware

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level connection handles (populated during startup)
# ---------------------------------------------------------------------------
_redis_client: aioredis.Redis | None = None
_db_engine = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup → yield → shutdown."""
    global _redis_client, _db_engine

    # ── Startup ────────────────────────────────────────────────────────────
    logger.info("application_startup", version=settings.APP_VERSION)

    # Download required NLTK corpora (no-op if already present)
    for corpus in ("punkt", "stopwords", "averaged_perceptron_tagger"):
        try:
            nltk.download(corpus, quiet=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("nltk_download_failed", corpus=corpus, error=str(exc))

    # Initialise Redis connection pool
    try:
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
        await _redis_client.ping()
        logger.info("redis_connected", url=settings.REDIS_URL)
    except Exception as exc:  # pragma: no cover
        logger.error("redis_connection_failed", error=str(exc))

    # Initialise PostgreSQL async engine (creates pool)
    try:
        _db_engine = create_async_engine(
            settings.DATABASE_URL,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=settings.DEBUG,
        )
        # Verify connectivity
        from sqlalchemy import text
        async with _db_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("postgres_connected")
    except Exception as exc:  # pragma: no cover
        logger.error("postgres_connection_failed", error=str(exc))

    # Store handles in app state so routes can access them
    app.state.redis = _redis_client
    app.state.db_engine = _db_engine

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("application_shutdown")

    if _redis_client:
        await _redis_client.aclose()
        logger.info("redis_disconnected")

    if _db_engine:
        await _db_engine.dispose()
        logger.info("postgres_disconnected")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multi-Agent Research System",
    description=(
        "Production-grade agentic AI research platform powered by LangChain, "
        "Qdrant, Redis, and PostgreSQL."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ---------------------------------------------------------------------------
# Middleware  (order matters – outer-most first)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production via env var
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)

# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

register_exception_handlers(app)


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    from datetime import datetime, timezone
    return JSONResponse(
        status_code=404,
        content={
            "error": "NotFound",
            "detail": f"The path '{request.url.path}' does not exist.",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from app.api.routes import auth, health, research  # noqa: E402  (avoid circular)

app.include_router(research.router, prefix="/api/v1/research", tags=["Research"])
app.include_router(auth.router,     prefix="/api/v1/auth",     tags=["Auth"])
app.include_router(health.router,                               tags=["Health"])

# ---------------------------------------------------------------------------
# Prometheus metrics – mounted as a sub-app at /metrics
# ---------------------------------------------------------------------------

try:
    from prometheus_client import make_asgi_app
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
except Exception:
    pass
