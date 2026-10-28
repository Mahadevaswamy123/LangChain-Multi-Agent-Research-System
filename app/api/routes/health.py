"""
Health-check endpoints.

GET /health          – Lightweight liveness probe (always fast)
GET /health/detailed – Deep readiness check (Qdrant, Redis, Postgres, OpenAI)
"""
from __future__ import annotations

from datetime import datetime, timezone

import httpx
import redis.asyncio as aioredis
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter()


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# GET /health  – liveness probe
# ---------------------------------------------------------------------------

@router.get("/health", tags=["Health"], summary="Liveness probe")
async def health_liveness() -> dict:
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "timestamp": _now_iso(),
    }


# ---------------------------------------------------------------------------
# GET /health/detailed  – readiness probe
# ---------------------------------------------------------------------------

@router.get("/health/detailed", tags=["Health"], summary="Detailed readiness probe")
async def health_detailed(request: Request) -> JSONResponse:
    components: dict[str, str] = {}
    overall = "healthy"

    # ── Redis ──────────────────────────────────────────────────────────────
    try:
        redis: aioredis.Redis = request.app.state.redis
        await redis.ping()
        components["redis"] = "ok"
    except Exception as exc:
        components["redis"] = f"error: {exc}"
        overall = "degraded"

    # ── PostgreSQL ─────────────────────────────────────────────────────────
    try:
        engine = request.app.state.db_engine
        if engine is not None:
            from sqlalchemy import text
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            components["postgres"] = "ok"
        else:
            components["postgres"] = "not initialised"
            overall = "degraded"
    except Exception as exc:
        components["postgres"] = f"error: {exc}"
        overall = "degraded"

    # ── Qdrant ─────────────────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.QDRANT_URL}/healthz")
        if resp.status_code == 200:
            components["qdrant"] = "ok"
        else:
            components["qdrant"] = f"http_{resp.status_code}"
            overall = "degraded"
    except Exception as exc:
        components["qdrant"] = f"error: {exc}"
        overall = "degraded"

    # ── OpenAI API ─────────────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
            )
        if resp.status_code == 200:
            components["openai"] = "ok"
        else:
            components["openai"] = f"http_{resp.status_code}"
            if overall == "healthy":
                overall = "degraded"
    except Exception as exc:
        components["openai"] = f"error: {exc}"
        if overall == "healthy":
            overall = "degraded"

    http_status = 200 if overall == "healthy" else 503

    return JSONResponse(
        status_code=http_status,
        content={
            "status": overall,
            "version": settings.APP_VERSION,
            "timestamp": _now_iso(),
            "components": components,
        },
    )
