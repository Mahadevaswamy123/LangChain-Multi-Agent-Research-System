"""
Redis-based sliding-window rate-limiter Starlette middleware.

Algorithm
---------
For each request we maintain a Redis sorted-set keyed by
``ratelimit:<identifier>``.  Members are request timestamps (as floats);
the score equals the timestamp so we can remove stale entries with
``ZREMRANGEBYSCORE`` and count remaining entries atomically via a Lua script.

Identifiers used (both checked; the *stricter* one wins):
- ``user:<user_id>`` – extracted from a ``X-User-ID`` header when present.
- ``ip:<client_ip>``  – the ASGI ``client`` address or ``X-Forwarded-For``.
"""
from __future__ import annotations

import time
from typing import Awaitable, Callable

import redis.asyncio as aioredis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.core.config import settings

# ---------------------------------------------------------------------------
# Lua script – atomic sliding-window check-and-record
# ---------------------------------------------------------------------------
_LUA_SLIDING_WINDOW = """
local key        = KEYS[1]
local now        = tonumber(ARGV[1])
local window     = tonumber(ARGV[2])
local limit      = tonumber(ARGV[3])
local window_start = now - window

redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)
local count = redis.call('ZCARD', key)
if count < limit then
    redis.call('ZADD', key, now, now)
    redis.call('EXPIRE', key, window + 1)
    return {0, count + 1}
else
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry_after = math.ceil((oldest[2] + window) - now)
    return {1, retry_after}
end
"""


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter backed by Redis.

    Settings are pulled from ``app.core.config.settings``:

    * ``RATE_LIMIT_REQUESTS`` – maximum requests allowed per window.
    * ``RATE_LIMIT_WINDOW``   – window duration in seconds.
    * ``REDIS_URL``           – Redis connection string.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._limit: int = settings.RATE_LIMIT_REQUESTS
        self._window: int = settings.RATE_LIMIT_WINDOW
        self._redis: aioredis.Redis | None = None  # lazy init

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = await aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    def _get_identifier(self, request: Request) -> str:
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return f"user:{user_id}"
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        return f"ip:{ip}"

    async def _check_rate_limit(self, identifier: str) -> tuple[bool, int]:
        """
        Returns ``(is_limited, retry_after_seconds)``.
        ``retry_after_seconds`` is only meaningful when ``is_limited`` is True.
        """
        redis = await self._get_redis()
        key = f"ratelimit:{identifier}"
        now = time.time()

        result: list[int] = await redis.eval(  # type: ignore[misc]
            _LUA_SLIDING_WINDOW,
            1,  # number of KEYS
            key,
            str(now),
            str(self._window),
            str(self._limit),
        )
        is_limited = bool(result[0])
        retry_after = int(result[1]) if is_limited else 0
        return is_limited, retry_after

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Health/metrics endpoints are exempt
        if request.url.path in {"/health", "/metrics", "/docs", "/openapi.json"}:
            return await call_next(request)

        identifier = self._get_identifier(request)
        is_limited, retry_after = await self._check_rate_limit(identifier)

        if is_limited:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "RateLimitExceeded",
                    "detail": "Too many requests. Please slow down.",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)

