"""
Async Redis cache utility with JSON serialization and connection pooling.

Usage
-----
::

    from src.utils.cache import RedisCache

    cache = RedisCache()          # uses REDIS_URL from settings

    await cache.set("key", {"data": 42}, ttl_seconds=120)
    value = await cache.get("key")          # {"data": 42}
    await cache.delete("key")

    # Cache-aside (fetch-or-compute)
    result = await cache.get_or_set(
        "heavy_key",
        factory_fn=some_async_function,
        ttl_seconds=300,
    )

    # Bulk invalidation
    await cache.invalidate_prefix("research:")
"""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, Optional

import redis.asyncio as aioredis
from redis.asyncio.connection import ConnectionPool

from app.core.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_SENTINEL = object()  # used to distinguish cache miss from cached None


class RedisCache:
    """
    Async Redis cache with automatic JSON serialization/deserialization.

    A single ``ConnectionPool`` is shared across all instances that use the
    same *url*, so it is safe to create multiple ``RedisCache`` objects.
    """

    _pools: dict[str, ConnectionPool] = {}

    def __init__(
        self,
        url: str | None = None,
        key_prefix: str = "",
        default_ttl: int = 300,
    ) -> None:
        self._url = url or settings.REDIS_URL
        self._key_prefix = key_prefix
        self._default_ttl = default_ttl
        self._client: aioredis.Redis | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _full_key(self, key: str) -> str:
        return f"{self._key_prefix}{key}" if self._key_prefix else key

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            if self._url not in self.__class__._pools:
                self.__class__._pools[self._url] = ConnectionPool.from_url(
                    self._url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=20,
                )
            self._client = aioredis.Redis(
                connection_pool=self.__class__._pools[self._url]
            )
        return self._client

    @staticmethod
    def _serialize(value: Any) -> str:
        return json.dumps(value, default=str)

    @staticmethod
    def _deserialize(raw: str) -> Any:
        return json.loads(raw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieve *key* from the cache.

        Returns
        -------
        Any
            The cached value, or ``None`` if the key does not exist.
        """
        client = await self._get_client()
        try:
            raw = await client.get(self._full_key(key))
            if raw is None:
                return None
            return self._deserialize(raw)
        except Exception as exc:
            logger.warning("cache_get_error", key=key, error=str(exc))
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> bool:
        """
        Store *value* under *key* with an optional TTL.

        Parameters
        ----------
        key:
            Cache key (without prefix – the prefix is added internally).
        value:
            Any JSON-serializable value.
        ttl_seconds:
            Expiry in seconds.  Defaults to ``self._default_ttl``.

        Returns
        -------
        bool
            ``True`` on success, ``False`` if the operation failed.
        """
        client = await self._get_client()
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        try:
            await client.set(self._full_key(key), self._serialize(value), ex=ttl)
            return True
        except Exception as exc:
            logger.warning("cache_set_error", key=key, error=str(exc))
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete *key* from the cache.

        Returns
        -------
        bool
            ``True`` if the key existed and was deleted, ``False`` otherwise.
        """
        client = await self._get_client()
        try:
            deleted = await client.delete(self._full_key(key))
            return bool(deleted)
        except Exception as exc:
            logger.warning("cache_delete_error", key=key, error=str(exc))
            return False

    async def get_or_set(
        self,
        key: str,
        factory_fn: Callable[[], Awaitable[Any]],
        ttl_seconds: int | None = None,
    ) -> Any:
        """
        Cache-aside pattern: return the cached value if it exists, otherwise
        call *factory_fn*, cache its result, and return it.

        Parameters
        ----------
        key:
            Cache key.
        factory_fn:
            Async callable (no arguments) that produces the value to cache.
        ttl_seconds:
            TTL for the cached entry.

        Returns
        -------
        Any
            The cached or freshly computed value.
        """
        cached = await self.get(key)
        if cached is not None:
            logger.debug("cache_hit", key=key)
            return cached

        logger.debug("cache_miss", key=key)
        value = await factory_fn()
        await self.set(key, value, ttl_seconds=ttl_seconds)
        return value

    async def invalidate_prefix(self, prefix: str) -> int:
        """
        Delete **all** keys whose Redis name starts with *prefix*
        (after applying ``self._key_prefix``).

        Uses ``SCAN`` to avoid blocking the server.

        Returns
        -------
        int
            Number of keys deleted.
        """
        client = await self._get_client()
        full_prefix = self._full_key(prefix)
        pattern = f"{full_prefix}*"
        deleted = 0
        try:
            async for key in client.scan_iter(match=pattern, count=100):
                await client.delete(key)
                deleted += 1
            logger.info("cache_prefix_invalidated", prefix=prefix, deleted=deleted)
        except Exception as exc:
            logger.warning("cache_invalidate_prefix_error", prefix=prefix, error=str(exc))
        return deleted

    async def close(self) -> None:
        """Close the underlying Redis client connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# Module-level singleton for convenience
# ---------------------------------------------------------------------------
cache: RedisCache = RedisCache()

