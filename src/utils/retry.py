"""
Retry decorators and a circuit breaker for resilient async operations.

Exports
-------
- ``retry_async``      – decorator factory for configurable exponential-backoff retries.
- ``with_timeout``     – decorator factory that wraps a coroutine with ``asyncio.wait_for``.
- ``CircuitBreaker``   – class implementing the three-state circuit-breaker pattern.

Usage
-----
::

    from src.utils.retry import retry_async, with_timeout, CircuitBreaker

    @retry_async(max_retries=3, backoff_factor=2, exceptions=(httpx.HTTPError,))
    async def call_external_api() -> dict: ...

    @with_timeout(seconds=10)
    async def slow_operation() -> str: ...

    cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

    async def protected() -> Any:
        async with cb:
            return await call_external_api()
"""
from __future__ import annotations

import asyncio
import enum
import functools
import time
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from src.utils.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])


# ---------------------------------------------------------------------------
# retry_async
# ---------------------------------------------------------------------------

def retry_async(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator factory that retries the decorated coroutine on failure.

    Parameters
    ----------
    max_retries:
        Maximum number of *additional* attempts after the first failure.
    backoff_factor:
        Multiplier applied to *initial_delay* on each subsequent attempt.
    initial_delay:
        Seconds to wait before the first retry.
    exceptions:
        Tuple of exception types that should trigger a retry.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        logger.error(
                            "retry_exhausted",
                            function=func.__qualname__,
                            attempts=attempt + 1,
                            error=str(exc),
                        )
                        raise
                    logger.warning(
                        "retry_attempt",
                        function=func.__qualname__,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay_seconds=delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# with_timeout
# ---------------------------------------------------------------------------

def with_timeout(seconds: float = 30.0) -> Callable[[F], F]:
    """
    Decorator factory that enforces a wall-clock timeout on an async function.

    Raises ``asyncio.TimeoutError`` if the coroutine does not complete in time.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(
                    "timeout_exceeded",
                    function=func.__qualname__,
                    timeout_seconds=seconds,
                )
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class _CircuitState(enum.Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """
    Async-compatible circuit breaker (CLOSED / OPEN / HALF_OPEN).

    Usage
    -----
    ::

        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

        async with cb:
            result = await risky_call()
    """

    class CircuitBreakerOpenError(Exception):
        """Raised when a request is rejected because the circuit is OPEN."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 1,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold

        self._state = _CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        return self._state.value

    async def __aenter__(self) -> "CircuitBreaker":
        async with self._lock:
            self._check_state()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        async with self._lock:
            if exc_type is None:
                await self._on_success()
            elif exc_type is not self.CircuitBreakerOpenError:
                await self._on_failure()
        return False

    def _check_state(self) -> None:
        if self._state == _CircuitState.OPEN:
            elapsed = time.monotonic() - (self._opened_at or 0.0)
            if elapsed >= self._recovery_timeout:
                logger.info("circuit_breaker_half_open")
                self._state = _CircuitState.HALF_OPEN
                self._success_count = 0
            else:
                raise self.CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN – retry in "
                    f"{self._recovery_timeout - elapsed:.1f}s"
                )

    async def _on_success(self) -> None:
        if self._state == _CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._success_threshold:
                logger.info("circuit_breaker_closed")
                self._state = _CircuitState.CLOSED
                self._failure_count = 0
        elif self._state == _CircuitState.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)

    async def _on_failure(self) -> None:
        self._failure_count += 1
        if self._failure_count >= self._failure_threshold or self._state == _CircuitState.HALF_OPEN:
            logger.warning(
                "circuit_breaker_opened",
                failures=self._failure_count,
                threshold=self._failure_threshold,
            )
            self._state = _CircuitState.OPEN
            self._opened_at = time.monotonic()
            self._failure_count = 0
