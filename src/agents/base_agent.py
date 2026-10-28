"""
Abstract base agent with common functionality.

All concrete agents inherit from BaseAgent and must implement `run()`.
Uses the dynamic multi-provider LLM factory to support Groq, Gemini, and OpenAI seamlessly.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from src.utils.llm_factory import get_llm
from src.utils.logging import get_logger
from src.utils.retry import retry_async


class BaseAgent(ABC):
    """
    Abstract base for every research pipeline agent.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        name: str = "BaseAgent",
    ) -> None:
        self.name = name
        self.logger = get_logger(name)
        self.llm: BaseChatModel = llm or get_llm()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def run(self, **kwargs: Any) -> Any:
        """
        Main agent execution. Must be implemented by subclasses.
        """
        ...

    # ------------------------------------------------------------------
    # Monitoring wrapper
    # ------------------------------------------------------------------

    async def _run_with_monitoring(self, **kwargs: Any) -> Any:
        """
        Wrap :meth:`run` with timing, structured logging, and error handling.
        """
        start = time.perf_counter()
        self.logger.info(f"[{self.name}] starting")
        try:
            result = await self.run(**kwargs)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            self.logger.info(
                f"[{self.name}] completed",
                extra={"agent": self.name, "duration_ms": elapsed_ms},
            )
            return result
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            self.logger.error(
                f"[{self.name}] failed: {exc}",
                extra={"agent": self.name, "duration_ms": elapsed_ms, "error": str(exc)},
                exc_info=True,
            )
            raise
