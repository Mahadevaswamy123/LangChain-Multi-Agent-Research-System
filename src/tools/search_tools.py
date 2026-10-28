"""
Production search tools using Tavily's async client.

Two @tool-decorated coroutines are provided:
  - web_search_tool   – general web search
  - news_search_tool  – news-specific search

Both return structured lists of result dicts rather than raw strings, so
the pipeline can work with individual fields (URL, score, etc.) directly.
"""
from __future__ import annotations

import os
from typing import Any, List

from langchain.tools import tool
from tavily import AsyncTavilyClient

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared async Tavily client (created lazily to pick up env vars after load)
# ---------------------------------------------------------------------------

_client: AsyncTavilyClient | None = None


def _get_client() -> AsyncTavilyClient:
    global _client
    if _client is None:
        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "TAVILY_API_KEY is not set. Add it to your .env file."
            )
        _client = AsyncTavilyClient(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
async def web_search_tool(query: str, max_results: int = 10) -> List[dict]:
    """
    Search the web for recent information using Tavily.

    Args:
        query:       The search query string.
        max_results: Maximum number of results to return (default 10).

    Returns:
        List of result dicts with keys:
            title, url, snippet, score, published_date
    """
    logger.info(f"[web_search_tool] query='{query}' max_results={max_results}")
    try:
        client = _get_client()
        response = await client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=False,
            include_raw_content=False,
        )
        results: List[dict] = []
        for r in response.get("results", []):
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")[:500],
                    "score": float(r.get("score", 0.0)),
                    "published_date": r.get("published_date", ""),
                }
            )
        logger.info(f"[web_search_tool] returned {len(results)} results")
        return results
    except Exception as exc:
        logger.error(f"[web_search_tool] failed: {exc}", exc_info=True)
        return []


@tool
async def news_search_tool(query: str, max_results: int = 5) -> List[dict]:
    """
    Search recent news articles using Tavily's news topic filter.

    Args:
        query:       The search query string.
        max_results: Maximum number of news results to return (default 5).

    Returns:
        List of result dicts with keys:
            title, url, snippet, score, published_date
    """
    logger.info(f"[news_search_tool] query='{query}' max_results={max_results}")
    try:
        client = _get_client()
        response = await client.search(
            query=query,
            max_results=max_results,
            topic="news",
            search_depth="basic",
            include_answer=False,
            include_raw_content=False,
        )
        results: List[dict] = []
        for r in response.get("results", []):
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")[:500],
                    "score": float(r.get("score", 0.0)),
                    "published_date": r.get("published_date", ""),
                }
            )
        logger.info(f"[news_search_tool] returned {len(results)} results")
        return results
    except Exception as exc:
        logger.error(f"[news_search_tool] failed: {exc}", exc_info=True)
        return []
