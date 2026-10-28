"""
Reader agent that scrapes URLs for deep content.

Scrapes the top-N most relevant URLs returned by the SearchAgent,
running all HTTP requests concurrently via asyncio.gather so the
pipeline isn't bottlenecked by sequential fetches.
"""
from __future__ import annotations

import asyncio
from typing import List

from langchain_openai import ChatOpenAI

from src.agents.base_agent import BaseAgent
from src.tools.scraper_tools import scrape_url_tool
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ReaderAgent(BaseAgent):
    """
    Scrapes the top-N URLs from search results and extracts clean content.

    URLs are sorted by relevance score (highest first) and scraped
    concurrently.  Failed or empty scrapes are silently dropped so the
    pipeline always receives at least the successfully scraped pages.
    """

    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        super().__init__(llm=llm, name="ReaderAgent")

    async def run(  # type: ignore[override]
        self,
        search_results: List[dict],
        max_urls: int = 3,
    ) -> List[dict]:
        """
        Scrape the top *max_urls* URLs from *search_results*.

        Args:
            search_results: List of result dicts from SearchAgent
                            (must contain 'url' and optionally 'score').
            max_urls:       Maximum number of URLs to scrape (default 3).

        Returns:
            List of dicts with keys:
                url, title, content, word_count, success
            Only successfully scraped pages (success=True) are returned.
        """
        self.logger.info(
            f"[ReaderAgent] scraping top {max_urls} URLs from "
            f"{len(search_results)} search results"
        )

        if not search_results:
            self.logger.warning("[ReaderAgent] no search results to scrape")
            return []

        # Sort by score and take the top N with valid URLs
        sorted_results = sorted(
            [r for r in search_results if r.get("url")],
            key=lambda r: float(r.get("score", 0.0)),
            reverse=True,
        )
        target_urls = [r["url"] for r in sorted_results[:max_urls]]

        self.logger.info(f"[ReaderAgent] target URLs: {target_urls}")

        # Scrape concurrently
        tasks = [scrape_url_tool.ainvoke({"url": url}) for url in target_urls]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        scraped: List[dict] = []
        for url, result in zip(target_urls, raw_results):
            if isinstance(result, Exception):
                self.logger.error(
                    f"[ReaderAgent] scrape exception for {url}: {result}",
                    exc_info=False,
                )
                continue
            if isinstance(result, dict) and result.get("success"):
                scraped.append(result)
            else:
                self.logger.warning(
                    f"[ReaderAgent] scrape unsuccessful for {url}: "
                    f"{result.get('content', '') if isinstance(result, dict) else result}"
                )

        self.logger.info(
            f"[ReaderAgent] successfully scraped {len(scraped)}/{len(target_urls)} URLs"
        )
        return scraped
