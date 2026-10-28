"""
Async URL scraper with a 3-strategy fallback chain.

Strategy priority:
  1. trafilatura  – best extraction for article / blog content
  2. readability-lxml – good for news and generic pages
  3. BeautifulSoup – plain-text fallback for everything else

Uses httpx for fully-async HTTP requests with a 15-second timeout.
Returns a structured dict so callers can inspect success/failure without
parsing free-form strings.
"""
from __future__ import annotations

import re
from typing import Optional

import httpx
from langchain.tools import tool

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TIMEOUT = 15.0          # seconds
_MIN_CONTENT_LEN = 200   # minimum chars to consider extraction successful
_MAX_CONTENT_CHARS = 8000

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.google.com/",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    """Collapse whitespace and strip leading/trailing space."""
    return re.sub(r"\s+", " ", text).strip()


def _extract_trafilatura(html: str) -> Optional[str]:
    """Strategy 1: trafilatura (best for articles/blogs)."""
    try:
        import trafilatura  # type: ignore

        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        if extracted and len(extracted.strip()) >= _MIN_CONTENT_LEN:
            return _clean(extracted)
    except Exception as exc:
        logger.debug(f"trafilatura failed: {exc}")
    return None


def _extract_readability(html: str) -> Optional[str]:
    """Strategy 2: readability-lxml."""
    try:
        from readability import Document  # type: ignore
        from bs4 import BeautifulSoup    # type: ignore

        doc = Document(html)
        clean_html = doc.summary()
        soup = BeautifulSoup(clean_html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        if text and len(text.strip()) >= _MIN_CONTENT_LEN:
            return _clean(text)
    except Exception as exc:
        logger.debug(f"readability failed: {exc}")
    return None


def _extract_bs4(html: str) -> Optional[str]:
    """Strategy 3: plain BeautifulSoup fallback."""
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        if text:
            return _clean(text)
    except Exception as exc:
        logger.debug(f"bs4 fallback failed: {exc}")
    return None


def _get_title(html: str) -> str:
    """Extract <title> text from HTML."""
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "lxml")
        tag = soup.find("title")
        return tag.get_text(strip=True) if tag else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------


@tool
async def scrape_url_tool(url: str) -> dict:
    """
    Async URL scraper with a 3-strategy fallback chain.

    Strategies tried in order:
      1. trafilatura  (best for articles)
      2. readability-lxml
      3. BeautifulSoup (plain-text fallback)

    Args:
        url: The web URL to scrape.

    Returns:
        Dict with keys:
            url        (str)  – the requested URL
            title      (str)  – page title (may be empty)
            content    (str)  – extracted text (up to 8 000 chars)
            word_count (int)  – number of words in content
            success    (bool) – False if all strategies failed
    """
    logger.info(f"[scrape_url_tool] scraping {url}")

    result_base = {"url": url, "title": "", "content": "", "word_count": 0, "success": False}

    try:
        async with httpx.AsyncClient(
            headers=_HEADERS,
            timeout=_TIMEOUT,
            follow_redirects=True,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text

    except httpx.TimeoutException:
        logger.warning(f"[scrape_url_tool] timeout: {url}")
        return {**result_base, "content": "Request timed out."}
    except httpx.HTTPStatusError as exc:
        logger.warning(f"[scrape_url_tool] HTTP {exc.response.status_code}: {url}")
        return {**result_base, "content": f"HTTP error {exc.response.status_code}."}
    except Exception as exc:
        logger.error(f"[scrape_url_tool] fetch error: {exc}", exc_info=True)
        return {**result_base, "content": f"Fetch error: {exc}"}

    title = _get_title(html)

    # Try each strategy in order
    content: Optional[str] = (
        _extract_trafilatura(html)
        or _extract_readability(html)
        or _extract_bs4(html)
    )

    if not content:
        logger.warning(f"[scrape_url_tool] all strategies failed for {url}")
        return {**result_base, "title": title, "content": "Could not extract content."}

    content = content[:_MAX_CONTENT_CHARS]
    word_count = len(content.split())
    logger.info(f"[scrape_url_tool] extracted {word_count} words from {url}")

    return {
        "url": url,
        "title": title,
        "content": content,
        "word_count": word_count,
        "success": True,
    }
