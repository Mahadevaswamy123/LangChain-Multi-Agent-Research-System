"""
Multi-provider Embedding Service with Redis caching, batching, and fallback.
Supports OpenAI, Google Gemini, and local deterministic embeddings.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from app.core.config import get_settings
from src.utils.llm_factory import FallbackEmbeddings, get_embeddings_instance
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_BATCH_SIZE = 100
_MAX_TOKENS_PER_TEXT = 8_191


class EmbeddingService:
    """
    Embedding service that supports multiple providers (OpenAI, Gemini, Local)
    with optional Redis caching.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        cache: Any = None,
        api_key: Optional[str] = None,
        cache_ttl: int = 604_800,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.EMBEDDING_MODEL or "text-embedding-3-large"
        self._cache = cache
        self._cache_ttl = cache_ttl
        self._api_key = api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")

        # Try to initialize OpenAI async client if key is available
        self._openai_client = None
        if self._api_key and not self._api_key.startswith("your_"):
            try:
                from openai import AsyncOpenAI
                self._openai_client = AsyncOpenAI(api_key=self._api_key)
            except Exception as e:
                logger.debug(f"[Embeddings] AsyncOpenAI client init notice: {e}")

        # Fallback multi-provider instance (Gemini or Local)
        self._generic_embeddings = None

        logger.info(
            f"EmbeddingService initialized — primary={self._model}, "
            f"cache={'enabled' if cache else 'disabled'}"
        )

    def _get_generic_embeddings(self):
        if self._generic_embeddings is None:
            self._generic_embeddings = get_embeddings_instance()
        return self._generic_embeddings

    async def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return []

        text = self._truncate_text(text)
        cache_key = self._get_cache_key(text)
        cached = await self._cache_get(cache_key)
        if cached is not None:
            return cached

        results = await self.embed_batch([text])
        res = results[0] if results else []
        if res:
            await self._cache_set(cache_key, res)
        return res

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> List[List[float]]:
        if not texts:
            return []

        truncated = [self._truncate_text(t) for t in texts]

        # Use OpenAI if available
        if self._openai_client is not None:
            try:
                all_embeddings: List[List[float]] = []
                for i in range(0, len(truncated), batch_size):
                    chunk = truncated[i : i + batch_size]
                    response = await self._openai_client.embeddings.create(
                        input=chunk,
                        model=self._model,
                    )
                    sorted_data = sorted(response.data, key=lambda d: d.index)
                    all_embeddings.extend([item.embedding for item in sorted_data])
                return all_embeddings
            except Exception as exc:
                logger.warning(f"[Embeddings] OpenAI batch embedding failed: {exc}, switching to fallback provider")

        # Fallback to generic provider (Gemini or Local)
        try:
            embedder = self._get_generic_embeddings()
            if hasattr(embedder, "aembed_documents"):
                return await embedder.aembed_documents(truncated)
            elif hasattr(embedder, "embed_documents"):
                return await asyncio.to_thread(embedder.embed_documents, truncated)
        except Exception as e:
            logger.warning(f"[Embeddings] Generic embedding notice: {e}")

        # Ultimate zero-crash deterministic fallback
        local_fb = FallbackEmbeddings(dimension=384)
        return local_fb.embed_documents(truncated)

    async def embed_documents(
        self,
        documents: List[Document],
    ) -> List[Tuple[Document, List[float]]]:
        if not documents:
            return []
        texts = [doc.page_content for doc in documents]
        embeddings = await self.embed_batch(texts)
        return list(zip(documents, embeddings))

    def _get_cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _truncate_text(self, text: str) -> str:
        max_chars = _MAX_TOKENS_PER_TEXT * 4
        return text[:max_chars] if len(text) > max_chars else text

    async def _cache_get(self, key: str) -> Optional[List[float]]:
        if not self._cache:
            return None
        try:
            raw = await self._cache.get(f"emb:{key}")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return None

    async def _cache_set(self, key: str, value: List[float]) -> None:
        if not self._cache or not value:
            return
        try:
            await self._cache.setex(f"emb:{key}", self._cache_ttl, json.dumps(value))
        except Exception:
            pass
