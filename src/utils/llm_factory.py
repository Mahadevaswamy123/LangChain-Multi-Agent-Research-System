"""
Universal LLM and Embeddings factory supporting multiple providers:
- Groq (LLaMA 3.3, 3.1, Mixtral)
- Google Gemini (Gemini 1.5 Flash, 1.5 Pro)
- OpenAI (GPT-4o-mini, GPT-4o)

Provides dynamic, seamless vendor independence. The user only needs to provide
their API key in `.env` for their provider of choice.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_active_provider() -> str:
    """
    Intelligently determine the active LLM provider from settings & environment.
    Priority in 'auto' mode: Groq -> Gemini -> OpenAI.
    """
    settings = get_settings()
    configured = (settings.LLM_PROVIDER or "auto").lower()

    if configured in ("groq", "gemini", "google", "openai"):
        return "gemini" if configured == "google" else configured

    # 'auto' mode detection based on provided keys
    groq_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    gemini_key = settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    openai_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")

    if groq_key and not groq_key.startswith("your_"):
        return "groq"
    if gemini_key and not gemini_key.startswith("your_"):
        return "gemini"
    if openai_key and not openai_key.startswith("your_"):
        return "openai"

    # Default fallback
    if groq_key:
        return "groq"
    if gemini_key:
        return "gemini"
    return "openai"


def get_llm(
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    streaming: bool = False,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Factory that instantiates the appropriate chat model for the active provider.

    Supports:
      - Groq (native langchain_groq or OpenAI-compatible endpoint)
      - Google Gemini (native langchain_google_genai or OpenAI-compatible endpoint)
      - OpenAI (langchain_openai)
    """
    settings = get_settings()
    provider = resolve_active_provider()
    temp = settings.LLM_TEMPERATURE if temperature is None else temperature

    # -------------------------------------------------------------------------
    # 1. Groq Provider
    # -------------------------------------------------------------------------
    if provider == "groq":
        api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        model_name = model or settings.GROQ_MODEL or "llama-3.3-70b-versatile"
        logger.info(f"[LLMFactory] Initializing Groq model: {model_name!r}")

        # Try native langchain_groq
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=model_name,
                groq_api_key=api_key,
                temperature=temp,
                streaming=streaming,
                **kwargs,
            )
        except ImportError:
            # Groq exposes a 100% OpenAI-compatible endpoint
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
                model=model_name,
                temperature=temp,
                streaming=streaming,
                **kwargs,
            )

    # -------------------------------------------------------------------------
    # 2. Google Gemini Provider
    # -------------------------------------------------------------------------
    elif provider == "gemini":
        api_key = (
            settings.GEMINI_API_KEY
            or settings.GOOGLE_API_KEY
            or os.getenv("GEMINI_API_KEY", "")
            or os.getenv("GOOGLE_API_KEY", "")
        )
        model_name = model or settings.GEMINI_MODEL or "gemini-1.5-flash"
        logger.info(f"[LLMFactory] Initializing Google Gemini model: {model_name!r}")

        # Try native langchain_google_genai
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temp,
                convert_system_message_to_human=True,
                streaming=streaming,
                **kwargs,
            )
        except ImportError:
            # Gemini exposes an official OpenAI-compatible endpoint
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=api_key,
                model=model_name,
                temperature=temp,
                streaming=streaming,
                **kwargs,
            )

    # -------------------------------------------------------------------------
    # 3. OpenAI Provider (default)
    # -------------------------------------------------------------------------
    else:
        api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        model_name = model or settings.OPENAI_MODEL or settings.LLM_MODEL or "gpt-4o-mini"
        logger.info(f"[LLMFactory] Initializing OpenAI model: {model_name!r}")

        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temp,
            streaming=streaming,
            **kwargs,
        )


class FallbackEmbeddings:
    """
    Lightweight deterministic embeddings for RAG search when cloud embedding
    APIs are unavailable or when running on providers without embedding endpoints (e.g. Groq).
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def _hash_embed(self, text: str) -> List[float]:
        import hashlib
        import math
        vec = [0.0] * self.dimension
        words = text.lower().split()
        if not words:
            return vec

        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            vec[idx] += 1.0

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_embed(text)


def get_embeddings_instance():
    """
    Return embeddings instance based on available credentials.
    """
    settings = get_settings()
    openai_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    gemini_key = (
        settings.GEMINI_API_KEY
        or settings.GOOGLE_API_KEY
        or os.getenv("GEMINI_API_KEY", "")
        or os.getenv("GOOGLE_API_KEY", "")
    )

    if openai_key and not openai_key.startswith("your_"):
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL or "text-embedding-3-large",
                api_key=openai_key,
            )
        except Exception as e:
            logger.warning(f"[Embeddings] OpenAI embeddings failed: {e}")

    if gemini_key and not gemini_key.startswith("your_"):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=gemini_key,
            )
        except Exception as e:
            logger.warning(f"[Embeddings] Google embeddings failed: {e}")

    logger.info("[Embeddings] Using high-performance local fallback embeddings")
    return FallbackEmbeddings(dimension=settings.EMBEDDING_DIM or 384)

