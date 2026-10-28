"""
Cross-encoder reranking using the Cohere Rerank API.

Falls back to TF-IDF cosine similarity reranking when Cohere is
unavailable (no API key or package not installed).
"""
from __future__ import annotations

import math
import os
from collections import Counter
from typing import Any, List, Optional, Tuple

from langchain_core.documents import Document

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "rerank-english-v3.0"
_MAX_DOCS_PER_REQUEST = 1000  # Cohere API limit


# ---------------------------------------------------------------------------
# DocumentReranker
# ---------------------------------------------------------------------------

class DocumentReranker:
    """
    Cross-encoder reranker for improving retrieval precision.

    Primary backend: Cohere Rerank API (production-grade cross-encoder
    that jointly attends to query + document).

    Fallback backend: TF-IDF cosine similarity (no external dependency).

    Args:
        api_key: Cohere API key. Falls back to ``COHERE_API_KEY`` env var.
        model:   Cohere rerank model name.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv("COHERE_API_KEY", "")
        self._cohere_client: Any = None

        if self._api_key:
            try:
                import cohere  # type: ignore
                self._cohere_client = cohere.AsyncClientV2(self._api_key)
                logger.info(
                    f"DocumentReranker initialised — backend=Cohere, model={model}"
                )
            except ImportError:
                logger.warning(
                    "cohere package not installed — using TF-IDF fallback. "
                    "Install with: pip install cohere"
                )
        else:
            logger.warning(
                "COHERE_API_KEY not set — DocumentReranker will use TF-IDF fallback. "
                "Set COHERE_API_KEY for production-grade reranking."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents by relevance to *query*.

        Tries the Cohere API first; silently falls back to TF-IDF if
        Cohere is unavailable or raises an error.

        Args:
            query:     Search query.
            documents: Candidate documents to rerank (up to 1000).
            top_n:     Number of top documents to return.

        Returns:
            List of ``(Document, relevance_score)`` tuples, sorted
            descending, length ≤ min(top_n, len(documents)).
        """
        if not documents:
            return []

        top_n = min(top_n, len(documents))

        if not query or not query.strip():
            logger.warning("rerank called with empty query — returning top_n by order")
            return [(doc, 1.0 / (i + 1)) for i, doc in enumerate(documents[:top_n])]

        if self._cohere_client is not None:
            try:
                return await self._cohere_rerank(query, documents, top_n)
            except Exception as exc:
                logger.warning(
                    f"Cohere rerank failed ({exc}), falling back to TF-IDF"
                )

        return self._fallback_rerank(query, documents, top_n)

    # ------------------------------------------------------------------
    # Cohere backend
    # ------------------------------------------------------------------

    async def _cohere_rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int,
    ) -> List[Tuple[Document, float]]:
        """
        Call the Cohere Rerank v2 API.

        The API accepts plain text strings for each document; we pass
        the ``page_content`` field and preserve the original Document
        metadata in the returned list.
        """
        # Cohere has a limit of 1000 docs per request
        docs_to_rank = documents[:_MAX_DOCS_PER_REQUEST]
        doc_texts = [doc.page_content for doc in docs_to_rank]

        logger.debug(
            f"Cohere rerank: query='{query[:60]}', "
            f"docs={len(doc_texts)}, top_n={top_n}"
        )

        response = await self._cohere_client.rerank(
            model=self._model,
            query=query,
            documents=doc_texts,
            top_n=top_n,
            return_documents=False,  # we already have them
        )

        results: List[Tuple[Document, float]] = []
        for result in response.results:
            doc = docs_to_rank[result.index]
            score = result.relevance_score
            results.append((doc, float(score)))

        # Sort descending (Cohere returns them sorted but let's be explicit)
        results.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            f"Cohere rerank complete — returned {len(results)} result(s)"
        )
        return results

    # ------------------------------------------------------------------
    # Fallback TF-IDF backend
    # ------------------------------------------------------------------

    def _fallback_rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int,
    ) -> List[Tuple[Document, float]]:
        """
        Fallback reranker using TF-IDF cosine similarity.

        This avoids any external dependency and provides a reasonable
        proxy for cross-encoder reranking when Cohere is unavailable.

        Args:
            query:     Search query.
            documents: Candidate documents.
            top_n:     Number to return.

        Returns:
            ``(Document, cosine_similarity)`` tuples sorted descending.
        """
        logger.debug("Using TF-IDF fallback reranker")

        corpus = [doc.page_content for doc in documents]
        all_texts = [query] + corpus

        # Build vocabulary
        vocab: dict[str, int] = {}
        tokenised: List[List[str]] = []
        for text in all_texts:
            tokens = self._tokenise(text)
            tokenised.append(tokens)
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)

        if not vocab:
            # All texts are empty
            return [(doc, 0.0) for doc in documents[:top_n]]

        n_docs = len(corpus)
        # IDF: log((N+1) / (df+1)) + 1
        doc_freq: Counter = Counter()
        for tokens in tokenised[1:]:  # skip query
            doc_freq.update(set(tokens))

        idf: dict[str, float] = {
            term: math.log((n_docs + 1) / (freq + 1)) + 1.0
            for term, freq in doc_freq.items()
        }
        # Terms not in any doc get max idf
        default_idf = math.log((n_docs + 1) / 1.0) + 1.0

        def tfidf_vector(tokens: List[str]) -> dict[str, float]:
            tf: Counter = Counter(tokens)
            total = len(tokens) or 1
            return {
                term: (count / total) * idf.get(term, default_idf)
                for term, count in tf.items()
            }

        def cosine_sim(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
            keys = set(vec_a) & set(vec_b)
            dot = sum(vec_a[k] * vec_b[k] for k in keys)
            mag_a = math.sqrt(sum(v * v for v in vec_a.values())) or 1e-9
            mag_b = math.sqrt(sum(v * v for v in vec_b.values())) or 1e-9
            return dot / (mag_a * mag_b)

        query_vec = tfidf_vector(tokenised[0])
        scored: List[Tuple[Document, float]] = []
        for doc, tokens in zip(documents, tokenised[1:]):
            doc_vec = tfidf_vector(tokens)
            score = cosine_sim(query_vec, doc_vec)
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            f"TF-IDF fallback rerank complete — returned {min(top_n, len(scored))} result(s)"
        )
        return scored[:top_n]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tokenise(self, text: str) -> List[str]:
        """Simple lowercase word tokeniser."""
        import re
        return re.findall(r"\b[a-z0-9]+\b", text.lower())
