"""
Production hybrid search combining dense + BM25 sparse retrieval with
Reciprocal Rank Fusion (RRF) to produce better-calibrated rankings than
either approach alone.

Pipeline per query:
  1. (Optional) LLM-powered query expansion → 3 alternative phrasings
  2. BM25 keyword search over the in-memory corpus
  3. Dense semantic search via Qdrant
  4. RRF score fusion
  5. Return top-k (Document, score) pairs
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from src.utils.llm_factory import get_llm

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy imports to avoid circular dependency
# (EmbeddingService and QdrantVectorStore imported inside __init__)


# ---------------------------------------------------------------------------
# HybridSearchEngine
# ---------------------------------------------------------------------------

class HybridSearchEngine:
    """
    Hybrid retrieval combining BM25 lexical search and dense semantic
    search via RRF fusion.

    Args:
        vector_store:       :class:`~src.rag.vector_store.QdrantVectorStore`
        embedding_service:  :class:`~src.rag.embeddings.EmbeddingService`
        bm25_weight:        Weight for BM25 component (0–1).
        dense_weight:       Weight for dense component (0–1).
        llm_model:          OpenAI model used for query expansion.
    """

    def __init__(
        self,
        vector_store: Any,
        embedding_service: Any,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._bm25_weight = bm25_weight
        self._dense_weight = dense_weight

        self._bm25_corpus: List[Document] = []
        self._bm25_index: Optional[BM25Okapi] = None

        self._llm = get_llm(temperature=0.2)

        logger.info(
            f"HybridSearchEngine ready — "
            f"bm25_weight={bm25_weight}, dense_weight={dense_weight}"
        )

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_bm25_index(self, documents: List[Document]) -> None:
        """
        Build (or rebuild) the in-memory BM25 index from a list of Documents.

        Call this after upserting documents into Qdrant so the two indexes
        stay in sync.

        Args:
            documents: Documents to index (same set as what's in Qdrant).
        """
        if not documents:
            logger.warning("build_bm25_index called with empty list — index cleared")
            self._bm25_corpus = []
            self._bm25_index = None
            return

        self._bm25_corpus = documents
        tokenised = [self._tokenise(doc.page_content) for doc in documents]
        self._bm25_index = BM25Okapi(tokenised)

        logger.info(f"BM25 index built with {len(documents)} document(s)")

    def add_to_bm25_index(self, documents: List[Document]) -> None:
        """
        Incrementally add documents to the BM25 index.
        (Rebuilds the index internally — BM25Okapi is not truly incremental.)
        """
        combined = self._bm25_corpus + documents
        self.build_bm25_index(combined)

    # ------------------------------------------------------------------
    # Main search entry-point
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        k: int = 10,
        filter_user_id: Optional[str] = None,
        expand_query: bool = True,
    ) -> List[Tuple[Document, float]]:
        """
        Full hybrid search pipeline.

        Args:
            query:          Natural-language query string.
            k:              Maximum number of results to return.
            filter_user_id: Restrict results to this user/tenant.
            expand_query:   Whether to expand the query with LLM variants.

        Returns:
            List of ``(Document, fused_score)`` tuples, sorted descending.
        """
        if not query or not query.strip():
            logger.warning("search called with empty query")
            return []

        logger.info(f"Hybrid search: query='{query[:80]}…', k={k}")

        # 1. Query expansion
        queries: List[str] = [query]
        if expand_query:
            try:
                expanded = await self._expand_query(query)
                queries.extend(expanded)
                logger.debug(f"Expanded to {len(queries)} query variant(s)")
            except Exception as exc:
                logger.warning(f"Query expansion failed (non-fatal): {exc}")

        # 2. Run BM25 and dense search concurrently for each query variant
        bm25_results_per_query: List[List[Tuple[Document, float]]] = []
        dense_tasks = []

        for q in queries:
            bm25_results_per_query.append(
                self._bm25_search(q, k=k * 2)
            )

        # Embed all query variants concurrently
        embeddings = await self._embedding_service.embed_batch(queries)

        dense_results_per_query: List[List[Tuple[Document, float]]] = []
        for q_emb, q in zip(embeddings, queries):
            if not q_emb:
                dense_results_per_query.append([])
                continue
            try:
                dense_docs = await self._vector_store.dense_search(
                    query_embedding=q_emb,
                    limit=k * 2,
                    score_threshold=0.3,
                    filter_user_id=filter_user_id,
                )
                dense_results_per_query.append(
                    [(doc, doc.metadata.get("_score", 0.0)) for doc in dense_docs]
                )
            except Exception as exc:
                logger.warning(f"Dense search for variant '{q[:40]}' failed: {exc}")
                dense_results_per_query.append([])

        # 3. Merge results across query variants (union, taking best score)
        merged_bm25 = self._merge_variant_results(bm25_results_per_query)
        merged_dense = self._merge_variant_results(dense_results_per_query)

        # 4. RRF fusion
        fused = self._reciprocal_rank_fusion(merged_dense, merged_bm25)

        # 5. Deduplicate by content and return top-k
        seen_content: set[str] = set()
        final_results: List[Tuple[Document, float]] = []
        for doc, score in fused:
            content_key = doc.page_content[:200]
            if content_key not in seen_content:
                seen_content.add(content_key)
                final_results.append((doc, score))
            if len(final_results) >= k:
                break

        logger.info(
            f"Hybrid search returned {len(final_results)} unique result(s)"
        )
        return final_results

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------

    async def _expand_query(self, query: str) -> List[str]:
        """
        Use an LLM to generate 3 alternative phrasings of the query to
        improve recall when the user's phrasing doesn't lexically match
        the indexed text.

        Args:
            query: Original user query.

        Returns:
            List of up to 3 alternative query strings.
        """
        prompt = (
            "You are a search query expansion assistant. "
            "Generate exactly 3 alternative phrasings of the following search query "
            "that capture the same information need but use different vocabulary. "
            "Return ONLY the 3 alternatives, one per line, no numbering, no explanations.\n\n"
            f"Original query: {query}"
        )

        try:
            from langchain_core.messages import HumanMessage
            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            lines = [
                line.strip()
                for line in response.content.strip().split("\n")
                if line.strip() and line.strip() != query
            ]
            return lines[:3]
        except Exception as exc:
            logger.debug(f"LLM query expansion error: {exc}")
            return []

    # ------------------------------------------------------------------
    # BM25 search
    # ------------------------------------------------------------------

    def _bm25_search(
        self,
        query: str,
        k: int,
    ) -> List[Tuple[Document, float]]:
        """
        Run BM25 keyword search against the in-memory index.

        Args:
            query: Search query.
            k:     Maximum number of results.

        Returns:
            List of ``(Document, normalised_bm25_score)`` tuples, sorted descending.
        """
        if self._bm25_index is None or not self._bm25_corpus:
            logger.debug("BM25 index not built — returning empty results")
            return []

        tokens = self._tokenise(query)
        if not tokens:
            return []

        try:
            scores: np.ndarray = self._bm25_index.get_scores(tokens)
        except Exception as exc:
            logger.warning(f"BM25 scoring failed: {exc}")
            return []

        # Normalise scores to [0, 1]
        max_score = float(np.max(scores)) if scores.size else 0.0
        if max_score > 0:
            normalised = scores / max_score
        else:
            normalised = scores

        top_indices = np.argsort(normalised)[::-1][:k]
        results: List[Tuple[Document, float]] = []
        for idx in top_indices:
            score = float(normalised[idx])
            if score <= 0:
                break
            results.append((self._bm25_corpus[idx], score))

        return results

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[Document, float]],
        sparse_results: List[Tuple[Document, float]],
        k_constant: int = 60,
    ) -> List[Tuple[Document, float]]:
        """
        Reciprocal Rank Fusion (RRF) merges two ranked lists into a single
        ordering robust to score-scale differences between retrievers.

        Formula: ``score = sum(1 / (k_constant + rank))`` across result lists.

        Args:
            dense_results:  (Document, score) list from dense search.
            sparse_results: (Document, score) list from BM25 search.
            k_constant:     RRF constant controlling rank sensitivity.

        Returns:
            Merged, deduplicated list sorted by descending RRF score.
        """
        # Map a stable doc key → (Document, rrf_score)
        doc_scores: Dict[str, Tuple[Document, float]] = {}

        def _doc_key(doc: Document) -> str:
            return doc.page_content[:200]

        for result_list, weight in (
            (dense_results, self._dense_weight),
            (sparse_results, self._bm25_weight),
        ):
            for rank, (doc, _score) in enumerate(result_list, start=1):
                key = _doc_key(doc)
                rrf_contribution = weight * (1.0 / (k_constant + rank))
                if key in doc_scores:
                    existing_doc, existing_score = doc_scores[key]
                    doc_scores[key] = (existing_doc, existing_score + rrf_contribution)
                else:
                    doc_scores[key] = (doc, rrf_contribution)

        sorted_results = sorted(
            doc_scores.values(), key=lambda x: x[1], reverse=True
        )
        return sorted_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenise(self, text: str) -> List[str]:
        """
        Simple whitespace + punctuation tokeniser for BM25.
        Lowercases and strips punctuation.
        """
        text = text.lower()
        tokens = re.findall(r"\b[a-z0-9]+\b", text)
        return tokens

    def _merge_variant_results(
        self,
        results_per_query: List[List[Tuple[Document, float]]],
    ) -> List[Tuple[Document, float]]:
        """
        Merge results from multiple query variants, keeping the highest
        score for duplicate documents.

        Args:
            results_per_query: List of ranked result lists, one per variant.

        Returns:
            Merged list sorted by descending score.
        """
        best: Dict[str, Tuple[Document, float]] = {}
        for results in results_per_query:
            for doc, score in results:
                key = doc.page_content[:200]
                if key not in best or score > best[key][1]:
                    best[key] = (doc, score)
        return sorted(best.values(), key=lambda x: x[1], reverse=True)
