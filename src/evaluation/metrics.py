"""
Custom Prometheus metrics and instrumentation for the Multi-Agent Research System.

Tracks API requests, pipeline latency, agent error rates, vector store retrieval
performance, RAGAS evaluation scores, and revision iterations.
"""
from __future__ import annotations

from typing import Optional

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Request & Pipeline Metrics
# ---------------------------------------------------------------------------

RESEARCH_REQUESTS_TOTAL = Counter(
    "research_requests_total",
    "Total number of research pipeline requests",
    ["status", "user_tier"],
)

RESEARCH_DURATION_SECONDS = Histogram(
    "research_duration_seconds",
    "End-to-end duration of research pipeline runs in seconds",
    ["status"],
    buckets=(10.0, 30.0, 60.0, 120.0, 180.0, 300.0, 600.0),
)

AGENT_ERRORS_TOTAL = Counter(
    "agent_errors_total",
    "Total count of agent invocation errors",
    ["agent_name", "error_type"],
)

ACTIVE_REQUESTS = Gauge(
    "active_requests_gauge",
    "Number of currently active research pipeline executions",
)

# ---------------------------------------------------------------------------
# RAG & Retrieval Metrics
# ---------------------------------------------------------------------------

VECTOR_SEARCH_LATENCY = Histogram(
    "vector_search_latency_seconds",
    "Latency of Qdrant vector store search operations",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

EMBEDDINGS_GENERATED = Counter(
    "embeddings_generated_total",
    "Total count of document text embeddings generated",
)

HYBRID_SEARCH_RESULTS = Histogram(
    "hybrid_search_results_count",
    "Count of search results returned by hybrid search",
    buckets=(1, 3, 5, 8, 10, 15, 20),
)

QDRANT_COLLECTION_SIZE = Gauge(
    "qdrant_collection_size",
    "Total count of vector points indexed in Qdrant",
)

# ---------------------------------------------------------------------------
# Quality & Evaluation Metrics
# ---------------------------------------------------------------------------

CRITIC_SCORE = Histogram(
    "critic_score",
    "Distribution of CriticAgent evaluation scores (0.0 - 10.0)",
    buckets=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 9.5, 10.0),
)

RAGAS_FAITHFULNESS = Histogram(
    "ragas_faithfulness",
    "RAGAS evaluation faithfulness score (0.0 - 1.0)",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0),
)

RAGAS_ANSWER_RELEVANCY = Histogram(
    "ragas_answer_relevancy",
    "RAGAS answer relevancy score (0.0 - 1.0)",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0),
)

REVISION_ITERATIONS = Histogram(
    "revision_iterations_count",
    "Number of writer-critic revision iterations performed",
    buckets=(0, 1, 2, 3, 4),
)


# ---------------------------------------------------------------------------
# Metric Recording Helper
# ---------------------------------------------------------------------------

def record_research_run(
    status: str,
    duration: float,
    score: Optional[float] = None,
    iterations: int = 0,
    user_tier: str = "free",
) -> None:
    """
    Convenience function to observe standard Prometheus metrics upon run completion.
    """
    RESEARCH_REQUESTS_TOTAL.labels(status=status, user_tier=user_tier).inc()
    RESEARCH_DURATION_SECONDS.labels(status=status).observe(duration)
    REVISION_ITERATIONS.observe(iterations)

    if score is not None:
        CRITIC_SCORE.observe(score)

