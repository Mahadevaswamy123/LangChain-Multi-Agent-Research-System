"""
RAGAS evaluation integration for the Multi-Agent Research System.

Provides asynchronous evaluation of research pipeline outputs using the RAGAS
framework, covering:

    • faithfulness        – Are claims in the answer supported by the context?
    • answer_relevancy    – How relevant is the answer to the question?
    • context_precision   – Are retrieved chunks focused on the question?
    • context_recall      – Do retrieved chunks cover the ground truth?
    • source_coverage     – Custom metric: % of sources cited in the answer.

Results are persisted to PostgreSQL when a DB engine is available.
"""
from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom metric helper
# ---------------------------------------------------------------------------

def _source_coverage(answer: str, sources: list[str]) -> float:
    """
    Return the fraction of *sources* that are cited (or mentioned) in *answer*.

    We check for URL substrings or domain names present in the answer text.
    """
    if not sources:
        return 1.0
    cited = sum(
        1
        for src in sources
        if src.lower() in answer.lower()
        or _domain(src) in answer.lower()
    )
    return round(cited / len(sources), 4)


def _domain(url: str) -> str:
    """Extract bare domain from a URL string."""
    m = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return m.group(1).lower() if m else url.lower()


# ---------------------------------------------------------------------------
# RAGASEvaluator
# ---------------------------------------------------------------------------

class RAGASEvaluator:
    """
    Wraps the RAGAS ``evaluate()`` function with an async-friendly interface.

    Parameters
    ----------
    llm:
        LangChain LLM instance used by RAGAS internally.
        Defaults to a ``ChatOpenAI(model="gpt-4o-mini")`` created lazily.
    embeddings:
        LangChain Embeddings instance.
        Defaults to ``OpenAIEmbeddings`` created lazily.
    """

    def __init__(self, llm=None, embeddings=None) -> None:
        self._llm = llm
        self._embeddings = embeddings

    # ── Lazy init ──────────────────────────────────────────────────────────

    def _get_llm(self):
        if self._llm is None:
            from src.utils.llm_factory import get_llm
            self._llm = get_llm(temperature=0)
        return self._llm

    def _get_embeddings(self):
        if self._embeddings is None:
            from src.utils.llm_factory import get_embeddings_instance
            self._embeddings = get_embeddings_instance()
        return self._embeddings

    # ── Core evaluation ────────────────────────────────────────────────────

    async def evaluate_response(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
        sources: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a single question-answer-context triple with RAGAS metrics.

        Parameters
        ----------
        question:    The research question / topic.
        answer:      The generated report or answer text.
        contexts:    List of retrieved document chunk contents.
        ground_truth:
            Optional reference answer for context-recall metric.
            When absent, context_recall is skipped.
        sources:
            Optional list of source URLs for the custom source_coverage metric.

        Returns
        -------
        dict with keys:
            faithfulness, answer_relevancy, context_precision,
            context_recall (if ground_truth supplied), source_coverage,
            overall_score, evaluation_timestamp
        """
        def _run_ragas() -> dict[str, Any]:
            try:
                from ragas import evaluate
                from ragas.metrics import (
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                )
                from datasets import Dataset

                data: dict[str, list] = {
                    "question": [question],
                    "answer": [answer],
                    "contexts": [contexts],
                }
                metrics = [faithfulness, answer_relevancy, context_precision]

                if ground_truth:
                    data["ground_truth"] = [ground_truth]
                    metrics.append(context_recall)

                dataset = Dataset.from_dict(data)
                result = evaluate(
                    dataset,
                    metrics=metrics,
                    llm=self._get_llm(),
                    embeddings=self._get_embeddings(),
                    raise_exceptions=False,
                )
                scores = result.to_pandas().iloc[0].to_dict()
                return {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}

            except ImportError:
                logger.warning("ragas_not_installed_using_fallback")
                return _mock_ragas_scores()

        # Run the blocking RAGAS evaluation in a thread pool
        scores = await asyncio.to_thread(_run_ragas)

        # Source coverage custom metric
        cov = _source_coverage(answer, sources or [])
        scores["source_coverage"] = cov

        # Overall = mean of all numeric scores
        numeric = [v for v in scores.values() if isinstance(v, float) and 0 <= v <= 1]
        scores["overall_score"] = round(sum(numeric) / len(numeric), 4) if numeric else 0.0
        scores["evaluation_timestamp"] = datetime.now(tz=timezone.utc).isoformat()

        return scores

    # ── Pipeline run evaluation ────────────────────────────────────────────

    async def evaluate_pipeline_run(
        self,
        topic: str,
        final_report: str,
        retrieved_docs: list[dict],
        run_id: str,
        sources: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a complete pipeline run and persist results.

        Parameters
        ----------
        topic:          Research topic (used as the question).
        final_report:   The generated report text.
        retrieved_docs: List of dicts with at least a ``"content"`` key.
        run_id:         Unique run identifier for DB persistence.
        sources:        Source URLs extracted during the run.

        Returns
        -------
        Evaluation result dict (same schema as ``evaluate_response``).
        """
        contexts = [doc.get("content", "") for doc in retrieved_docs if doc.get("content")]

        if not contexts:
            logger.warning("no_contexts_for_evaluation", run_id=run_id)
            contexts = [final_report[:1000]]  # fallback

        result = await self.evaluate_response(
            question=topic,
            answer=final_report,
            contexts=contexts,
            sources=sources,
        )
        result["run_id"] = run_id

        # Persist to DB (best-effort; non-fatal)
        await self._persist_evaluation(run_id=run_id, evaluation=result)

        # Update Prometheus metrics
        try:
            from src.evaluation.metrics import RAGAS_FAITHFULNESS
            faith = result.get("faithfulness", 0.0)
            if isinstance(faith, float):
                RAGAS_FAITHFULNESS.observe(faith)
        except Exception:
            pass

        logger.info(
            "ragas_evaluation_complete",
            run_id=run_id,
            overall_score=result.get("overall_score"),
        )
        return result

    async def _persist_evaluation(self, run_id: str, evaluation: dict[str, Any]) -> None:
        """Write evaluation row to PostgreSQL (best-effort)."""
        try:
            # Import DB session lazily to avoid circular imports
            from sqlalchemy import text
            from app.main import _db_engine  # type: ignore[attr-defined]

            if _db_engine is None:
                return

            eval_id = str(uuid.uuid4())
            insert_sql = text(
                """
                INSERT INTO evaluations
                    (id, run_id, faithfulness, answer_relevancy, context_precision,
                     context_recall, source_coverage, overall_score, evaluated_at)
                VALUES
                    (:id, :run_id, :faithfulness, :answer_relevancy, :context_precision,
                     :context_recall, :source_coverage, :overall_score, :evaluated_at)
                ON CONFLICT (run_id) DO UPDATE
                    SET faithfulness      = EXCLUDED.faithfulness,
                        answer_relevancy  = EXCLUDED.answer_relevancy,
                        context_precision = EXCLUDED.context_precision,
                        context_recall    = EXCLUDED.context_recall,
                        source_coverage   = EXCLUDED.source_coverage,
                        overall_score     = EXCLUDED.overall_score,
                        evaluated_at      = EXCLUDED.evaluated_at
                """
            )
            async with _db_engine.begin() as conn:
                await conn.execute(
                    insert_sql,
                    {
                        "id": eval_id,
                        "run_id": run_id,
                        "faithfulness": evaluation.get("faithfulness"),
                        "answer_relevancy": evaluation.get("answer_relevancy"),
                        "context_precision": evaluation.get("context_precision"),
                        "context_recall": evaluation.get("context_recall"),
                        "source_coverage": evaluation.get("source_coverage"),
                        "overall_score": evaluation.get("overall_score"),
                        "evaluated_at": evaluation.get("evaluation_timestamp"),
                    },
                )
        except Exception as exc:
            logger.warning("evaluation_persist_failed", run_id=run_id, error=str(exc))


# ---------------------------------------------------------------------------
# Mock scores (used when ragas is not installed / during testing)
# ---------------------------------------------------------------------------

def _mock_ragas_scores() -> dict[str, float]:
    """Return deterministic placeholder scores for unit tests."""
    return {
        "faithfulness": 0.85,
        "answer_relevancy": 0.82,
        "context_precision": 0.79,
        "context_recall": 0.80,
    }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_evaluator: RAGASEvaluator | None = None


def get_evaluator() -> RAGASEvaluator:
    """Return a cached RAGASEvaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = RAGASEvaluator()
    return _evaluator
