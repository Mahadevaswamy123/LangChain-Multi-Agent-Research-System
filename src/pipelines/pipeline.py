"""
Production Research Pipeline orchestrator.

Integrates the stateful LangGraph multi-agent workflow with RAG retrieval,
hybrid search, evaluation loop, and structured observability.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.core.config import get_settings
from src.evaluation.metrics import record_research_run
from src.evaluation.ragas_eval import RAGASEvaluator
from src.pipelines.graph import build_research_graph
from src.pipelines.state import ResearchState
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ResearchPipeline:
    """
    Production-grade Multi-Agent Research Pipeline.

    Orchestrates Search, Reader, Retrieval (Qdrant + BM25 Hybrid),
    Writer, Critic (with conditional revision loop), and Synthesis.
    """

    def __init__(
        self,
        topic: Optional[str] = None,
        max_iterations: int = 2,
        user_id: str = "public",
    ) -> None:
        self.topic = topic or ""
        self.max_iterations = max_iterations
        self.user_id = user_id
        self._graph = build_research_graph()

    async def arun(
        self,
        topic: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        evaluate_ragas: bool = True,
    ) -> Dict[str, Any]:
        """
        Asynchronously execute the full multi-agent research pipeline.

        Args:
            topic:          The research question or topic.
            user_id:        Tenant / user ID.
            session_id:     Unique identifier for checkpointer & vector scoping.
            evaluate_ragas: Whether to run asynchronous RAGAS evaluation after generation.

        Returns:
            Dictionary containing final report, sources, scores, and evaluation metrics.
        """
        target_topic = topic or self.topic
        if not target_topic:
            raise ValueError("Research topic must not be empty.")

        target_user = user_id or self.user_id
        sid = session_id or str(uuid.uuid4())
        start_time = time.time()

        logger.info(
            f"[ResearchPipeline] Starting run for topic: {target_topic!r} (session: {sid})"
        )

        initial_state: ResearchState = {
            "topic": target_topic,
            "user_id": target_user,
            "session_id": sid,
            "messages": [],
            "search_results": [],
            "scraped_content": [],
            "retrieved_docs": [],
            "draft_report": "",
            "critic_feedback": "",
            "critic_score": 0.0,
            "final_report": "",
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "should_revise": False,
            "sources": [],
            "error": None,
            "pipeline_metadata": {
                "started_at": time.time(),
                "session_id": sid,
            },
        }

        config = {"configurable": {"thread_id": sid}}

        try:
            final_state = await self._graph.ainvoke(initial_state, config=config)
        except Exception as e:
            logger.error(f"[ResearchPipeline] Execution failed: {e}", exc_info=True)
            record_research_run(status="failed", duration=time.time() - start_time)
            raise

        elapsed = round(time.time() - start_time, 2)
        final_report = final_state.get("final_report") or final_state.get("draft_report", "")
        sources = final_state.get("sources", [])
        critic_score = final_state.get("critic_score", 0.0)
        retrieved_docs = final_state.get("retrieved_docs", [])
        iterations_taken = final_state.get("iteration", 1)

        # Run RAGAS evaluation on the generated report and retrieved context
        ragas_result = None
        if evaluate_ragas and retrieved_docs and final_report:
            try:
                evaluator = RAGASEvaluator()
                context_texts = [d.get("content", "") for d in retrieved_docs if d.get("content")]
                if context_texts:
                    logger.info("[ResearchPipeline] Running RAGAS evaluation on generated deliverable...")
                    ragas_result = await evaluator.evaluate_response(
                        question=target_topic,
                        answer=final_report,
                        contexts=context_texts,
                        sources=sources,
                    )
            except Exception as eval_err:
                logger.warning(f"[ResearchPipeline] RAGAS evaluation skipped/failed: {eval_err}")

        # Record metrics
        record_research_run(
            status="success",
            duration=elapsed,
            score=critic_score,
            iterations=iterations_taken,
        )

        metadata = dict(final_state.get("pipeline_metadata", {}))
        metadata["total_duration_sec"] = elapsed
        metadata["iterations_completed"] = iterations_taken

        logger.info(
            f"[ResearchPipeline] Completed run in {elapsed}s with critic score {critic_score}/10"
        )

        return {
            "topic": target_topic,
            "report": final_report,
            "final_report": final_report,
            "draft_report": final_state.get("draft_report", ""),
            "sources": sources,
            "critic_score": critic_score,
            "critic_feedback": final_state.get("critic_feedback", ""),
            "evaluation": ragas_result,
            "search_results": final_state.get("search_results", []),
            "scraped_content": final_state.get("scraped_content", []),
            "retrieved_docs": retrieved_docs,
            "metadata": metadata,
            "session_id": sid,
        }

    def run(
        self,
        topic: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        evaluate_ragas: bool = True,
    ) -> Dict[str, Any]:
        """
        Synchronous execution entry point.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(
                    self.arun(topic=topic, user_id=user_id, session_id=session_id, evaluate_ragas=evaluate_ragas)
                )
            return loop.run_until_complete(
                self.arun(topic=topic, user_id=user_id, session_id=session_id, evaluate_ragas=evaluate_ragas)
            )
        except RuntimeError:
            return asyncio.run(
                self.arun(topic=topic, user_id=user_id, session_id=session_id, evaluate_ragas=evaluate_ragas)
            )

    async def astream(
        self,
        topic: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream events as each node executes in the LangGraph state machine.
        """
        target_topic = topic or self.topic
        sid = session_id or str(uuid.uuid4())
        initial_state: ResearchState = {
            "topic": target_topic,
            "user_id": user_id or self.user_id,
            "session_id": sid,
            "messages": [],
            "search_results": [],
            "scraped_content": [],
            "retrieved_docs": [],
            "draft_report": "",
            "critic_feedback": "",
            "critic_score": 0.0,
            "final_report": "",
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "should_revise": False,
            "sources": [],
            "error": None,
            "pipeline_metadata": {"started_at": time.time(), "session_id": sid},
        }

        config = {"configurable": {"thread_id": sid}}
        async for event in self._graph.astream(initial_state, config=config):
            yield event


def run_research_pipeline(topic: str) -> Dict[str, Any]:
    """
    Backward-compatible synchronous function for quick invocation (e.g., from main.py).
    """
    pipeline = ResearchPipeline(topic=topic)
    return pipeline.run()
