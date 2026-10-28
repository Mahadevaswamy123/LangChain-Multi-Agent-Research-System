"""
Celery asynchronous tasks for distributed research pipeline execution.
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

from celery import Celery

from src.pipelines.pipeline import ResearchPipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Initialize Celery app
celery_app = Celery("research_worker")
celery_app.config_from_object("workers.celery_config")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@celery_app.task(
    bind=True,
    max_retries=2,
    soft_time_limit=300,  # 5 min soft timeout
    time_limit=360,       # 6 min hard timeout
    name="tasks.run_research",
)
def run_research_task(
    self,
    run_id: str,
    topic: str,
    user_id: str = "public",
    max_iterations: int = 2,
) -> Dict[str, Any]:
    """
    Execute the Multi-Agent Research Pipeline asynchronously inside a Celery worker.

    Steps:
      1. Initializes the stateful ResearchPipeline.
      2. Executes multi-agent research with hybrid RAG.
      3. Performs RAGAS evaluation.
      4. Returns structured results for persistence.
    """
    logger.info(
        f"[CeleryTask] Processing task {self.request.id} for run_id={run_id!r}, topic={topic!r}"
    )
    t0 = time.time()

    pipeline = ResearchPipeline(
        topic=topic,
        max_iterations=max_iterations,
        user_id=user_id,
    )

    try:
        # Run async pipeline using event loop in worker thread
        result = asyncio.run(
            pipeline.arun(
                topic=topic,
                user_id=user_id,
                session_id=run_id,
                evaluate_ragas=True,
            )
        )
        duration = round(time.time() - t0, 2)
        logger.info(f"[CeleryTask] Successfully completed run_id={run_id} in {duration}s")

        return {
            "run_id": run_id,
            "status": "done",
            "topic": topic,
            "final_report": result.get("final_report", ""),
            "sources": result.get("sources", []),
            "critic_score": result.get("critic_score"),
            "critic_feedback": result.get("critic_feedback", ""),
            "evaluation": result.get("evaluation"),
            "duration_sec": duration,
            "completed_at": _now_iso(),
        }

    except Exception as exc:
        logger.error(f"[CeleryTask] Task failed for run_id={run_id}: {exc}", exc_info=True)
        # Attempt retry if eligible
        if self.request.retries < self.max_retries:
            logger.warning(f"[CeleryTask] Retrying task {self.request.id} (attempt {self.request.retries + 1})")
            raise self.retry(exc=exc, countdown=10 * (2 ** self.request.retries))

        return {
            "run_id": run_id,
            "status": "failed",
            "error": str(exc),
            "completed_at": _now_iso(),
        }

