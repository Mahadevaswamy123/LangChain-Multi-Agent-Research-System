"""
Research API routes.

Endpoints
---------
POST   /api/v1/research/run              – Submit async research task
GET    /api/v1/research/status/{run_id}  – Poll run status
GET    /api/v1/research/result/{run_id}  – Retrieve completed result
GET    /api/v1/research/history          – Paginated run history
DELETE /api/v1/research/{run_id}         – Delete a run
WS     /api/v1/research/stream/{run_id}  – Stream pipeline events
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from app.core.security import get_current_active_user
from src.evaluation.metrics import ACTIVE_REQUESTS, RESEARCH_REQUESTS_TOTAL

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Research topic or question",
    )
    max_iterations: int = Field(
        2,
        ge=1,
        le=3,
        description="Maximum critic-revision cycles (1–3)",
    )


class ResearchAccepted(BaseModel):
    run_id: str
    status: str = "accepted"
    message: str


class RunStatus(BaseModel):
    run_id: str
    status: str          # queued | running | done | failed
    progress_pct: int    # 0-100
    created_at: str
    completed_at: str | None


class RunResult(BaseModel):
    run_id: str
    final_report: str
    sources: list[str]
    critic_score: float | None
    evaluation: dict[str, Any] | None
    metadata: dict[str, Any]


class HistoryItem(BaseModel):
    run_id: str
    topic: str
    status: str
    critic_score: float | None
    created_at: str


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# In-memory run store (replace with DB in production)
# ---------------------------------------------------------------------------
# Keyed by run_id → {status, topic, user_id, created_at, result, error, ...}
_runs: dict[str, dict[str, Any]] = {}

# WebSocket connections: run_id → list of WebSocket
_ws_connections: dict[str, list[WebSocket]] = {}


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Helper: broadcast event to all WebSocket subscribers of a run
# ---------------------------------------------------------------------------

async def broadcast_event(run_id: str, event: str, data: dict[str, Any]) -> None:
    connections = _ws_connections.get(run_id, [])
    payload = json.dumps({"event": event, "data": data, "timestamp": _now_iso()})
    dead: list[WebSocket] = []
    for ws in connections:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connections.remove(ws)


# ---------------------------------------------------------------------------
# POST /run
# ---------------------------------------------------------------------------

@router.post(
    "/run",
    response_model=ResearchAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a research pipeline run",
)
async def run_research(
    body: ResearchRequest,
    current_user: Annotated[dict[str, Any], Depends(get_current_active_user)],
) -> ResearchAccepted:
    """Submit a new research job. Returns immediately with a ``run_id``."""
    run_id = str(uuid.uuid4())
    user_id = current_user.get("sub", "anonymous")

    _runs[run_id] = {
        "run_id": run_id,
        "topic": body.topic,
        "user_id": user_id,
        "max_iterations": body.max_iterations,
        "status": "queued",
        "progress_pct": 0,
        "created_at": _now_iso(),
        "completed_at": None,
        "final_report": None,
        "sources": [],
        "critic_score": None,
        "evaluation": None,
        "error": None,
    }

    # Submit Celery task
    try:
        from workers.tasks import run_research_task
        run_research_task.apply_async(
            kwargs={
                "run_id": run_id,
                "topic": body.topic,
                "user_id": user_id,
                "max_iterations": body.max_iterations,
            },
            task_id=run_id,
        )
    except Exception as exc:
        # If Celery is unavailable, run synchronously in background
        logger.warning("celery_unavailable_running_inline", error=str(exc))
        asyncio.create_task(_run_inline(run_id, body.topic, user_id, body.max_iterations))

    RESEARCH_REQUESTS_TOTAL.labels(status="accepted", user_tier="free").inc()
    ACTIVE_REQUESTS.inc()

    logger.info("research_run_submitted", run_id=run_id, topic=body.topic)
    return ResearchAccepted(
        run_id=run_id,
        status="accepted",
        message=f"Research pipeline started. Poll /status/{run_id} for progress.",
    )


async def _run_inline(run_id: str, topic: str, user_id: str, max_iterations: int) -> None:
    """Fallback: execute the pipeline in-process when Celery is not available."""
    import time
    from src.pipelines.pipeline import ResearchPipeline

    run = _runs[run_id]
    try:
        run["status"] = "running"
        await broadcast_event(run_id, "pipeline_started", {"topic": topic})

        pipeline = ResearchPipeline(topic=topic, max_iterations=max_iterations)
        t0 = time.time()
        result = await asyncio.to_thread(pipeline.run)
        duration = time.time() - t0

        run.update(
            status="done",
            progress_pct=100,
            completed_at=_now_iso(),
            final_report=result.get("report", ""),
            sources=result.get("sources", []),
            critic_score=result.get("critic_score"),
            evaluation=result.get("evaluation"),
        )
        await broadcast_event(run_id, "pipeline_complete", {"duration_s": round(duration, 2)})

        from src.evaluation.metrics import ACTIVE_REQUESTS, RESEARCH_DURATION_SECONDS
        ACTIVE_REQUESTS.dec()
        RESEARCH_DURATION_SECONDS.labels(status="success").observe(duration)

    except Exception as exc:
        run["status"] = "failed"
        run["error"] = str(exc)
        run["completed_at"] = _now_iso()
        await broadcast_event(run_id, "pipeline_error", {"error": str(exc)})
        ACTIVE_REQUESTS.dec()
        logger.exception("inline_run_failed", run_id=run_id, error=str(exc))


# ---------------------------------------------------------------------------
# GET /status/{run_id}
# ---------------------------------------------------------------------------

@router.get(
    "/status/{run_id}",
    response_model=RunStatus,
    summary="Get run status",
)
async def get_run_status(
    run_id: str,
    current_user: Annotated[dict[str, Any], Depends(get_current_active_user)],
) -> RunStatus:
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    _assert_owner(run, current_user)

    return RunStatus(
        run_id=run_id,
        status=run["status"],
        progress_pct=run["progress_pct"],
        created_at=run["created_at"],
        completed_at=run.get("completed_at"),
    )


# ---------------------------------------------------------------------------
# GET /result/{run_id}
# ---------------------------------------------------------------------------

@router.get(
    "/result/{run_id}",
    response_model=RunResult,
    summary="Get run result",
)
async def get_run_result(
    run_id: str,
    current_user: Annotated[dict[str, Any], Depends(get_current_active_user)],
) -> RunResult:
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    _assert_owner(run, current_user)

    if run["status"] not in ("done",):
        raise HTTPException(
            status_code=409,
            detail=f"Run is not yet complete (current status: {run['status']}).",
        )

    return RunResult(
        run_id=run_id,
        final_report=run.get("final_report") or "",
        sources=run.get("sources") or [],
        critic_score=run.get("critic_score"),
        evaluation=run.get("evaluation"),
        metadata={
            "topic": run["topic"],
            "created_at": run["created_at"],
            "completed_at": run.get("completed_at"),
            "max_iterations": run.get("max_iterations", 2),
        },
    )


# ---------------------------------------------------------------------------
# GET /history
# ---------------------------------------------------------------------------

@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="List past research runs",
)
async def get_history(
    current_user: Annotated[dict[str, Any], Depends(get_current_active_user)],
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
) -> HistoryResponse:
    user_id = current_user.get("sub", "anonymous")
    user_runs = [r for r in _runs.values() if r["user_id"] == user_id]
    user_runs.sort(key=lambda r: r["created_at"], reverse=True)

    total = len(user_runs)
    start = (page - 1) * page_size
    end = start + page_size
    page_runs = user_runs[start:end]

    items = [
        HistoryItem(
            run_id=r["run_id"],
            topic=r["topic"],
            status=r["status"],
            critic_score=r.get("critic_score"),
            created_at=r["created_at"],
        )
        for r in page_runs
    ]

    return HistoryResponse(items=items, total=total, page=page, page_size=page_size)


# ---------------------------------------------------------------------------
# DELETE /{run_id}
# ---------------------------------------------------------------------------

@router.delete(
    "/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a research run",
)
async def delete_run(
    run_id: str,
    current_user: Annotated[dict[str, Any], Depends(get_current_active_user)],
) -> None:
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    _assert_owner(run, current_user)

    # Attempt to remove associated vectors from Qdrant
    try:
        from qdrant_client import AsyncQdrantClient
        from app.core.config import settings as cfg
        client = AsyncQdrantClient(url=cfg.QDRANT_URL, api_key=cfg.QDRANT_API_KEY or None)
        await client.delete(
            collection_name=cfg.QDRANT_COLLECTION_NAME,
            points_selector={"filter": {"must": [{"key": "run_id", "match": {"value": run_id}}]}},
        )
    except Exception as exc:
        logger.warning("qdrant_delete_failed", run_id=run_id, error=str(exc))

    del _runs[run_id]
    logger.info("run_deleted", run_id=run_id)


# ---------------------------------------------------------------------------
# WebSocket /stream/{run_id}
# ---------------------------------------------------------------------------

@router.websocket("/stream/{run_id}")
async def stream_run(websocket: WebSocket, run_id: str) -> None:
    """
    Stream real-time pipeline events for a run as JSON:
    ``{"event": "search_complete", "data": {...}, "timestamp": "..."}``
    """
    await websocket.accept()

    if run_id not in _runs:
        await websocket.send_text(
            json.dumps({"event": "error", "data": {"detail": "Run not found"}, "timestamp": _now_iso()})
        )
        await websocket.close(code=4004)
        return

    if run_id not in _ws_connections:
        _ws_connections[run_id] = []
    _ws_connections[run_id].append(websocket)

    # Send current state immediately so the client has a baseline
    run = _runs[run_id]
    await websocket.send_text(
        json.dumps({
            "event": "status_update",
            "data": {"status": run["status"], "progress_pct": run["progress_pct"]},
            "timestamp": _now_iso(),
        })
    )

    try:
        # Keep connection alive; events are pushed via broadcast_event()
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send a heartbeat ping
                await websocket.send_text(
                    json.dumps({"event": "ping", "data": {}, "timestamp": _now_iso()})
                )
            # If the run is finished, close gracefully
            if _runs.get(run_id, {}).get("status") in ("done", "failed"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        conns = _ws_connections.get(run_id, [])
        if websocket in conns:
            conns.remove(websocket)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _assert_owner(run: dict[str, Any], user: dict[str, Any]) -> None:
    """Raise 403 if the authenticated user is not the run owner."""
    user_id = user.get("sub", "")
    if run.get("user_id") and run["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied.")
