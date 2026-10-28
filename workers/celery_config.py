"""
Celery configuration for the Multi-Agent Research System.
"""
from __future__ import annotations

import os

broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

task_serializer = "json"
result_serializer = "json"
accept_content = ["json"]
timezone = "UTC"
enable_utc = True

task_track_started = True
worker_prefetch_multiplier = 1
task_acks_late = True
worker_max_tasks_per_child = 50  # Prevent memory leaks from large language model contexts

# Task routing
task_routes = {
    "tasks.run_research": {"queue": "research_queue"},
}

