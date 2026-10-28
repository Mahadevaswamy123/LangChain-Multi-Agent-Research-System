"""
Research pipeline state using TypedDict for LangGraph.

This module defines the shared state schema that flows through every node
in the LangGraph research pipeline. All state mutations are handled by
node return dicts; the Annotated[list, add] pattern lets multiple nodes
append to `messages` without overwriting each other.
"""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, List, Optional, TypedDict


class ResearchState(TypedDict):
    """Full state passed between LangGraph pipeline nodes."""

    # ------------------------------------------------------------------ #
    # Input                                                                #
    # ------------------------------------------------------------------ #
    topic: str          # Research question / topic requested by the user
    user_id: str        # Caller identity (auth / billing context)
    session_id: str     # Unique pipeline run ID (used as LangGraph thread_id)

    # ------------------------------------------------------------------ #
    # Agent message bus (accumulated across all nodes)                    #
    # ------------------------------------------------------------------ #
    # Annotated with the `add` reducer so each node can append without
    # overwriting earlier messages from other nodes.
    messages: Annotated[list, add]

    # ------------------------------------------------------------------ #
    # Research data (populated progressively by pipeline nodes)           #
    # ------------------------------------------------------------------ #
    search_results: List[dict]   # Raw results from Tavily: {title, url, snippet, score, published_date}
    scraped_content: List[dict]  # Full page content: {url, title, content, word_count, success}
    retrieved_docs: List[dict]   # Qdrant hybrid-search hits: {content, source, score, chunk_index}

    # ------------------------------------------------------------------ #
    # Pipeline outputs                                                     #
    # ------------------------------------------------------------------ #
    draft_report: str       # Markdown report produced by WriterAgent
    critic_feedback: str    # Free-text critique from CriticAgent
    critic_score: float     # Overall numeric score (0-10) assigned by CriticAgent
    final_report: str       # Polished markdown produced by SynthesisAgent

    # ------------------------------------------------------------------ #
    # Control flow                                                         #
    # ------------------------------------------------------------------ #
    iteration: int          # Current writer→critic refinement iteration (starts 0)
    max_iterations: int     # Maximum allowed refinement loops (default 2)
    should_revise: bool     # Set by critic_node; drives the conditional edge

    # ------------------------------------------------------------------ #
    # Metadata                                                             #
    # ------------------------------------------------------------------ #
    sources: List[str]              # Collected source URLs (de-duplicated)
    error: Optional[str]            # Last error message (None if no error)
    pipeline_metadata: dict         # Timing, token usage, node latencies, etc.
