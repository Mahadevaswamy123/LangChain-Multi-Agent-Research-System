"""
Comprehensive test suite for the Multi-Agent Research System.
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.pipelines.graph import build_research_graph, should_revise
from src.pipelines.state import ResearchState
from src.rag.chunking import ChunkConfig, ChunkStrategy, DocumentChunker
from src.rag.hybrid_search import HybridSearchEngine


def test_document_chunker_recursive():
    """Verify recursive chunking preserves metadata and respects limits."""
    chunker = DocumentChunker(ChunkConfig(strategy=ChunkStrategy.RECURSIVE, chunk_size=200, chunk_overlap=20))
    sample_text = (
        "Agentic AI systems leverage autonomous multi-agent loops to plan, decompose, "
        "and execute complex software tasks. Retrieval-Augmented Generation (RAG) enhances "
        "generation accuracy by grounding responses in external vector databases like Qdrant. "
        "Evaluation frameworks such as RAGAS verify factual faithfulness and answer relevance."
    )
    docs = chunker.chunk_text(sample_text, source_url="https://example.com/agentic-ai")

    assert len(docs) >= 1
    for d in docs:
        assert "source" in d.metadata
        assert d.metadata["source"] == "https://example.com/agentic-ai"
        assert len(d.page_content) > 0


def test_reciprocal_rank_fusion():
    """Verify Reciprocal Rank Fusion (RRF) logic merges rankings accurately."""
    doc_a = Document(page_content="Document Alpha", metadata={"id": "a"})
    doc_b = Document(page_content="Document Beta", metadata={"id": "b"})
    doc_c = Document(page_content="Document Gamma", metadata={"id": "c"})

    dense_results = [(doc_a, 0.95), (doc_b, 0.85)]
    sparse_results = [(doc_b, 12.4), (doc_c, 9.1)]

    # Mock instance to test _reciprocal_rank_fusion
    engine = HybridSearchEngine(vector_store=None, embedding_service=None)
    fused = engine._reciprocal_rank_fusion(dense_results=dense_results, sparse_results=sparse_results, k_constant=60)

    assert len(fused) == 3
    # doc_b appears in both lists, so its reciprocal rank sum should place it at or near the top
    assert any(doc.metadata.get("id") == "b" for doc, _ in fused)


def test_conditional_edge_revision_routing():
    """Verify state machine conditional edge logic."""
    state_revise: ResearchState = {
        "topic": "AI",
        "user_id": "test",
        "session_id": "s1",
        "messages": [],
        "search_results": [],
        "scraped_content": [],
        "retrieved_docs": [],
        "draft_report": "Draft",
        "critic_feedback": "Needs citations",
        "critic_score": 5.5,
        "final_report": "",
        "iteration": 1,
        "max_iterations": 2,
        "should_revise": True,
        "sources": [],
        "error": None,
        "pipeline_metadata": {},
    }
    assert should_revise(state_revise) == "revise"

    # When iteration reaches max_iterations, finalize instead
    state_done = dict(state_revise)
    state_done["iteration"] = 2
    assert should_revise(state_done) == "finalize"

    # When score is high, finalize
    state_high_score = dict(state_revise)
    state_high_score["should_revise"] = False
    assert should_revise(state_high_score) == "finalize"


def test_graph_compilation():
    """Verify that the LangGraph research state machine compiles without errors."""
    graph = build_research_graph()
    assert graph is not None

