"""
LangGraph-based state machine for the Multi-Agent Research System.

Graph Topology:
  START
    ↓
  [search_node]      ← SearchAgent (Tavily web + news)
    ↓
  [reader_node]      ← ReaderAgent (concurrent multi-strategy scraping)
    ↓
  [retrieval_node]   ← DocumentChunker + Qdrant + BM25 HybridSearch + Reranker
    ↓
  [writer_node]      ← WriterAgent (drafts or revises grounded report)
    ↓
  [critic_node]      ← CriticAgent (scores report 0-10 & identifies gaps)
    ↓
  {should_revise?}   ← Conditional edge:
    ├─ "revise"   → back to [writer_node] (if score < 7.0 and iter < max_iter)
    └─ "finalize" → [synthesis_node]
    ↓
  [synthesis_node]   ← SynthesisAgent (polishes, links sources, appends metadata)
    ↓
   END
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.core.config import get_settings
from src.agents.critic_agent import CriticAgent
from src.agents.reader_agent import ReaderAgent
from src.agents.search_agent import SearchAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.agents.writer_agent import WriterAgent
from src.pipelines.state import ResearchState
from src.rag.chunking import ChunkConfig, ChunkStrategy, DocumentChunker
from src.rag.embeddings import EmbeddingService
from src.rag.hybrid_search import HybridSearchEngine
from src.rag.reranker import DocumentReranker
from src.rag.vector_store import QdrantVectorStore
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Node 1: Search Node
# ---------------------------------------------------------------------------

async def search_node(state: ResearchState) -> Dict[str, Any]:
    """Execute SearchAgent to gather recent and reliable web sources."""
    topic = state["topic"]
    logger.info(f"[Pipeline:search_node] Starting search for topic: {topic!r}")
    start_time = time.time()

    search_agent = SearchAgent()
    max_results = get_settings().MAX_SEARCH_RESULTS

    try:
        results = await search_agent.run(topic=topic, max_results=max_results)
    except Exception as e:
        logger.error(f"[Pipeline:search_node] Search failed: {e}", exc_info=True)
        results = []

    # Extract source URLs
    sources = [r["url"] for r in results if r.get("url")]

    elapsed = round(time.time() - start_time, 2)
    meta = dict(state.get("pipeline_metadata", {}))
    meta["search_duration_sec"] = elapsed
    meta["search_count"] = len(results)

    logger.info(f"[Pipeline:search_node] Completed in {elapsed}s with {len(results)} results")
    return {
        "search_results": results,
        "sources": sources,
        "pipeline_metadata": meta,
        "messages": [f"SearchAgent gathered {len(results)} sources."],
    }


# ---------------------------------------------------------------------------
# Node 2: Reader Node
# ---------------------------------------------------------------------------

async def reader_node(state: ResearchState) -> Dict[str, Any]:
    """Execute ReaderAgent to scrape the top relevant pages."""
    search_results = state.get("search_results", [])
    logger.info(f"[Pipeline:reader_node] Scraping top pages from {len(search_results)} search results")
    start_time = time.time()

    reader_agent = ReaderAgent()
    try:
        scraped = await reader_agent.run(search_results=search_results, max_urls=4)
    except Exception as e:
        logger.error(f"[Pipeline:reader_node] Scraping failed: {e}", exc_info=True)
        scraped = []

    # Update sources with any successfully scraped URLs
    sources = list(state.get("sources", []))
    for s in scraped:
        url = s.get("url")
        if url and url not in sources:
            sources.append(url)

    elapsed = round(time.time() - start_time, 2)
    meta = dict(state.get("pipeline_metadata", {}))
    meta["scrape_duration_sec"] = elapsed
    meta["scraped_pages_count"] = len(scraped)

    logger.info(f"[Pipeline:reader_node] Completed in {elapsed}s with {len(scraped)} scraped pages")
    return {
        "scraped_content": scraped,
        "sources": sources,
        "pipeline_metadata": meta,
        "messages": [f"ReaderAgent successfully extracted deep content from {len(scraped)} pages."],
    }


# ---------------------------------------------------------------------------
# Node 3: Retrieval Node (RAG + Qdrant + Hybrid Search)
# ---------------------------------------------------------------------------

async def retrieval_node(state: ResearchState) -> Dict[str, Any]:
    """
    Chunk scraped text, embed it, index into Qdrant & BM25, and perform hybrid search.
    """
    topic = state["topic"]
    scraped_content = state.get("scraped_content", [])
    user_id = state.get("user_id", "public")
    session_id = state.get("session_id", "default_session")
    start_time = time.time()

    logger.info(f"[Pipeline:retrieval_node] Processing {len(scraped_content)} scraped documents for RAG")

    retrieved_docs: List[dict] = []
    meta = dict(state.get("pipeline_metadata", {}))

    # Convert scraped contents into LangChain Documents
    raw_docs: List[Document] = []
    for page in scraped_content:
        content = page.get("content", "").strip()
        if content:
            raw_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": page.get("url", ""),
                        "title": page.get("title", ""),
                    },
                )
            )

    if raw_docs:
        # 1. Chunk documents using Recursive Strategy with metadata preservation
        chunker = DocumentChunker(
            ChunkConfig(
                strategy=ChunkStrategy.RECURSIVE,
                chunk_size=get_settings().CHUNK_SIZE,
                chunk_overlap=get_settings().CHUNK_OVERLAP,
            )
        )
        chunks = chunker.chunk_documents(raw_docs)
        logger.info(f"[Pipeline:retrieval_node] Generated {len(chunks)} chunks from raw documents")

        # 2. Embedding & Vector Store Upsert
        embedding_service = EmbeddingService(model=get_settings().EMBEDDING_MODEL)
        vector_store = QdrantVectorStore(
            url=get_settings().QDRANT_URL,
            api_key=get_settings().QDRANT_API_KEY,
            embedding_service=embedding_service,
        )

        try:
            # Initialize collection if not existing
            await vector_store.initialize()
            
            # Embed chunks
            embedded_pairs = await embedding_service.embed_documents(chunks)
            docs_to_upsert = [p[0] for p in embedded_pairs]
            vectors_to_upsert = [p[1] for p in embedded_pairs]

            if docs_to_upsert:
                await vector_store.upsert_documents(
                    documents=docs_to_upsert,
                    embeddings=vectors_to_upsert,
                    user_id=user_id,
                    session_id=session_id,
                )

            # 3. Hybrid Search (Dense + BM25 with RRF)
            hybrid_engine = HybridSearchEngine(
                vector_store=vector_store,
                embedding_service=embedding_service,
                bm25_weight=0.35,
                dense_weight=0.65,
            )
            hybrid_engine.build_bm25_index(chunks)

            hits = await hybrid_engine.search(
                query=topic,
                k=8,
                filter_user_id=user_id if user_id != "public" else None,
            )

            # 4. Optional Cohere Reranker
            if get_settings().COHERE_API_KEY:
                reranker = DocumentReranker(api_key=get_settings().COHERE_API_KEY)
                candidate_docs = [h[0] for h in hits]
                reranked_hits = await reranker.rerank(query=topic, documents=candidate_docs, top_n=6)
                for doc, score in reranked_hits:
                    retrieved_docs.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", ""),
                        "title": doc.metadata.get("title", ""),
                        "score": score,
                    })
            else:
                for doc, score in hits:
                    retrieved_docs.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", ""),
                        "title": doc.metadata.get("title", ""),
                        "score": score,
                    })

        except Exception as e:
            logger.warning(f"[Pipeline:retrieval_node] Vector store hybrid search fallback: {e}")
            # In-memory fallback if Qdrant is offline during dev/test
            for doc in chunks[:6]:
                retrieved_docs.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "title": doc.metadata.get("title", ""),
                    "score": 0.8,
                })

    elapsed = round(time.time() - start_time, 2)
    meta["retrieval_duration_sec"] = elapsed
    meta["retrieved_chunks_count"] = len(retrieved_docs)

    logger.info(f"[Pipeline:retrieval_node] Completed in {elapsed}s with {len(retrieved_docs)} grounded chunks")
    return {
        "retrieved_docs": retrieved_docs,
        "pipeline_metadata": meta,
        "messages": [f"Hybrid RAG engine retrieved {len(retrieved_docs)} grounded chunks."],
    }


# ---------------------------------------------------------------------------
# Node 4: Writer Node
# ---------------------------------------------------------------------------

async def writer_node(state: ResearchState) -> Dict[str, Any]:
    """Execute WriterAgent to draft or refine the research report."""
    topic = state["topic"]
    iteration = state.get("iteration", 0)
    critic_feedback = state.get("critic_feedback", "")
    search_results = state.get("search_results", [])
    scraped_content = state.get("scraped_content", [])
    retrieved_docs = state.get("retrieved_docs", [])
    start_time = time.time()

    logger.info(f"[Pipeline:writer_node] Writing report (iteration={iteration}) for topic: {topic!r}")

    writer = WriterAgent()
    draft = await writer.run(
        topic=topic,
        search_results=search_results,
        retrieved_docs=retrieved_docs,
        scraped_content=scraped_content,
        critic_feedback=critic_feedback,
        iteration=iteration,
    )

    elapsed = round(time.time() - start_time, 2)
    meta = dict(state.get("pipeline_metadata", {}))
    meta[f"writer_iter_{iteration}_duration_sec"] = elapsed

    return {
        "draft_report": draft,
        "iteration": iteration + 1,
        "pipeline_metadata": meta,
        "messages": [f"WriterAgent completed drafting iteration {iteration}."],
    }


# ---------------------------------------------------------------------------
# Node 5: Critic Node
# ---------------------------------------------------------------------------

async def critic_node(state: ResearchState) -> Dict[str, Any]:
    """Execute CriticAgent to evaluate report quality and decide if revision is needed."""
    topic = state["topic"]
    draft = state.get("draft_report", "")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 2)
    start_time = time.time()

    logger.info(f"[Pipeline:critic_node] Evaluating draft (length: {len(draft)}) for topic: {topic!r}")

    critic = CriticAgent()
    eval_res = await critic.run(report=draft, topic=topic)

    score = float(eval_res.get("score", 7.0))
    feedback = str(eval_res.get("feedback_text", ""))
    should_revise_flag = bool(eval_res.get("should_revise", False))

    # Enforce loop termination if reached max_iterations
    if iteration >= max_iterations:
        logger.info(f"[Pipeline:critic_node] Max iterations ({max_iterations}) reached; concluding revision.")
        should_revise_flag = False

    elapsed = round(time.time() - start_time, 2)
    meta = dict(state.get("pipeline_metadata", {}))
    meta[f"critic_iter_{iteration}_score"] = score
    meta[f"critic_iter_{iteration}_duration_sec"] = elapsed

    logger.info(f"[Pipeline:critic_node] Score: {score}/10, should_revise: {should_revise_flag}")
    return {
        "critic_score": score,
        "critic_feedback": feedback,
        "should_revise": should_revise_flag,
        "pipeline_metadata": meta,
        "messages": [f"Critic scored report {score:.1f}/10. Revision required: {should_revise_flag}."],
    }


# ---------------------------------------------------------------------------
# Conditional Edge Router
# ---------------------------------------------------------------------------

def should_revise(state: ResearchState) -> str:
    """Determine whether to route back for revision or proceed to final synthesis."""
    if state.get("should_revise", False) and state.get("iteration", 0) < state.get("max_iterations", 2):
        logger.info("[Router] Routing back to 'writer_node' for revision.")
        return "revise"
    logger.info("[Router] Quality approved or iteration limit reached. Routing to 'synthesis_node'.")
    return "finalize"


# ---------------------------------------------------------------------------
# Node 6: Synthesis Node
# ---------------------------------------------------------------------------

async def synthesis_node(state: ResearchState) -> Dict[str, Any]:
    """Execute SynthesisAgent to finalize, format, and cite the report."""
    topic = state["topic"]
    draft = state.get("draft_report", "")
    sources = state.get("sources", [])
    feedback = state.get("critic_feedback", "")
    start_time = time.time()

    logger.info(f"[Pipeline:synthesis_node] Polishing and finalizing report for: {topic!r}")

    synthesis = SynthesisAgent()
    final_report = await synthesis.run(
        draft_report=draft,
        sources=sources,
        topic=topic,
        critic_feedback=feedback,
    )

    elapsed = round(time.time() - start_time, 2)
    meta = dict(state.get("pipeline_metadata", {}))
    meta["synthesis_duration_sec"] = elapsed

    return {
        "final_report": final_report,
        "pipeline_metadata": meta,
        "messages": ["SynthesisAgent finalized publication-ready report."],
    }


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_research_graph():
    """
    Compile and return the stateful LangGraph research pipeline.
    """
    workflow = StateGraph(ResearchState)

    # Register nodes
    workflow.add_node("search_node", search_node)
    workflow.add_node("reader_node", reader_node)
    workflow.add_node("retrieval_node", retrieval_node)
    workflow.add_node("writer_node", writer_node)
    workflow.add_node("critic_node", critic_node)
    workflow.add_node("synthesis_node", synthesis_node)

    # Define sequential edges
    workflow.add_edge(START, "search_node")
    workflow.add_edge("search_node", "reader_node")
    workflow.add_edge("reader_node", "retrieval_node")
    workflow.add_edge("retrieval_node", "writer_node")
    workflow.add_edge("writer_node", "critic_node")

    # Define conditional evaluation loop
    workflow.add_conditional_edges(
        "critic_node",
        should_revise,
        {
            "revise": "writer_node",
            "finalize": "synthesis_node",
        },
    )

    # Final edge to finish
    workflow.add_edge("synthesis_node", END)

    # In-memory checkpointer for resumable stateful executions
    checkpointer = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    logger.info("LangGraph research workflow compiled successfully.")
    return compiled_graph

