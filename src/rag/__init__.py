"""
RAG subsystem public API.

All core classes are re-exported here for clean single-import access:

    from src.rag import (
        DocumentChunker, ChunkConfig, ChunkStrategy,
        EmbeddingService,
        QdrantVectorStore,
        HybridSearchEngine,
        DocumentReranker,
    )
"""
from src.rag.chunking import ChunkConfig, ChunkStrategy, DocumentChunker
from src.rag.embeddings import EmbeddingService
from src.rag.hybrid_search import HybridSearchEngine
from src.rag.reranker import DocumentReranker
from src.rag.vector_store import QdrantVectorStore

__all__ = [
    # Chunking
    "DocumentChunker",
    "ChunkConfig",
    "ChunkStrategy",
    # Embeddings
    "EmbeddingService",
    # Vector store
    "QdrantVectorStore",
    # Hybrid search
    "HybridSearchEngine",
    # Reranking
    "DocumentReranker",
]
