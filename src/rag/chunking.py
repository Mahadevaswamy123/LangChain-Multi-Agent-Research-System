"""
Production chunking module supporting multiple strategies:
- RecursiveCharacterTextSplitter (default, good for general text)
- SemanticChunker using sentence embeddings (splits at semantic boundaries)
- SentenceSplitter with sliding window overlap
- TokenTextSplitter for exact token control
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Enumerations & configuration
# ---------------------------------------------------------------------------

class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"
    TOKEN = "token"


@dataclass
class ChunkConfig:
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100  # discard tiny chunks that add noise


# ---------------------------------------------------------------------------
# DocumentChunker
# ---------------------------------------------------------------------------

class DocumentChunker:
    """
    Production-grade document chunker with metadata preservation.

    All splitting strategies preserve the original document metadata and
    augment each chunk with rich provenance fields:
        - source         : original document source URL / path
        - chunk_index    : 0-based index within the parent document
        - total_chunks   : total number of chunks produced from that parent
        - char_start     : character offset where the chunk starts (approximate)
        - char_end       : character offset where the chunk ends (approximate)
        - chunk_strategy : strategy name used
    """

    def __init__(self, config: Optional[ChunkConfig] = None) -> None:
        self.config = config or ChunkConfig()
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
            strip_whitespace=True,
        )
        self._token_splitter = TokenTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        logger.info(
            f"DocumentChunker initialised — strategy={self.config.strategy.value}, "
            f"chunk_size={self.config.chunk_size}, overlap={self.config.chunk_overlap}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk a list of LangChain Documents and enrich each chunk with
        rich metadata (source, chunk_index, total_chunks, char_start,
        char_end, chunk_strategy).

        Args:
            documents: Raw LangChain Document objects.

        Returns:
            Flat list of chunked Document objects with augmented metadata.
        """
        if not documents:
            logger.warning("chunk_documents called with empty list — returning []")
            return []

        logger.info(
            f"Chunking {len(documents)} document(s) using strategy={self.config.strategy.value}"
        )

        strategy_map = {
            ChunkStrategy.RECURSIVE: self._recursive_chunk,
            ChunkStrategy.SEMANTIC: self._semantic_chunk,
            ChunkStrategy.SENTENCE: self._sentence_chunk,
            ChunkStrategy.TOKEN: self._token_chunk,
        }

        try:
            chunks = strategy_map[self.config.strategy](documents)
        except Exception as exc:
            logger.error(
                f"Chunking failed with strategy={self.config.strategy.value}: {exc}. "
                "Falling back to recursive strategy."
            )
            chunks = self._recursive_chunk(documents)

        # Filter out tiny chunks that are likely boilerplate
        before = len(chunks)
        chunks = [c for c in chunks if len(c.page_content.strip()) >= self.config.min_chunk_size]
        removed = before - len(chunks)
        if removed:
            logger.debug(f"Removed {removed} chunk(s) below min_chunk_size={self.config.min_chunk_size}")

        enriched = self._add_chunk_metadata(chunks, documents)
        logger.info(f"Produced {len(enriched)} chunk(s) from {len(documents)} document(s)")
        return enriched

    def chunk_text(
        self,
        text: str,
        source_url: str = "",
        metadata: Optional[dict] = None,
    ) -> List[Document]:
        """
        Convenience wrapper — chunk a raw string and wrap results in
        LangChain Documents with auto metadata.

        Args:
            text:       Raw text to chunk.
            source_url: Source URL or file path for provenance.
            metadata:   Additional metadata to merge onto every chunk.

        Returns:
            List of chunked Document objects.
        """
        if metadata is None:
            metadata = {}

        doc = Document(
            page_content=text,
            metadata={"source": source_url, **metadata},
        )
        return self.chunk_documents([doc])

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _recursive_chunk(self, docs: List[Document]) -> List[Document]:
        """
        LangChain RecursiveCharacterTextSplitter — best general-purpose
        strategy that respects paragraph / sentence / word boundaries
        in that order.
        """
        logger.debug("Running recursive character chunking")
        return self._recursive_splitter.split_documents(docs)

    def _semantic_chunk(self, docs: List[Document]) -> List[Document]:
        """
        Semantic chunking using sentence-level cosine similarity to find
        natural breakpoints in the text.

        Attempts to import langchain_experimental.text_splitter.SemanticChunker
        with OpenAI embeddings. Falls back to recursive chunking when the
        dependency is unavailable.
        """
        logger.debug("Running semantic chunking")
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95,
            )
            chunks: List[Document] = []
            for doc in docs:
                try:
                    doc_chunks = splitter.create_documents(
                        [doc.page_content],
                        metadatas=[doc.metadata],
                    )
                    chunks.extend(doc_chunks)
                except Exception as doc_exc:
                    logger.warning(
                        f"SemanticChunker failed for doc source="
                        f"'{doc.metadata.get('source', 'unknown')}': {doc_exc}. "
                        "Falling back to recursive for this document."
                    )
                    chunks.extend(self._recursive_splitter.split_documents([doc]))
            return chunks
        except ImportError:
            logger.warning(
                "langchain_experimental not installed — falling back to recursive chunking. "
                "Install with: pip install langchain-experimental"
            )
            return self._recursive_chunk(docs)

    def _sentence_chunk(self, docs: List[Document]) -> List[Document]:
        """
        Sentence-boundary chunking with a sliding window overlap using NLTK.

        Splits text into sentences first, then groups them into chunks of
        ~chunk_size characters using a sliding window approach so contextual
        continuity is preserved across chunk boundaries.
        """
        logger.debug("Running sentence-window chunking")
        import nltk

        # Download tokenizer data if not already present
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            try:
                nltk.download("punkt", quiet=True)
            except Exception:
                pass

        chunks: List[Document] = []

        for doc in docs:
            try:
                sentences = nltk.sent_tokenize(doc.page_content)
            except Exception as exc:
                logger.warning(
                    f"NLTK sent_tokenize failed: {exc}. Splitting on '. ' instead."
                )
                sentences = re.split(r"(?<=[.!?])\s+", doc.page_content)

            if not sentences:
                continue

            current_chunk_sentences: List[str] = []
            current_len = 0
            all_sentence_groups: List[List[str]] = []

            for sentence in sentences:
                sentence_len = len(sentence)
                if current_len + sentence_len > self.config.chunk_size and current_chunk_sentences:
                    all_sentence_groups.append(list(current_chunk_sentences))
                    # Sliding window: keep the last N sentences for overlap
                    overlap_chars = 0
                    overlap_sentences: List[str] = []
                    for s in reversed(current_chunk_sentences):
                        if overlap_chars + len(s) <= self.config.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_chars += len(s)
                        else:
                            break
                    current_chunk_sentences = overlap_sentences
                    current_len = overlap_chars

                current_chunk_sentences.append(sentence)
                current_len += sentence_len

            if current_chunk_sentences:
                all_sentence_groups.append(current_chunk_sentences)

            for group in all_sentence_groups:
                text = " ".join(group)
                chunks.append(
                    Document(
                        page_content=text,
                        metadata=dict(doc.metadata),
                    )
                )

        return chunks

    def _token_chunk(self, docs: List[Document]) -> List[Document]:
        """
        Token-aware chunking using LangChain's TokenTextSplitter (tiktoken).
        Provides exact control over the number of tokens per chunk.
        """
        logger.debug("Running token-based chunking")
        return self._token_splitter.split_documents(docs)

    # ------------------------------------------------------------------
    # Metadata enrichment
    # ------------------------------------------------------------------

    def _add_chunk_metadata(
        self,
        chunks: List[Document],
        source_docs: List[Document],
    ) -> List[Document]:
        """
        Add provenance metadata to each chunk.

        Assigns:
            - ``source``          : from parent document metadata
            - ``chunk_index``     : 0-based index within parent doc's chunks
            - ``total_chunks``    : total chunks produced from this source
            - ``char_start``      : approximate start char offset in original text
            - ``char_end``        : approximate end char offset in original text
            - ``chunk_strategy``  : strategy name
        """
        # Build a fast lookup: source text → list of chunks (in order)
        # Group chunks by their declared source to compute chunk indices
        source_to_chunks: dict[str, List[Document]] = {}
        for chunk in chunks:
            src = chunk.metadata.get("source", "unknown")
            source_to_chunks.setdefault(src, []).append(chunk)

        # Build full-text lookup for char offset estimation
        source_to_fulltext: dict[str, str] = {
            doc.metadata.get("source", "unknown"): doc.page_content
            for doc in source_docs
        }

        enriched: List[Document] = []
        # Preserve global ordering — iterate source groups in insertion order
        for src, src_chunks in source_to_chunks.items():
            full_text = source_to_fulltext.get(src, "")
            total = len(src_chunks)
            search_start = 0  # cursor for char offset estimation

            for idx, chunk in enumerate(src_chunks):
                content = chunk.page_content

                # Estimate character offsets within original text
                char_start = full_text.find(content[:min(40, len(content))], search_start)
                if char_start == -1:
                    char_start = search_start
                char_end = char_start + len(content)
                search_start = max(search_start, char_start + 1)

                new_metadata = {
                    **chunk.metadata,
                    "source": src,
                    "chunk_index": idx,
                    "total_chunks": total,
                    "char_start": char_start,
                    "char_end": char_end,
                    "chunk_strategy": self.config.strategy.value,
                }
                enriched.append(
                    Document(page_content=content, metadata=new_metadata)
                )

        return enriched
