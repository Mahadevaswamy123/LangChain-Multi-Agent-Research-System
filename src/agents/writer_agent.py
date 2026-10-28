"""
Writer agent that generates structured, citation-rich research reports.

On the first iteration (iteration==0) a fresh report is drafted from the
available search results and RAG-retrieved docs.  On subsequent iterations
the agent revises the existing draft guided by the critic's feedback.
"""
from __future__ import annotations

from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agents.base_agent import BaseAgent
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

WRITER_SYSTEM_PROMPT = """You are an expert research analyst and technical writer with deep experience \
in producing rigorous, well-cited research reports.

Your reports are:
- Comprehensive: cover the topic from multiple angles
- Factual: grounded in the provided sources; never hallucinate
- Well-structured: clear headings, bullet points, and prose paragraphs
- Properly cited: every key claim references a source URL inline

When revising, carefully incorporate the critic's feedback while maintaining \
or improving overall quality. Do NOT simply repeat the previous draft."""

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_DRAFT_TEMPLATE = """\
Write a comprehensive research report on the following topic.

TOPIC: {topic}

---
SEARCH RESULTS (titles, URLs, and snippets):
{search_context}

---
RETRIEVED DOCUMENT CHUNKS (RAG context — use these for deep grounding):
{rag_context}

---
SCRAPED FULL CONTENT (deep content from top URLs):
{scraped_context}

---
REPORT REQUIREMENTS:
1. Executive Summary (3–5 sentences capturing the essence)
2. Key Findings (minimum 5 distinct findings, each with an inline citation)
3. In-Depth Analysis (discuss trends, implications, controversies)
4. Conclusion (synthesise findings, future outlook)
5. References (numbered list of all cited URLs)

Format the entire report as clean Markdown.
Inline citations should use the format: [Source Title](URL)
"""

_REVISION_TEMPLATE = """\
You are revising a research report based on critic feedback.

TOPIC: {topic}

---
PREVIOUS DRAFT:
{draft_report}

---
CRITIC FEEDBACK (incorporate ALL improvement points):
{critic_feedback}

---
ADDITIONAL RAG CONTEXT (use to strengthen weak sections):
{rag_context}

---
REVISION INSTRUCTIONS:
- Address every improvement point raised by the critic
- Strengthen citations where the critic flagged gaps
- Do NOT remove information that was praised as a strength
- Improve clarity and depth where requested
- Return the COMPLETE revised report in Markdown

Format: Executive Summary → Key Findings → Analysis → Conclusion → References
"""


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class WriterAgent(BaseAgent):
    """
    Generates or revises a structured research report in Markdown.

    Uses the LLM directly (no ReAct loop needed — pure generation) with
    a carefully engineered prompt that changes between the initial draft
    and revision passes.
    """

    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        super().__init__(llm=llm, name="WriterAgent")

    async def run(  # type: ignore[override]
        self,
        topic: str,
        search_results: List[dict],
        retrieved_docs: List[dict],
        scraped_content: List[dict],
        critic_feedback: str = "",
        iteration: int = 0,
        draft_report: str = "",
    ) -> str:
        """
        Generate or revise a comprehensive research report.

        Args:
            topic:           Research topic / question.
            search_results:  Structured results from SearchAgent.
            retrieved_docs:  Hybrid-search hits from Qdrant retrieval node.
            scraped_content: Full-page content from ReaderAgent.
            critic_feedback: Textual feedback from CriticAgent (empty on first pass).
            iteration:       Current refinement loop counter (0 = first draft).
            draft_report:    Existing draft to revise (empty on first pass).

        Returns:
            Markdown-formatted research report as a string.
        """
        self.logger.info(
            f"[WriterAgent] iteration={iteration} "
            f"revision={'yes' if iteration > 0 else 'no'}"
        )

        # ------------------------------------------------------------------ #
        # Build context strings                                               #
        # ------------------------------------------------------------------ #
        search_context = self._format_search_results(search_results)
        rag_context = self._format_rag_docs(retrieved_docs)
        scraped_context = self._format_scraped(scraped_content)

        # ------------------------------------------------------------------ #
        # Compose prompt                                                      #
        # ------------------------------------------------------------------ #
        if iteration == 0 or not draft_report:
            user_content = _DRAFT_TEMPLATE.format(
                topic=topic,
                search_context=search_context,
                rag_context=rag_context,
                scraped_context=scraped_context,
            )
        else:
            user_content = _REVISION_TEMPLATE.format(
                topic=topic,
                draft_report=draft_report,
                critic_feedback=critic_feedback,
                rag_context=rag_context,
            )

        messages = [
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            report: str = response.content.strip()
        except Exception as exc:
            self.logger.error(f"[WriterAgent] LLM call failed: {exc}", exc_info=True)
            raise

        self.logger.info(
            f"[WriterAgent] produced report: {len(report)} chars"
        )
        return report

    # ------------------------------------------------------------------
    # Private formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_search_results(results: List[dict]) -> str:
        if not results:
            return "No search results available."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. **{r.get('title', 'No title')}**\n"
                f"   URL: {r.get('url', '')}\n"
                f"   Snippet: {r.get('snippet', '')[:300]}\n"
                f"   Score: {r.get('score', 0.0):.2f}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _format_rag_docs(docs: List[dict]) -> str:
        if not docs:
            return "No RAG documents available."
        lines = []
        for i, doc in enumerate(docs, 1):
            content = doc.get("content", doc.get("page_content", ""))[:600]
            source = doc.get("source", doc.get("metadata", {}).get("source", "unknown"))
            lines.append(f"[Chunk {i} | Source: {source}]\n{content}")
        return "\n\n---\n\n".join(lines)

    @staticmethod
    def _format_scraped(scraped: List[dict]) -> str:
        if not scraped:
            return "No scraped content available."
        lines = []
        for item in scraped:
            title = item.get("title", item.get("url", ""))
            content = item.get("content", "")[:1500]
            url = item.get("url", "")
            lines.append(f"### {title}\nURL: {url}\n\n{content}")
        return "\n\n---\n\n".join(lines)
