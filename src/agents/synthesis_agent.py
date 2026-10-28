"""
Synthesis agent that formats, polishes, and finalizes the research report.

The SynthesisAgent takes the final draft, verifies source citations, structures
a standardized references section, sharpens executive summary readability,
and appends formal metadata footers for production-ready reporting.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agents.base_agent import BaseAgent
from src.utils.logging import get_logger

logger = get_logger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are a senior publishing editor and research director.
Your objective is to take an evaluated draft research report and transform it into a
pristine, publication-ready, executive-level deliverable.

Ensure that:
1. The title is commanding and professional.
2. The Executive Summary is impactful, crisp, and data-driven.
3. Headings, bullet points, and tables are consistently formatted in GitHub-Flavored Markdown.
4. Claims have inline citations referencing source indices or URLs.
5. All references provided are cleanly listed at the end under a 'References & Sources' section.
6. The tone is authoritative, objective, and analytical.
"""

_SYNTHESIS_TEMPLATE = """Transform and polish the following draft research report into a final executive deliverable.

TOPIC: {topic}

DRAFT REPORT:
\"\"\"
{draft_report}
\"\"\"

VERIFIED SOURCE URLS:
{sources_text}

CRITIC FEEDBACK CONSIDERED:
{critic_feedback}

INSTRUCTIONS:
1. Retain all valuable technical depth and findings from the draft.
2. Ensure markdown formatting is flawless with appropriate visual rhythm (subheadings, lists, bold highlights).
3. Ensure the 'References & Sources' section at the end contains all verified URLs with descriptive titles.
4. Output the complete, polished Markdown document.
"""


class SynthesisAgent(BaseAgent):
    """
    Final synthesis agent that polishes the research deliverable and formats sources.
    """

    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        super().__init__(llm=llm, name="SynthesisAgent")

    async def run(  # type: ignore[override]
        self,
        draft_report: str,
        sources: List[str],
        topic: str,
        critic_feedback: str = "",
    ) -> str:
        """
        Synthesize and finalize the report.

        Args:
            draft_report:    Markdown draft report from WriterAgent.
            sources:         List of verified source URLs.
            topic:           The research query.
            critic_feedback: Critique summary.

        Returns:
            Polished final Markdown string.
        """
        self.logger.info(f"[SynthesisAgent] synthesizing final report for topic: {topic!r}")

        unique_sources = list(dict.fromkeys(s for s in sources if s and s.startswith("http")))
        sources_text = "\n".join(f"- {s}" for s in unique_sources) if unique_sources else "No explicit URLs tracked."

        prompt = _SYNTHESIS_TEMPLATE.format(
            topic=topic,
            draft_report=draft_report[:15000],
            sources_text=sources_text,
            critic_feedback=critic_feedback[:2000] if critic_feedback else "None provided.",
        )

        messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            final_content = str(getattr(response, "content", "")).strip()
        except Exception as e:
            self.logger.error(f"[SynthesisAgent] LLM invocation failed: {e}", exc_info=True)
            final_content = draft_report

        # Ensure a clean metadata footer is present
        timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        footer = (
            f"\n\n---\n"
            f"*Report synthesized autonomously by Multi-Agent Research System | "
            f"Generated: {timestamp_str} | Verified Sources: {len(unique_sources)}*"
        )

        if "Report synthesized autonomously" not in final_content:
            final_content += footer

        self.logger.info(f"[SynthesisAgent] final report generated ({len(final_content)} chars)")
        return final_content

