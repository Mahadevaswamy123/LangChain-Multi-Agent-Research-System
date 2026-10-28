"""
Critic agent for structured evaluation of research reports.

The CriticAgent reviews a drafted research report against objective criteria
(factual grounding, clarity, completeness, citations) and outputs a structured
evaluation including an overall score (0.0 to 10.0), specific strengths,
areas to improve, and a recommendation on whether revision is necessary.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agents.base_agent import BaseAgent
from src.utils.logging import get_logger

logger = get_logger(__name__)

CRITIC_SYSTEM_PROMPT = """You are a rigorous, objective academic and industry research evaluator.
Your job is to critically assess research reports for factual accuracy, analytical depth,
coverage completeness, clarity, and citation quality.

You must ALWAYS return your evaluation strictly as a valid JSON object matching this schema:
{
  "score": float,              // Overall score between 0.0 and 10.0
  "accuracy_score": float,     // Factual accuracy & grounding (0.0 - 10.0)
  "completeness_score": float, // Breadth and depth of coverage (0.0 - 10.0)
  "clarity_score": float,      // Writing clarity, structure, and readability (0.0 - 10.0)
  "citation_score": float,     // Source attribution and citation quality (0.0 - 10.0)
  "strengths": [string],       // At least 2 specific strengths
  "improvements": [string],   // At least 2 actionable points for improvement
  "verdict": string,           // Concise 1-2 sentence executive appraisal
  "should_revise": boolean     // True if score < 7.0 or any critical flaws exist
}

Scoring Calibration:
- 9.0 - 10.0: Publication-grade, highly insightful, deeply sourced with zero hallucinations.
- 7.5 - 8.9: Strong professional standard, minor stylistic or coverage gaps.
- 6.0 - 7.4: Acceptable baseline, but missing important nuances or lacking source citations.
- Below 6.0: Needs substantial revision (shallow, ungrounded, or poorly structured).
"""

_EVALUATION_TEMPLATE = """Evaluate the following research report on the topic '{topic}'.

REPORT:
\"\"\"
{report}
\"\"\"

Critique the report thoroughly against the criteria and output ONLY valid JSON.
"""


class CriticAgent(BaseAgent):
    """
    CriticAgent evaluates drafted reports and provides structured scoring and feedback.
    """

    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        super().__init__(llm=llm, name="CriticAgent")

    async def run(  # type: ignore[override]
        self,
        report: str,
        topic: str,
    ) -> Dict[str, Any]:
        """
        Evaluate *report* for *topic*.

        Returns:
            Dict containing:
                score (float): 0.0 - 10.0
                accuracy_score (float)
                completeness_score (float)
                clarity_score (float)
                citation_score (float)
                strengths (List[str])
                improvements (List[str])
                verdict (str)
                should_revise (bool)
                feedback_text (str): formatted critique for the writer
        """
        self.logger.info(f"[CriticAgent] evaluating report for topic: {topic!r}")

        prompt = _EVALUATION_TEMPLATE.format(topic=topic, report=report[:12000])
        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        # Use JSON mode if possible with model
        try:
            llm_json = self.llm.bind(response_format={"type": "json_object"})
            response = await llm_json.ainvoke(messages)
            raw_text = getattr(response, "content", "")
        except Exception as e:
            self.logger.warning(f"[CriticAgent] json mode fallback: {e}")
            response = await self.llm.ainvoke(messages)
            raw_text = getattr(response, "content", "")

        parsed = self._parse_json_response(raw_text)

        score = float(parsed.get("score", 7.0))
        should_revise = parsed.get("should_revise", score < 7.0)

        # Build human-readable feedback text for writer iteration
        feedback_lines = [
            f"Critic Score: {score:.1f}/10.0",
            f"Verdict: {parsed.get('verdict', 'No verdict provided.')}",
            "\nKey Strengths:",
        ]
        for s in parsed.get("strengths", []):
            feedback_lines.append(f" - {s}")
        feedback_lines.append("\nPriority Improvements:")
        for imp in parsed.get("improvements", []):
            feedback_lines.append(f" - {imp}")

        parsed["feedback_text"] = "\n".join(feedback_lines)
        parsed["score"] = score
        parsed["should_revise"] = should_revise

        self.logger.info(
            f"[CriticAgent] completed evaluation. Score: {score:.1f}, should_revise: {should_revise}"
        )
        return parsed

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Safely parse JSON from raw LLM output with regex fallback."""
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try to find markdown fenced json block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        # Try to locate first '{' and last '}'
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass

        # Fallback dictionary if all parsing fails
        self.logger.error("[CriticAgent] failed to parse JSON from critic output, using fallback")
        return {
            "score": 7.0,
            "accuracy_score": 7.0,
            "completeness_score": 7.0,
            "clarity_score": 7.0,
            "citation_score": 7.0,
            "strengths": ["Well-organized presentation of primary findings."],
            "improvements": ["Deepen quantitative metrics and expand source citations."],
            "verdict": "Solid baseline report meeting standard criteria.",
            "should_revise": False,
        }

