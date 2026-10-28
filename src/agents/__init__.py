"""Agents module exporting production agents for the Multi-Agent Research System."""
from src.agents.base_agent import BaseAgent
from src.agents.search_agent import SearchAgent
from src.agents.reader_agent import ReaderAgent
from src.agents.writer_agent import WriterAgent
from src.agents.critic_agent import CriticAgent
from src.agents.synthesis_agent import SynthesisAgent

__all__ = [
    "BaseAgent",
    "SearchAgent",
    "ReaderAgent",
    "WriterAgent",
    "CriticAgent",
    "SynthesisAgent",
]

