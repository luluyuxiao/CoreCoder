"""CoreCoder - Minimal AI coding agent inspired by Claude Code's architecture."""

__version__ = "0.7.0"

from corecoder.agent import Agent
from corecoder.config import Config
from corecoder.context import ContextOverflowError
from corecoder.llm import LLM, BudgetExceededError
from corecoder.tools import ALL_TOOLS, build_tools
from corecoder.trace import CompositeTrace, JsonlTrace, MemoryTrace, TraceSink

__all__ = [
    "ALL_TOOLS",
    "LLM",
    "Agent",
    "BudgetExceededError",
    "CompositeTrace",
    "Config",
    "ContextOverflowError",
    "JsonlTrace",
    "MemoryTrace",
    "TraceSink",
    "__version__",
    "build_tools",
]
