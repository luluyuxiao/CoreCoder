"""CoreCoder - Minimal AI coding agent inspired by Claude Code's architecture."""

__version__ = "0.7.0"

from corecoder.agent import Agent
from corecoder.capabilities import CapabilityPolicy
from corecoder.config import Config
from corecoder.context import ContextOverflowError
from corecoder.llm import LLM, BudgetExceededError, ProviderRoute
from corecoder.memory import MemoryState
from corecoder.skills import Skill, SkillRegistry
from corecoder.storage import JsonSessionStore, SessionRecord, SessionStore, SQLiteSessionStore
from corecoder.tools import ALL_TOOLS, build_tools
from corecoder.trace import CompositeTrace, JsonlTrace, MemoryTrace, TraceSink

__all__ = [
    "ALL_TOOLS",
    "LLM",
    "Agent",
    "BudgetExceededError",
    "CapabilityPolicy",
    "CompositeTrace",
    "Config",
    "ContextOverflowError",
    "JsonSessionStore",
    "JsonlTrace",
    "MemoryState",
    "MemoryTrace",
    "ProviderRoute",
    "SQLiteSessionStore",
    "SessionRecord",
    "SessionStore",
    "Skill",
    "SkillRegistry",
    "TraceSink",
    "__version__",
    "build_tools",
]
