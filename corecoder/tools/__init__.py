"""Tool registry."""

from typing import TYPE_CHECKING

from ..checkpoints import CheckpointManager
from ..sandbox import CommandExecutor, WorkspacePathPolicy
from .agent import AgentResumeTool, AgentStatusTool, AgentTool
from .bash import BashTool
from .edit import EditFileTool
from .fetch import FetchUrlTool
from .glob_tool import GlobTool
from .grep import GrepTool
from .memory import MemoryUpdateTool
from .now import NowTool
from .read import ReadFileTool
from .skill import LoadSkillTool
from .todo import TodoWriteTool
from .write import WriteFileTool

if TYPE_CHECKING:
    from ..skills import SkillRegistry


def build_tools(
    executor: CommandExecutor | None = None,
    path_policy: WorkspacePathPolicy | None = None,
    skill_registry: "SkillRegistry | None" = None,
    checkpoint_manager: CheckpointManager | None = None,
):
    """Create a fresh tool set, optionally bound to a sandbox policy."""
    checkpoints = checkpoint_manager or CheckpointManager()
    tools = [
        BashTool(executor=executor),
        ReadFileTool(path_policy=path_policy),
        WriteFileTool(path_policy=path_policy, checkpoint_manager=checkpoints),
        EditFileTool(path_policy=path_policy, checkpoint_manager=checkpoints),
        GlobTool(path_policy=path_policy),
        GrepTool(path_policy=path_policy),
        TodoWriteTool(),
        AgentTool(),
        AgentStatusTool(),
        AgentResumeTool(),
        MemoryUpdateTool(),
        FetchUrlTool(),
        NowTool(),
    ]
    if skill_registry:
        tools.append(LoadSkillTool(skill_registry))
    return tools


# Backwards-compatible default registry. The CLI uses build_tools() so each
# Agent gets fresh state and the selected execution backend.
ALL_TOOLS = build_tools()
