"""Tool registry."""

from ..sandbox import CommandExecutor, WorkspacePathPolicy
from .agent import AgentStatusTool, AgentTool
from .bash import BashTool
from .edit import EditFileTool
from .fetch import FetchUrlTool
from .glob_tool import GlobTool
from .grep import GrepTool
from .now import NowTool
from .read import ReadFileTool
from .todo import TodoWriteTool
from .write import WriteFileTool


def build_tools(
    executor: CommandExecutor | None = None,
    path_policy: WorkspacePathPolicy | None = None,
):
    """Create a fresh tool set, optionally bound to a sandbox policy."""
    return [
        BashTool(executor=executor),
        ReadFileTool(path_policy=path_policy),
        WriteFileTool(path_policy=path_policy),
        EditFileTool(path_policy=path_policy),
        GlobTool(path_policy=path_policy),
        GrepTool(path_policy=path_policy),
        TodoWriteTool(),
        AgentTool(),
        AgentStatusTool(),
        FetchUrlTool(),
        NowTool(),
    ]


# Backwards-compatible default registry. The CLI uses build_tools() so each
# Agent gets fresh state and the selected execution backend.
ALL_TOOLS = build_tools()
