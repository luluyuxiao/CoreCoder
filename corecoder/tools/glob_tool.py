"""File pattern matching."""

from pathlib import Path
from typing import ClassVar

from ..sandbox import WorkspacePathPolicy
from .base import Tool, ToolEffect


class GlobTool(Tool):
    name = "glob"
    effect = ToolEffect.READ
    description = (
        "Find files matching a glob pattern. "
        "Supports ** for recursive matching (e.g. '**/*.py')."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: cwd)",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, path_policy: WorkspacePathPolicy | None = None):
        self.path_policy = path_policy

    def execute(self, pattern: str, path: str = ".") -> str:
        try:
            if self.path_policy and (Path(pattern).is_absolute() or ".." in Path(pattern).parts):
                return "Error: glob pattern may not escape the sandbox workspace"
            base = (
                self.path_policy.resolve(path)
                if self.path_policy
                else Path(path).expanduser().resolve()
            )
            if not base.exists():
                return f"Error: {path} not found"
            if not base.is_dir():
                return f"Error: {path} is not a directory"

            hits = list(base.glob(pattern))
            if self.path_policy:
                hits = [hit for hit in hits if self.path_policy.contains(hit)]
            # sort by mtime, newest first
            hits.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

            total = len(hits)
            shown = hits[:100]
            lines = [str(h) for h in shown]
            result = "\n".join(lines)

            if total > 100:
                result += f"\n... ({total} matches, showing first 100)"
            return result or "No files matched."
        except Exception as e:  # noqa: BLE001
            # boundary: the agent gets an error string, not a traceback
            return f"Error: {e}"
