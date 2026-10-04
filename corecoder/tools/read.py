"""File reading with line numbers."""

from pathlib import Path
from typing import ClassVar

from ..capabilities import FILESYSTEM_READ
from ..resources import ResourceClaim
from ..sandbox import WorkspacePathPolicy
from .base import Tool, ToolEffect


class ReadFileTool(Tool):
    name = "read_file"
    effect = ToolEffect.READ
    capabilities = frozenset({FILESYSTEM_READ})
    description = (
        "Read a file's contents with line numbers. "
        "Always read a file before editing it."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file",
            },
            "offset": {
                "type": "integer",
                "description": "Start line (1-based). Default 1.",
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to read. Default 2000.",
            },
        },
        "required": ["file_path"],
    }

    def __init__(self, path_policy: WorkspacePathPolicy | None = None):
        self.path_policy = path_policy

    def _path(self, file_path: str) -> Path:
        return (
            self.path_policy.resolve(file_path)
            if self.path_policy else Path(file_path).expanduser().resolve()
        )

    def resource_claims(self, arguments: dict) -> tuple[ResourceClaim, ...]:
        raw = arguments.get("file_path")
        if not isinstance(raw, str) or not raw:
            return super().resource_claims(arguments)
        try:
            path = self._path(raw)
        except (OSError, ValueError):
            return super().resource_claims(arguments)
        return (ResourceClaim(f"file:{path}", "read"),)

    def execute(self, file_path: str, offset: int = 1, limit: int = 2000) -> str:
        try:
            p = self._path(file_path)
            if not p.exists():
                return f"Error: {file_path} not found"
            if not p.is_file():
                return f"Error: {file_path} is a directory, not a file"

            text = p.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            total = len(lines)

            start = max(0, offset - 1)
            chunk = lines[start : start + limit]
            numbered = [f"{start + i + 1}\t{ln}" for i, ln in enumerate(chunk)]
            result = "\n".join(numbered)

            if total > start + limit:
                result += f"\n... ({total} lines total, showing {start+1}-{start+len(chunk)})"
            return result or "(empty file)"
        except Exception as e:  # noqa: BLE001
            # boundary: the agent gets an error string, not a traceback
            return f"Error: {e}"
