"""File creation / overwrite."""

from pathlib import Path
from typing import ClassVar

from ..capabilities import FILESYSTEM_WRITE
from ..checkpoints import DEFAULT_MANAGER, CheckpointManager
from ..resources import ResourceClaim
from ..sandbox import WorkspacePathPolicy
from .base import Tool, ToolEffect
from .edit import _changed_files


class WriteFileTool(Tool):
    name = "write_file"
    effect = ToolEffect.WRITE
    resource_parallel = True
    capabilities = frozenset({FILESYSTEM_WRITE})
    description = (
        "Create a new file or completely overwrite an existing one. "
        "For small edits to existing files, prefer edit_file instead."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path for the file",
            },
            "content": {
                "type": "string",
                "description": "Full file content to write",
            },
        },
        "required": ["file_path", "content"],
    }

    def __init__(
        self,
        path_policy: WorkspacePathPolicy | None = None,
        checkpoint_manager: CheckpointManager | None = None,
    ):
        self.path_policy = path_policy
        self.checkpoints = checkpoint_manager or DEFAULT_MANAGER

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
        return (ResourceClaim(f"file:{path}", "write"),)

    def execute(self, file_path: str, content: str) -> str:
        try:
            p = self._path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self.checkpoints.record(p)
            p.write_text(content, encoding="utf-8")
            _changed_files.add(str(p))
            n_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return f"Wrote {n_lines} lines to {file_path}"
        except Exception as e:  # noqa: BLE001
            # boundary: the agent gets an error string, not a traceback
            return f"Error: {e}"
