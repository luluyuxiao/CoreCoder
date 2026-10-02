"""Content search with regex support."""

import re
from pathlib import Path
from typing import ClassVar

from ..capabilities import FILESYSTEM_READ
from ..sandbox import WorkspacePathPolicy
from .base import Tool, ToolEffect

# skip these dirs to avoid noise
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", "dist", "build"}


class GrepTool(Tool):
    name = "grep"
    effect = ToolEffect.READ
    capabilities = frozenset({FILESYSTEM_READ})
    description = (
        "Search file contents with regex. "
        "Returns matching lines with file path and line number."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search (default: cwd)",
            },
            "include": {
                "type": "string",
                "description": "Only search files matching this glob (e.g. '*.py')",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, path_policy: WorkspacePathPolicy | None = None):
        self.path_policy = path_policy

    def execute(self, pattern: str, path: str = ".", include: str | None = None) -> str:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex: {e}"

        if self.path_policy and include and (
            Path(include).is_absolute() or ".." in Path(include).parts
        ):
            return "Error: include pattern may not escape the sandbox workspace"

        try:
            base = (
                self.path_policy.resolve(path)
                if self.path_policy
                else Path(path).expanduser().resolve()
            )
        except ValueError as e:
            return f"Error: {e}"
        if not base.exists():
            return f"Error: {path} not found"

        if base.is_file():
            files = [base]
            scan_truncated = False
        else:
            files, scan_truncated = self._walk(base, include)
            if self.path_policy:
                files = [fp for fp in files if self.path_policy.contains(fp)]

        scan_limit_msg = "... (5000 file scan limit reached; results may be incomplete)"
        matches = []
        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{fp}:{lineno}: {line.rstrip()}")
                    if len(matches) >= 200:
                        matches.append("... (200 match limit reached)")
                        if scan_truncated:
                            matches.append(scan_limit_msg)
                        return "\n".join(matches)

        if matches:
            if scan_truncated:
                matches.append(scan_limit_msg)
            return "\n".join(matches)
        if scan_truncated:
            return f"No matches found in scanned files.\n{scan_limit_msg}"
        return "No matches found."

    @staticmethod
    def _walk(root: Path, include: str | None) -> tuple[list[Path], bool]:
        """Walk dir tree, skipping junk dirs."""
        results = []
        truncated = False
        for item in root.rglob(include or "*"):
            # skip junk dirs *inside* the search root - matching item.parts would
            # also catch an ancestor named e.g. "build" and hide the whole tree
            if any(part in _SKIP_DIRS for part in item.relative_to(root).parts):
                continue
            if item.is_file():
                results.append(item)
            if len(results) >= 5000:
                truncated = True
                break
        return results, truncated
