"""Shell command execution with safety checks.

Claude Code's BashTool is 1,143 lines. This is the distilled version:
- Output capture with truncation (head+tail preserved)
- Timeout support
- Dangerous command detection
- Working directory tracking (cd awareness)
"""

import os
import re
import subprocess
import threading
from typing import ClassVar

from ..capabilities import FILESYSTEM_READ, FILESYSTEM_WRITE, NETWORK, PROCESS
from ..sandbox import CommandExecutor, LocalCommandExecutor, SandboxViolation
from .base import Tool, ToolEffect

# Track cwd across commands (Claude Code does this too). Thread-local, so that
# when the agent executes tools in parallel two bash calls never race on one
# shared global: each worker thread carries its own cwd. See article 05.
_local = threading.local()

# patterns that could wreck the filesystem or leak secrets
_DANGEROUS_PATTERNS = [
    # recursive delete aimed at root/home (force flag optional)
    (
        r"\brm\b(?=[^;&|\n]*\s-[^\s]*[rR])[^;&|\n]*\s(?:/|~|\$HOME)(?:\s|$)",
        "recursive delete on home/root",
    ),
    # recursive (-r/-R) and force (-f) flags together, in any order or spacing
    (r"\brm\b(?=(?:.*\s)?-\w*[rR])(?=(?:.*\s)?-\w*f)", "force recursive delete"),
    # the same, written with long-form flags
    (r"\brm\b.*--recursive\b.*--force\b|\brm\b.*--force\b.*--recursive\b", "force recursive delete"),
    (r"\bmkfs\b", "format filesystem"),
    (r"\bdd\s+.*of=/dev/", "raw disk write"),
    (r">\s*/dev/sd[a-z]", "overwrite block device"),
    (r"\bchmod\s+(-R\s+)?777\s+/", "chmod 777 on root"),
    (r":\(\)\s*\{.*:\|:.*\}", "fork bomb"),
    (r"\bcurl\b.*\|\s*(sudo\s+)?(ba)?sh\b", "pipe curl to shell"),
    (r"\bwget\b.*\|\s*(sudo\s+)?(ba)?sh\b", "pipe wget to shell"),
]


class BashTool(Tool):
    name = "bash"
    effect = ToolEffect.EXTERNAL
    description = (
        "Execute a shell command. Returns stdout, stderr, and exit code. "
        "Use this for running tests, installing packages, git operations, etc."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 120)",
            },
        },
        "required": ["command"],
    }

    def __init__(self, executor: CommandExecutor | None = None):
        self.executor = executor or LocalCommandExecutor()
        capabilities = {FILESYSTEM_READ, FILESYSTEM_WRITE, PROCESS}
        # Local commands can always attempt network access. Docker removes that
        # capability only when its enforced network mode is exactly ``none``.
        if self.executor.mode != "docker" or getattr(self.executor, "network", None) != "none":
            capabilities.add(NETWORK)
        self.capabilities = frozenset(capabilities)
        # Cwd is isolated both by BashTool instance and by worker thread.  This
        # prevents separate Agents in the same Python process from inheriting
        # one another's last `cd`, while retaining safe parallel tool calls.
        self._local = threading.local()
        if self.executor.mode == "docker":
            self.description = (
                type(self).description
                + " Commands run in an isolated Docker container; the project root is /workspace "
                "and is the only host directory mounted. Prefer relative paths."
            )

    def execute(self, command: str, timeout: int = 120) -> str:
        # safety check
        warning = _check_dangerous(command)
        if warning:
            return f"⚠ Blocked: {warning}\nCommand: {command}\nIf intentional, modify the command to be more specific."

        # use this thread's own tracked working directory
        default_cwd = getattr(self.executor, "workspace_root", None) or os.getcwd()
        cwd = getattr(self._local, "cwd", None) or str(default_cwd)

        try:
            proc = self.executor.run(command, cwd=cwd, timeout=timeout)

            # track cd commands so next command runs in the right place
            if proc.returncode == 0:
                _update_cwd(
                    command,
                    cwd,
                    getattr(self.executor, "workspace_root", None),
                    getattr(self.executor, "container_workspace", None),
                    self._local,
                )
            out = proc.stdout
            if proc.stderr:
                out += f"\n[stderr]\n{proc.stderr}"
            if proc.returncode != 0:
                out += f"\n[exit code: {proc.returncode}]"
            # keep head + tail to preserve the most useful info
            if len(out) > 15_000:
                out = (
                    out[:6000]
                    + f"\n\n... truncated ({len(out)} chars total) ...\n\n"
                    + out[-3000:]
                )
            return out.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: timed out after {timeout}s"
        except SandboxViolation as e:
            return f"Sandbox violation: {e}"
        except Exception as e:  # noqa: BLE001
            # anything else from the OS (spawn failure etc.) also comes back as text
            return f"Error running command: {e}"


def _check_dangerous(cmd: str) -> str | None:
    """Return a warning string if the command looks destructive, else None."""
    for pattern, reason in _DANGEROUS_PATTERNS:
        if re.search(pattern, cmd):
            return reason
    return None


def _update_cwd(
    command: str,
    current_cwd: str,
    workspace_root=None,
    container_workspace=None,
    state=None,
):
    """Track directory changes from cd commands, per thread."""
    state = state or _local
    # walk each cd in a && chain, resolving relative targets against the dir the
    # previous cd landed in (not the original cwd) so `cd a && cd b` ends in a/b.
    # a parenthesized group is a subshell — `( cd a )` never changes this shell's
    # cwd, so scrub those before scanning; `cd a; cd b` splits on `;` too.
    scrubbed = re.sub(r"\([^()]*\)", " ", command)
    running = current_cwd
    changed = False
    for part in re.split(r"&&|;", scrubbed):
        part = part.strip()
        if part.startswith("cd "):
            target = part[3:].strip().strip("'\"")
            if target:
                # Docker commands see the host workspace as /workspace. Map
                # that path back before persisting cwd for the next invocation.
                if workspace_root is not None and container_workspace is not None:
                    container_root = str(container_workspace)
                    if target == container_root or target.startswith(container_root + "/"):
                        suffix = target[len(container_root) :].lstrip("/")
                        target = os.path.join(str(workspace_root), suffix)
                new_dir = os.path.normpath(os.path.join(running, os.path.expanduser(target)))
                if os.path.isdir(new_dir):
                    running = new_dir
                    changed = True
    if changed:
        if workspace_root is not None:
            root = os.path.realpath(workspace_root)
            try:
                if os.path.commonpath([root, os.path.realpath(running)]) != root:
                    return
            except ValueError:
                return
        state.cwd = running
