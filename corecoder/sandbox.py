"""Execution and workspace isolation for tools.

The local executor preserves CoreCoder's original behaviour.  The Docker
executor is the opt-in security boundary: commands run in a short-lived,
unprivileged container with only the selected workspace mounted writable.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol


class SandboxViolation(ValueError):
    """A tool tried to address a path outside the configured workspace."""


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    returncode: int


class CommandExecutor(Protocol):
    """Backend used by BashTool."""

    mode: str

    def run(self, command: str, cwd: str, timeout: int) -> CommandResult: ...


class WorkspacePathPolicy:
    """Resolve tool paths while preventing ``..`` and symlink escapes."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError as e:
            raise SandboxViolation(
                f"path {str(path)!r} is outside workspace {str(self.root)!r}"
            ) from e
        return candidate

    def contains(self, path: str | Path) -> bool:
        try:
            self.resolve(path)
        except SandboxViolation:
            return False
        return True


class LocalCommandExecutor:
    """Host execution, optionally rooted at a specific workspace.

    ``workspace_root`` does not turn local execution into a sandbox: a shell
    command can still address the rest of the host.  It only gives an Agent a
    stable initial cwd, which is important for worktree-backed sub-agents.
    """

    mode = "local"

    def __init__(self, workspace_root: str | Path | None = None):
        self.workspace_root = (
            Path(workspace_root).expanduser().resolve()
            if workspace_root is not None
            else None
        )

    def run(self, command: str, cwd: str, timeout: int) -> CommandResult:
        proc = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=cwd,
        )
        return CommandResult(proc.stdout, proc.stderr, proc.returncode)


class DockerCommandExecutor:
    """Run each command in a locked-down, disposable Docker container.

    The workspace is the only host path exposed to the container.  The root
    filesystem is read-only, capabilities are dropped, privilege escalation is
    disabled, and network access defaults to none.  ``--pull=never`` makes a
    missing image fail closed instead of silently reaching the network.
    """

    mode = "docker"
    container_workspace = PurePosixPath("/workspace")

    def __init__(
        self,
        workspace_root: str | Path,
        image: str = "corecoder-sandbox:latest",
        network: str = "none",
        memory: str = "1g",
        cpus: float = 1.0,
        pids_limit: int = 128,
        docker_binary: str = "docker",
    ):
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        if not self.workspace_root.is_dir():
            raise ValueError(f"sandbox workspace is not a directory: {self.workspace_root}")
        if not image.strip():
            raise ValueError("sandbox image may not be empty")
        if not network.strip():
            raise ValueError("sandbox network may not be empty")
        if cpus <= 0:
            raise ValueError("sandbox cpus must be positive")
        if pids_limit <= 0:
            raise ValueError("sandbox pids limit must be positive")
        self.image = image
        self.network = network
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.docker_binary = docker_binary

    def run(self, command: str, cwd: str, timeout: int) -> CommandResult:
        container_cwd = self._container_path(cwd)
        container_name = f"corecoder-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        argv = [
            self.docker_binary,
            "run",
            "--rm",
            "--name",
            container_name,
            "--init",
            "--pull",
            "never",
            "--network",
            self.network,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            self.memory,
            "--cpus",
            str(self.cpus),
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=128m",
            "--env",
            "HOME=/tmp",
            "--env",
            "TMPDIR=/tmp",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--mount",
            # Bind mounts are read-write by default. Docker's --mount parser
            # rejects a bare `rw` field (unlike the shorter --volume syntax).
            f"type=bind,src={self.workspace_root},dst={self.container_workspace}",
            "--workdir",
            str(container_cwd),
        ]
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            argv += ["--user", f"{os.getuid()}:{os.getgid()}"]
        argv += ["--entrypoint", "/bin/sh", self.image, "-lc", command]

        try:
            proc = subprocess.run(
                argv,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            # Killing the docker CLI does not guarantee that the container dies.
            # Remove the precisely named container before surfacing the timeout.
            try:
                subprocess.run(
                    [self.docker_binary, "rm", "-f", container_name],
                    shell=False,
                    check=False,
                    capture_output=True,
                    timeout=10,
                )
            except OSError:
                # Preserve the original timeout if the cleanup command itself
                # cannot be started (for example, Docker disappeared).
                pass
            raise
        return CommandResult(proc.stdout, proc.stderr, proc.returncode)

    def _container_path(self, cwd: str | Path) -> PurePosixPath:
        host_cwd = Path(cwd).expanduser().resolve()
        try:
            relative = host_cwd.relative_to(self.workspace_root)
        except ValueError as e:
            raise SandboxViolation(
                f"working directory {str(host_cwd)!r} is outside workspace "
                f"{str(self.workspace_root)!r}"
            ) from e
        return self.container_workspace.joinpath(*relative.parts)


def create_command_executor(
    mode: str,
    workspace_root: str | Path,
    *,
    image: str = "corecoder-sandbox:latest",
    network: str = "none",
    memory: str = "1g",
    cpus: float = 1.0,
    pids_limit: int = 128,
) -> CommandExecutor:
    if mode == "local":
        return LocalCommandExecutor(workspace_root=workspace_root)
    if mode == "docker":
        return DockerCommandExecutor(
            workspace_root=workspace_root,
            image=image,
            network=network,
            memory=memory,
            cpus=cpus,
            pids_limit=pids_limit,
        )
    raise ValueError(f"unknown sandbox mode {mode!r}; use 'local' or 'docker'")


def command_executor_for_workspace(
    executor: CommandExecutor, workspace_root: str | Path
) -> CommandExecutor:
    """Clone a built-in executor for an independent workspace.

    Worktree mode must not silently downgrade a Docker-backed parent to local
    execution.  Unknown custom executors therefore fail closed unless they
    expose their own ``for_workspace(path)`` factory.
    """
    root = Path(workspace_root).expanduser().resolve()
    if isinstance(executor, DockerCommandExecutor):
        return DockerCommandExecutor(
            workspace_root=root,
            image=executor.image,
            network=executor.network,
            memory=executor.memory,
            cpus=executor.cpus,
            pids_limit=executor.pids_limit,
            docker_binary=executor.docker_binary,
        )
    if isinstance(executor, LocalCommandExecutor):
        return LocalCommandExecutor(workspace_root=root)
    factory = getattr(executor, "for_workspace", None)
    if callable(factory):
        return factory(root)
    raise ValueError(
        f"command executor {type(executor).__name__} cannot be rebound to a worktree"
    )
