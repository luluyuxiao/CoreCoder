"""Sub-agents with foreground/background and shared/worktree modes.

Every child gets a fresh ``Agent`` and conversation. Foreground children keep
the original blocking behaviour. Background children run in daemon threads and
are observed through ``agent_status``. Git worktree isolation is orthogonal to
that choice, so either run mode can use the current checkout or a new branch.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from ..capabilities import SUBAGENT
from ..checkpoints import CheckpointManager
from ..sandbox import WorkspacePathPolicy, command_executor_for_workspace
from .base import Tool, ToolEffect

WORKTREES_DIR = Path.home() / ".corecoder" / "worktrees"
_CONTROL_TOOLS = frozenset({"agent", "agent_status", "agent_resume"})
_MAX_ACTIVE_BACKGROUND = 4
_MAX_RETAINED_JOBS = 32


@dataclass
class _Worktree:
    path: Path
    branch: str
    repo: Path


@dataclass
class _Job:
    id: str
    task: str
    isolation: str
    status: str = "queued"
    result: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    done: threading.Event = field(default_factory=threading.Event)

    def as_dict(self) -> dict:
        """Return the durable, thread-free representation saved in a session."""
        return {
            "id": self.id,
            "task": self.task,
            "isolation": self.isolation,
            "status": self.status,
            "result": self.result,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> _Job | None:
        try:
            task_id = str(raw["id"])
            task = str(raw["task"])
            isolation = str(raw.get("isolation") or "shared")
            status = str(raw.get("status") or "interrupted")
            if not task_id or not task or isolation not in {"shared", "worktree"}:
                return None
            if status not in {"queued", "running", "completed", "failed", "interrupted"}:
                status = "interrupted"
            job = cls(
                id=task_id,
                task=task,
                isolation=isolation,
                status=status,
                result=str(raw.get("result") or ""),
                created_at=float(raw.get("created_at") or time.time()),
                started_at=(
                    float(raw["started_at"])
                    if raw.get("started_at") is not None else None
                ),
                finished_at=(
                    float(raw["finished_at"])
                    if raw.get("finished_at") is not None else None
                ),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if job.status in {"completed", "failed", "interrupted"}:
            job.done.set()
        return job


class AgentTool(Tool):
    name = "agent"
    effect = ToolEffect.EXTERNAL
    capabilities = frozenset({SUBAGENT})
    description = (
        "Run a sub-agent with its own conversation and a 20-round limit. "
        "run_mode=foreground waits for the result; background returns a task ID "
        "for agent_status. isolation=shared uses the current checkout; worktree "
        "creates an independent Git branch from HEAD. Background workers cannot "
        "open permission prompts, so mutating tools must already be pre-approved. "
        "Job records are saved with the parent session; interrupted work is only "
        "restarted after an explicit agent_resume call."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What the sub-agent should accomplish",
            },
            "run_mode": {
                "type": "string",
                "enum": ["foreground", "background"],
                "description": "Wait for completion or return a background task ID (default foreground)",
            },
            "isolation": {
                "type": "string",
                "enum": ["shared", "worktree"],
                "description": "Use this checkout or a new Git worktree from HEAD (default shared)",
            },
        },
        "required": ["task"],
    }

    def __init__(self):
        # Wired by Agent.__init__ after construction.
        self._parent_agent = None
        self._jobs: dict[str, _Job] = {}
        self._jobs_lock = threading.Lock()

    def execute(
        self,
        task: str,
        run_mode: str = "foreground",
        isolation: str = "shared",
    ) -> str:
        if self._parent_agent is None:
            return "Error: agent tool not initialized (no parent agent)"
        if not isinstance(task, str) or not task.strip():
            return "Error: sub-agent task must be a non-empty string"
        if run_mode not in {"foreground", "background"}:
            return "Error: run_mode must be 'foreground' or 'background'"
        if isolation not in {"shared", "worktree"}:
            return "Error: isolation must be 'shared' or 'worktree'"

        if run_mode == "background":
            return self._start_background(task.strip(), isolation)

        task_id = uuid.uuid4().hex[:12]
        self._trace_parent(
            "subagent.started",
            task_id=task_id,
            run_mode=run_mode,
            isolation=isolation,
            **self._parent_agent._trace_content(task=task.strip()),
        )
        try:
            result = self._run(task.strip(), isolation, task_id, background=False)
        except Exception as e:  # noqa: BLE001
            # A child failure is an observation for the parent, never a crash.
            self._trace_parent(
                "subagent.failed",
                task_id=task_id,
                run_mode=run_mode,
                isolation=isolation,
                error_type=type(e).__name__,
            )
            return f"Sub-agent error: {e}"
        self._trace_parent(
            "subagent.completed",
            task_id=task_id,
            run_mode=run_mode,
            isolation=isolation,
            result_chars=len(result),
        )
        return result

    def status(self, task_id: str = "", wait_seconds: int = 0) -> str:
        """Return one job or a compact listing of all retained jobs."""
        if not isinstance(wait_seconds, int) or isinstance(wait_seconds, bool):
            return "Error: wait_seconds must be an integer from 0 to 60"
        if not 0 <= wait_seconds <= 60:
            return "Error: wait_seconds must be between 0 and 60"

        with self._jobs_lock:
            if not task_id:
                jobs = list(self._jobs.values())
                if not jobs:
                    return "No background sub-agents."
                return "\n".join(
                    f"{job.id}: {job.status} ({job.isolation}) - {job.task[:80]}"
                    for job in jobs
                )
            job = self._jobs.get(task_id)
        if job is None:
            return f"Error: unknown background sub-agent task {task_id!r}"

        if wait_seconds and not job.done.is_set():
            job.done.wait(wait_seconds)

        with self._jobs_lock:
            status = job.status
            result = job.result
            started_at = job.started_at
        if status in {"completed", "failed"}:
            return f"[Background sub-agent {status}: {job.id}]\n{result}"
        if status == "interrupted":
            detail = result or "The process exited before this job reached a terminal state."
            return (
                f"[Background sub-agent interrupted: {job.id}]\n{detail}\n"
                f"Use agent_resume(task_id={job.id!r}) to restart it explicitly."
            )
        elapsed = time.time() - (started_at or job.created_at)
        return f"Background sub-agent {job.id}: {status} ({elapsed:.1f}s elapsed)"

    def _start_background(self, task: str, isolation: str) -> str:
        with self._jobs_lock:
            active = sum(job.status in {"queued", "running"} for job in self._jobs.values())
            if active >= _MAX_ACTIVE_BACKGROUND:
                return (
                    f"Error: {_MAX_ACTIVE_BACKGROUND} background sub-agents are already active; "
                    "wait for one with agent_status before starting another"
                )
            self._prune_jobs_locked()
            task_id = uuid.uuid4().hex[:12]
            job = _Job(id=task_id, task=task, isolation=isolation)
            self._jobs[task_id] = job

        self._persist_parent()

        thread = threading.Thread(
            target=self._background_main,
            args=(job,),
            name=f"corecoder-subagent-{task_id}",
            daemon=True,
        )
        thread.start()
        self._trace_parent(
            "subagent.started",
            task_id=task_id,
            run_mode="background",
            isolation=isolation,
            **self._parent_agent._trace_content(task=task),
        )
        return (
            f"[Sub-agent started in background]\nTask ID: {task_id}\n"
            f"Isolation: {isolation}\nUse agent_status(task_id={task_id!r}) to poll, "
            "or pass wait_seconds to wait up to 60 seconds."
        )

    def _background_main(self, job: _Job):
        with self._jobs_lock:
            job.status = "running"
            job.started_at = time.time()
            job.finished_at = None
            job.result = ""
            job.done.clear()
        self._persist_parent()
        try:
            result = self._run(job.task, job.isolation, job.id, background=True)
            status = "completed"
        except Exception as e:  # noqa: BLE001
            result = f"Sub-agent error: {e}"
            status = "failed"
        with self._jobs_lock:
            job.result = result
            job.status = status
            job.finished_at = time.time()
            job.done.set()
        self._persist_parent()
        self._trace_parent(
            f"subagent.{status}",
            task_id=job.id,
            run_mode="background",
            isolation=job.isolation,
            result_chars=len(result),
        )

    def _run(
        self,
        task: str,
        isolation: str,
        task_id: str,
        *,
        background: bool,
    ) -> str:
        # Local import avoids the Agent -> AgentTool -> Agent cycle.
        from ..agent import Agent
        from ..memory import MemoryState

        parent = self._parent_agent
        worktree = _create_worktree(parent.workspace, task_id) if isolation == "worktree" else None
        try:
            workspace = worktree.path if worktree else parent.workspace
            tools = _subagent_tools(parent.tools, workspace, worktree is not None)
            permission = parent.permission
            if background and permission is not None:
                factory = getattr(permission, "for_background", None)
                if not callable(factory):
                    raise RuntimeError(
                        "background mode requires CoreCoder Permission or no permission layer; "
                        "a custom interactive permission callback cannot safely run in a worker"
                    )
                permission = factory()

            child_memory = MemoryState.from_dict({
                "goal": task,
                "constraints": parent.memory.constraints,
                "decisions": parent.memory.decisions,
            })
            sub = Agent(
                llm=parent.llm,
                tools=tools,
                max_context_tokens=parent.context.max_tokens,
                max_rounds=20,
                permission=permission,
                hooks=parent.hooks,
                workspace=workspace,
                llm_lock=parent._llm_lock,
                trace=parent.trace,
                parent_agent_id=parent.agent_id,
                subagent_task_id=task_id,
                capability_policy=parent.capability_policy,
                memory_state=child_memory,
                resource_locks=parent._resource_locks,
            )
            result = sub.chat(task)
        except Exception as e:
            if worktree:
                raise RuntimeError(
                    f"{e}\nWorktree retained at {worktree.path} on branch {worktree.branch}"
                ) from e
            raise
        if len(result) > 5000:
            result = result[:4500] + "\n... (sub-agent output truncated)"

        lines = ["[Sub-agent completed]"]
        if worktree:
            lines.extend([
                f"Worktree: {worktree.path}",
                f"Branch: {worktree.branch}",
                "Base: parent HEAD (uncommitted parent changes were not copied)",
                "Worktree status:",
                _worktree_status(worktree),
            ])
        lines.extend(["Result:", result])
        return "\n".join(lines)

    def _prune_jobs_locked(self):
        if len(self._jobs) < _MAX_RETAINED_JOBS:
            return
        for task_id in list(self._jobs):
            if self._jobs[task_id].status in {"completed", "failed", "interrupted"}:
                del self._jobs[task_id]
                if len(self._jobs) < _MAX_RETAINED_JOBS:
                    return

    def _trace_parent(self, event: str, **fields):
        if self._parent_agent is not None:
            self._parent_agent._trace(event, **fields)

    def _persist_parent(self):
        if (
            self._parent_agent is not None
            and self._parent_agent.provider_valid_state()
        ):
            self._parent_agent.persist_state("running")

    def snapshot_jobs(self) -> list[dict]:
        """Return a stable copy suitable for ``Agent.state_snapshot``."""
        with self._jobs_lock:
            return [job.as_dict() for job in self._jobs.values()]

    def restore_jobs(self, records: list[dict]):
        """Restore retained jobs without silently repeating their side effects.

        A process cannot reattach Python threads after restart. Saved queued or
        running jobs therefore become ``interrupted`` and require the explicit,
        permission-gated ``agent_resume`` tool before they are run again.
        """
        restored: dict[str, _Job] = {}
        for raw in records[-_MAX_RETAINED_JOBS:]:
            if not isinstance(raw, dict):
                continue
            job = _Job.from_dict(raw)
            if job is None:
                continue
            if job.status in {"queued", "running"}:
                previous = job.status
                job.status = "interrupted"
                job.result = (
                    f"Restored from a session while previously {previous}; "
                    "not automatically replayed to avoid duplicate side effects."
                )
                job.finished_at = time.time()
                job.done.set()
            restored[job.id] = job
        with self._jobs_lock:
            self._jobs = restored

    def resume(self, task_id: str) -> str:
        """Explicitly restart one interrupted/failed background task."""
        with self._jobs_lock:
            job = self._jobs.get(task_id)
            if job is None:
                return f"Error: unknown background sub-agent task {task_id!r}"
            if job.status not in {"interrupted", "failed"}:
                return (
                    f"Error: background sub-agent {task_id!r} is {job.status}; "
                    "only interrupted or failed jobs can be resumed"
                )
            active = sum(
                item.status in {"queued", "running"} for item in self._jobs.values()
            )
            if active >= _MAX_ACTIVE_BACKGROUND:
                return f"Error: {_MAX_ACTIVE_BACKGROUND} background sub-agents are already active"
            job.status = "queued"
            job.result = ""
            job.started_at = None
            job.finished_at = None
            job.done.clear()
        self._persist_parent()
        thread = threading.Thread(
            target=self._background_main,
            args=(job,),
            name=f"corecoder-subagent-{task_id}",
            daemon=True,
        )
        thread.start()
        self._trace_parent(
            "subagent.resumed",
            task_id=task_id,
            run_mode="background",
            isolation=job.isolation,
        )
        return (
            f"[Sub-agent resumed in background]\nTask ID: {task_id}\n"
            f"Isolation: {job.isolation}\nUse agent_status to poll."
        )


class AgentStatusTool(Tool):
    """Read the lifecycle/result of in-process background sub-agents."""

    name = "agent_status"
    effect = ToolEffect.READ
    capabilities = frozenset()
    description = (
        "List background sub-agents or read one result. Supply wait_seconds to "
        "wait briefly. Job records survive Session Resume; unfinished jobs are "
        "marked interrupted rather than silently replayed."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Background task ID; omit to list all jobs",
            },
            "wait_seconds": {
                "type": "integer",
                "minimum": 0,
                "maximum": 60,
                "description": "Wait this many seconds if the task is still running",
            },
        },
        "required": [],
    }

    def __init__(self):
        self._parent_agent = None

    def execute(self, task_id: str = "", wait_seconds: int = 0) -> str:
        if self._parent_agent is None:
            return "Error: agent_status tool not initialized (no parent agent)"
        manager = next(
            (tool for tool in self._parent_agent.tools if isinstance(tool, AgentTool)),
            None,
        )
        if manager is None:
            return "Error: no agent tool is registered on this Agent"
        return manager.status(task_id, wait_seconds)


class AgentResumeTool(Tool):
    """Permission-gated explicit restart for a persisted interrupted job."""

    name = "agent_resume"
    effect = ToolEffect.EXTERNAL
    capabilities = frozenset({SUBAGENT})
    description = (
        "Restart an interrupted or failed persisted background sub-agent. "
        "This is explicit because automatically replaying a task could repeat "
        "file writes, commands, or external side effects."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Persisted background task ID to restart",
            },
        },
        "required": ["task_id"],
    }

    def __init__(self):
        self._parent_agent = None

    def execute(self, task_id: str) -> str:
        if self._parent_agent is None:
            return "Error: agent_resume tool not initialized (no parent agent)"
        manager = next(
            (tool for tool in self._parent_agent.tools if isinstance(tool, AgentTool)),
            None,
        )
        if manager is None:
            return "Error: no agent tool is registered on this Agent"
        return manager.resume(task_id)


def _create_worktree(workspace: Path, task_id: str) -> _Worktree:
    """Create or reopen a retained worktree for a durable task ID."""
    repo_result = _git(workspace, "rev-parse", "--show-toplevel")
    if repo_result.returncode != 0:
        raise RuntimeError("worktree isolation requires the current workspace to be a Git repository")
    repo = Path(repo_result.stdout.strip()).resolve()
    digest = hashlib.sha256(os.fspath(repo).encode()).hexdigest()[:8]
    path = WORKTREES_DIR / f"{repo.name}-{digest}" / task_id
    branch = f"corecoder/subagent-{task_id}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        existing = _git(path, "rev-parse", "--show-toplevel")
        if existing.returncode == 0 and Path(existing.stdout.strip()).resolve() == path.resolve():
            return _Worktree(path=path.resolve(), branch=branch, repo=repo)
        raise RuntimeError(f"retained worktree path is occupied: {path}")
    branch_exists = _git(repo, "show-ref", "--verify", f"refs/heads/{branch}")
    args = (
        ("worktree", "add", os.fspath(path), branch)
        if branch_exists.returncode == 0
        else ("worktree", "add", "-b", branch, os.fspath(path), "HEAD")
    )
    added = _git(repo, *args)
    if added.returncode != 0:
        detail = (added.stderr or added.stdout).strip()
        raise RuntimeError(f"could not create Git worktree: {detail}")
    return _Worktree(path=path.resolve(), branch=branch, repo=repo)


def _worktree_status(worktree: _Worktree) -> str:
    status = _git(worktree.path, "status", "--short")
    if status.returncode != 0:
        return f"(status unavailable: {(status.stderr or status.stdout).strip()})"
    text = status.stdout.strip()
    if len(text) > 3000:
        text = text[:2800] + "\n... (worktree status truncated)"
    return text or "(clean)"


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"could not run git {' '.join(args)}: {e}") from e


def _subagent_tools(tools: list[Tool], workspace: Path, isolated: bool) -> list[Tool]:
    """Give a child fresh built-ins, while retaining custom/MCP capabilities."""
    from .bash import BashTool
    from .edit import EditFileTool
    from .fetch import FetchUrlTool
    from .glob_tool import GlobTool
    from .grep import GrepTool
    from .memory import MemoryUpdateTool
    from .now import NowTool
    from .read import ReadFileTool
    from .todo import TodoWriteTool
    from .write import WriteFileTool

    cloned: list[Tool] = []
    checkpoints = CheckpointManager()
    for tool in tools:
        if tool.name in _CONTROL_TOOLS:
            continue  # no recursive descendants and no orphan status tool
        if isinstance(tool, BashTool):
            try:
                executor = command_executor_for_workspace(tool.executor, workspace)
            except ValueError:
                if isolated:
                    raise
                executor = tool.executor
            cloned.append(BashTool(executor=executor))
        elif isinstance(tool, ReadFileTool):
            cloned.append(ReadFileTool(_path_policy(tool, workspace, isolated)))
        elif isinstance(tool, WriteFileTool):
            cloned.append(WriteFileTool(
                _path_policy(tool, workspace, isolated),
                checkpoint_manager=checkpoints,
            ))
        elif isinstance(tool, EditFileTool):
            cloned.append(EditFileTool(
                _path_policy(tool, workspace, isolated),
                checkpoint_manager=checkpoints,
            ))
        elif isinstance(tool, GlobTool):
            cloned.append(GlobTool(_path_policy(tool, workspace, isolated)))
        elif isinstance(tool, GrepTool):
            cloned.append(GrepTool(_path_policy(tool, workspace, isolated)))
        elif isinstance(tool, TodoWriteTool):
            cloned.append(TodoWriteTool())
        elif isinstance(tool, MemoryUpdateTool):
            cloned.append(MemoryUpdateTool())
        elif isinstance(tool, FetchUrlTool):
            cloned.append(FetchUrlTool())
        elif isinstance(tool, NowTool):
            cloned.append(NowTool())
        else:
            # MCP/custom tools have no general cloning contract. Sharing keeps
            # their capability available; UNKNOWN/EXTERNAL effects still keep
            # them exclusive inside each Agent's scheduler.
            cloned.append(tool)
    return cloned


def _path_policy(tool: Tool, workspace: Path, isolated: bool):
    if isolated:
        return WorkspacePathPolicy(workspace)
    policy = getattr(tool, "path_policy", None)
    return WorkspacePathPolicy(policy.root) if policy is not None else None
