"""Foreground/background and shared/worktree sub-agent modes."""

import re
import subprocess
from pathlib import Path

from corecoder.agent import Agent
from corecoder.demo import ScriptedLLM
from corecoder.llm import LLMResponse, ToolCall
from corecoder.permissions import Permission
from corecoder.tools.agent import AgentResumeTool, AgentStatusTool, AgentTool
from corecoder.tools.write import WriteFileTool


def _task_id(started: str) -> str:
    match = re.search(r"Task ID: ([0-9a-f]+)", started)
    assert match is not None
    return match.group(1)


def test_background_subagent_returns_id_and_status_result():
    agent_tool = AgentTool()
    status_tool = AgentStatusTool()
    Agent(
        llm=ScriptedLLM([LLMResponse(content="background answer")]),
        tools=[agent_tool, status_tool],
    )

    started = agent_tool.execute("research it", run_mode="background")
    task_id = _task_id(started)
    result = status_tool.execute(task_id, wait_seconds=2)

    assert f"completed: {task_id}" in result
    assert "background answer" in result
    assert task_id in status_tool.execute()


def test_background_subagent_never_opens_interactive_permission_prompt(tmp_path):
    target = tmp_path / "background.txt"
    asked = []
    agent_tool = AgentTool()
    status_tool = AgentStatusTool()
    Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall(
                id="c1",
                name="write_file",
                arguments={"file_path": str(target), "content": "unsafe\n"},
            )]),
            LLMResponse(content="permission was refused"),
        ]),
        tools=[agent_tool, status_tool, WriteFileTool()],
        permission=Permission(ask=lambda name, args: asked.append(name) or "once"),
    )

    task_id = _task_id(agent_tool.execute("write", run_mode="background"))
    result = status_tool.execute(task_id, wait_seconds=2)

    assert "completed" in result
    assert not target.exists()
    assert asked == []


def test_background_subagent_uses_existing_always_allow_decision(tmp_path):
    target = tmp_path / "background.txt"
    permission = Permission(ask=lambda name, args: "always")
    assert permission.check("write_file", {}) is None
    agent_tool = AgentTool()
    status_tool = AgentStatusTool()
    Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall(
                id="c1",
                name="write_file",
                arguments={"file_path": str(target), "content": "ok\n"},
            )]),
            LLMResponse(content="written"),
        ]),
        tools=[agent_tool, status_tool, WriteFileTool()],
        permission=permission,
    )

    task_id = _task_id(agent_tool.execute("write", run_mode="background"))
    status_tool.execute(task_id, wait_seconds=2)

    assert target.read_text() == "ok\n"


def test_worktree_subagent_writes_only_to_independent_checkout(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "base.txt"], cwd=repo, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=CoreCoder Test",
            "-c", "user.email=corecoder@example.test",
            "commit", "-qm", "base",
        ],
        cwd=repo,
        check=True,
    )
    from corecoder.tools import agent as agent_module
    monkeypatch.setattr(agent_module, "WORKTREES_DIR", tmp_path / "worktrees")

    agent_tool = AgentTool()
    Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall(
                id="c1",
                name="write_file",
                arguments={"file_path": "child.txt", "content": "child\n"},
            )]),
            LLMResponse(content="done"),
        ]),
        tools=[agent_tool, WriteFileTool()],
        permission=Permission(allow_all=True),
        workspace=repo,
    )

    result = agent_tool.execute("make child.txt", isolation="worktree")
    worktree_line = next(line for line in result.splitlines() if line.startswith("Worktree: "))
    worktree = Path(worktree_line.removeprefix("Worktree: "))

    assert not (repo / "child.txt").exists()
    assert (worktree / "child.txt").read_text() == "child\n"
    assert "Branch: corecoder/subagent-" in result
    assert "?? child.txt" in result
    assert "uncommitted parent changes were not copied" in result


def test_worktree_mode_requires_git_repository(tmp_path):
    agent_tool = AgentTool()
    Agent(
        llm=ScriptedLLM([]),
        tools=[agent_tool],
        workspace=tmp_path,
    )

    result = agent_tool.execute("inspect", isolation="worktree")

    assert "Sub-agent error" in result
    assert "requires the current workspace to be a Git repository" in result


def test_durable_task_id_reopens_its_retained_worktree(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "base.txt"], cwd=repo, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=CoreCoder Test",
            "-c", "user.email=corecoder@example.test",
            "commit", "-qm", "base",
        ],
        cwd=repo,
        check=True,
    )
    from corecoder.tools import agent as agent_module
    monkeypatch.setattr(agent_module, "WORKTREES_DIR", tmp_path / "worktrees")

    first = agent_module._create_worktree(repo, "durable-task")
    (first.path / "progress.txt").write_text("keep me\n", encoding="utf-8")
    reopened = agent_module._create_worktree(repo, "durable-task")

    assert reopened.path == first.path
    assert reopened.branch == first.branch
    assert (reopened.path / "progress.txt").read_text() == "keep me\n"


def test_subagent_toolset_omits_both_control_tools():
    agent_tool = AgentTool()
    status_tool = AgentStatusTool()
    parent = Agent(
        llm=ScriptedLLM([]),
        tools=[agent_tool, status_tool, WriteFileTool()],
    )
    from corecoder.tools.agent import _subagent_tools

    names = {tool.name for tool in _subagent_tools(parent.tools, parent.workspace, False)}

    assert names == {"write_file"}


def test_background_jobs_are_serialized_and_unfinished_jobs_restore_interrupted():
    manager = AgentTool()
    manager.restore_jobs([{
        "id": "saved-task",
        "task": "write a report",
        "isolation": "shared",
        "status": "running",
        "created_at": 10,
        "started_at": 11,
    }])

    status = manager.status("saved-task")
    snapshot = manager.snapshot_jobs()

    assert "interrupted" in status
    assert "not automatically replayed" in status
    assert snapshot[0]["status"] == "interrupted"
    assert snapshot[0]["task"] == "write a report"


def test_agent_resume_explicitly_restarts_a_persisted_job():
    manager = AgentTool()
    status_tool = AgentStatusTool()
    resume_tool = AgentResumeTool()
    Agent(
        llm=ScriptedLLM([LLMResponse(content="resumed answer")]),
        tools=[manager, status_tool, resume_tool],
    )
    manager.restore_jobs([{
        "id": "saved-task",
        "task": "continue research",
        "isolation": "shared",
        "status": "running",
    }])

    resumed = resume_tool.execute("saved-task")
    result = status_tool.execute("saved-task", wait_seconds=2)

    assert "resumed in background" in resumed
    assert "completed: saved-task" in result
    assert "resumed answer" in result


def test_background_transitions_are_in_parent_session_snapshots():
    snapshots = []
    manager = AgentTool()
    status_tool = AgentStatusTool()
    Agent(
        llm=ScriptedLLM([LLMResponse(content="durable result")]),
        tools=[manager, status_tool],
        state_callback=snapshots.append,
    )

    task_id = _task_id(manager.execute("persist me", run_mode="background"))
    assert "completed" in status_tool.execute(task_id, wait_seconds=2)

    saved_jobs = [
        job
        for snapshot in snapshots
        for job in snapshot["metadata"].get("background_jobs", [])
        if job["id"] == task_id
    ]
    assert saved_jobs
    assert saved_jobs[-1]["status"] == "completed"
    assert "durable result" in saved_jobs[-1]["result"]


def test_background_transition_never_persists_half_a_parent_tool_batch():
    snapshots = []
    manager = AgentTool()
    status_tool = AgentStatusTool()
    parent = Agent(
        llm=ScriptedLLM([LLMResponse(content="child done")]),
        tools=[manager, status_tool],
        state_callback=snapshots.append,
    )
    parent._append_message({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "parent-agent-call",
            "type": "function",
            "function": {"name": "agent", "arguments": "{}"},
        }],
    })

    task_id = _task_id(manager.execute("finish", run_mode="background"))
    assert "completed" in status_tool.execute(task_id, wait_seconds=2)
    assert snapshots == []

    parent._append_tool_result("parent-agent-call", "background task started")
    assert parent.persist_state("running") is True
    assert snapshots[-1]["metadata"]["background_jobs"][0]["status"] == "completed"
    assert parent.provider_valid_state() is True
