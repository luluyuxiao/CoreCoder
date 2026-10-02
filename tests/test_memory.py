"""Structured memory survives compaction/resume without becoming a fake sandbox."""

import json

import pytest

from corecoder.agent import Agent
from corecoder.demo import ScriptedLLM
from corecoder.llm import ToolCall
from corecoder.memory import MemoryState
from corecoder.permissions import Permission
from corecoder.tools.memory import MemoryUpdateTool
from corecoder.tools.write import WriteFileTool


def test_memory_state_roundtrips_versioned_constraints_and_plan():
    state = MemoryState()
    state.set_goal("Ship the MCP sandbox")
    constraint = state.add_constraint("Do not modify production.yaml")
    step = state.add_plan_step("Add container transport")
    state.update_plan_step(step["id"], status="in_progress")
    state.add_decision("Keep host mode as the compatibility default", reason="no migration break")
    state.record_file("corecoder/mcp.py")
    assert state.revoke_constraint(constraint["id"])

    restored = MemoryState.from_dict(json.loads(json.dumps(state.to_dict())))

    assert restored.to_dict() == state.to_dict()
    assert restored.constraints[0]["status"] == "revoked"
    assert "Ship the MCP sandbox" in restored.render()


def test_structured_memory_is_reinjected_and_survives_context_compression(tmp_path):
    agent = Agent(
        llm=ScriptedLLM([]),
        tools=[],
        workspace=tmp_path,
        max_context_tokens=8_000,
    )
    agent.memory.set_goal("Preserve this goal")
    agent.memory.add_constraint("Never access the production database")
    for index in range(20):
        agent._append_message({
            "role": "user",
            "content": f"old message {index} " + "x" * 1200,
        })

    compressed, _, _ = agent.compress_context("test")

    assert compressed
    system = agent._full_messages()[0]["content"]
    assert "Preserve this goal" in system
    assert "Never access the production database" in system
    assert "do not replace Permission" in system


def test_structured_memory_restores_with_session_snapshot(tmp_path):
    original = Agent(llm=ScriptedLLM([]), tools=[], workspace=tmp_path)
    original.memory.set_goal("Resume me")
    original.memory.add_plan_step("Run tests")
    original.memory.add_decision("Use SQLite")
    snapshot = original.state_snapshot("saved")

    restored = Agent(llm=ScriptedLLM([]), tools=[], workspace=tmp_path)
    restored.restore_state(snapshot)

    assert restored.memory.to_dict() == original.memory.to_dict()
    assert "Resume me" in restored._full_messages()[0]["content"]


def test_child_memory_is_copied_instead_of_shared(tmp_path):
    parent_state = MemoryState()
    parent_state.set_goal("Parent goal")
    constraint = parent_state.add_constraint("Never touch production")
    parent_state.add_decision("Use a worktree")

    child = Agent(
        llm=ScriptedLLM([]),
        tools=[],
        workspace=tmp_path,
        memory_state=parent_state,
    )
    child.memory.revoke_constraint(constraint["id"])
    child.memory.add_plan_step("Child-only step")

    assert parent_state.constraints[0]["status"] == "active"
    assert parent_state.plan == []
    assert "Never touch production" not in child.memory.render()


def test_memory_update_tool_is_bound_to_its_agent_and_cannot_add_constraints(tmp_path):
    tool = MemoryUpdateTool()
    agent = Agent(llm=ScriptedLLM([]), tools=[tool], workspace=tmp_path)

    result = agent._exec_tool(ToolCall(
        id="memory-1",
        name="memory_update",
        arguments={"operation": "add_plan_step", "content": "Inspect the repository"},
    ))

    assert result == "Plan step added: step-1"
    assert agent.memory.plan[0]["content"] == "Inspect the repository"
    operations = tool.parameters["properties"]["operation"]["enum"]
    assert "add_constraint" not in operations
    assert Permission().check("memory_update", {}) is None


def test_successful_write_records_a_session_scoped_modified_file(tmp_path):
    agent = Agent(
        llm=ScriptedLLM([]),
        tools=[WriteFileTool()],
        workspace=tmp_path,
    )

    result = agent._exec_tool(ToolCall(
        id="write-1",
        name="write_file",
        arguments={"file_path": "notes.txt", "content": "hello"},
    ))

    assert result.startswith("Wrote")
    assert agent.memory.files_modified == ["notes.txt"]


def test_structured_memory_budget_rejects_growth_without_partial_update():
    state = MemoryState()
    error = None
    for index in range(100):
        before = state.to_dict()
        try:
            state.add_decision(f"decision {index} " + "x" * 2_000)
        except ValueError as exc:
            error = exc
            assert state.to_dict() == before
            break

    assert error is not None
    assert "budget" in str(error)
    with pytest.raises(ValueError):
        state.add_decision("another oversized decision " + "y" * 2_000)
