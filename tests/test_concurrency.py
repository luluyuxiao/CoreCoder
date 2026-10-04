"""Effect-aware scheduling: parallel reads, exclusive stateful calls."""

import concurrent.futures
import io
import threading
import time
from typing import ClassVar

from rich.console import Console

from corecoder.agent import Agent
from corecoder.cli import _ToolProgressDisplay
from corecoder.demo import ScriptedLLM
from corecoder.llm import LLMResponse, ToolCall
from corecoder.resources import ResourceClaim, ResourceLockManager
from corecoder.tools import build_tools
from corecoder.tools.base import Tool, ToolEffect


def _call(call_id: str, name: str, label: str) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments={"label": label})


class _BarrierReadTool(Tool):
    name = "barrier_read"
    description = "A test read that must overlap another read."
    effect = ToolEffect.READ
    parameters: ClassVar[dict] = {"type": "object", "properties": {}}

    def __init__(self):
        self.barrier = threading.Barrier(2)

    def execute(self, label: str) -> str:
        self.barrier.wait(timeout=2)
        return f"read {label}"


class _TrackedTool(Tool):
    description = "Track concurrent calls for scheduler tests."
    parameters: ClassVar[dict] = {"type": "object", "properties": {}}

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.events = []
        self.lock = threading.Lock()

    def execute(self, label: str) -> str:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.events.append(f"start:{label}")
        time.sleep(0.02)
        with self.lock:
            self.events.append(f"end:{label}")
            self.active -= 1
        return label


class _TrackedWriteTool(_TrackedTool):
    name = "tracked_write"
    effect = ToolEffect.WRITE


class _UnknownTool(_TrackedTool):
    name = "unknown_effect"


class _ResourceWriteTool(_TrackedTool):
    name = "resource_write"
    effect = ToolEffect.WRITE
    resource_parallel = True

    def resource_claims(self, arguments: dict) -> tuple[ResourceClaim, ...]:
        return (ResourceClaim(f"file:{arguments['label']}", "write"),)


def _run_batch(tools, calls):
    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=calls),
            LLMResponse(content="done"),
        ]),
        tools=tools,
    )
    assert agent.chat("go") == "done"
    return agent


def test_consecutive_read_tools_really_overlap():
    tool = _BarrierReadTool()
    agent = _run_batch([
        tool,
    ], [
        _call("c1", tool.name, "one"),
        _call("c2", tool.name, "two"),
    ])

    results = [m["content"] for m in agent.messages if m.get("role") == "tool"]
    assert results == ["read one", "read two"]


def test_write_tools_are_serial_and_keep_model_order():
    tool = _TrackedWriteTool()
    _run_batch([tool], [
        _call("c1", tool.name, "one"),
        _call("c2", tool.name, "two"),
    ])

    assert tool.max_active == 1
    assert tool.events == ["start:one", "end:one", "start:two", "end:two"]


def test_unknown_effect_fails_closed_to_serial_execution():
    tool = _UnknownTool()
    _run_batch([tool], [
        _call("c1", tool.name, "one"),
        _call("c2", tool.name, "two"),
    ])

    assert tool.effect == ToolEffect.UNKNOWN
    assert tool.max_active == 1


def test_resource_writes_to_different_targets_overlap():
    tool = _ResourceWriteTool()
    _run_batch([tool], [
        _call("c1", tool.name, "one"),
        _call("c2", tool.name, "two"),
    ])

    assert tool.max_active == 2


def test_resource_writes_to_same_target_remain_serial():
    tool = _ResourceWriteTool()
    _run_batch([tool], [
        _call("c1", tool.name, "same"),
        _call("c2", tool.name, "same"),
    ])

    assert tool.max_active == 1


def test_shared_resource_manager_serializes_parent_and_child_agents():
    tool = _ResourceWriteTool()
    locks = ResourceLockManager()
    first = Agent(llm=ScriptedLLM([]), tools=[tool], resource_locks=locks)
    second = Agent(llm=ScriptedLLM([]), tools=[tool], resource_locks=locks)
    call_one = _call("c1", tool.name, "same")
    call_two = _call("c2", tool.name, "same")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(first._exec_tool, call_one)
        two = pool.submit(second._exec_tool, call_two)
        assert one.result() == "same"
        assert two.result() == "same"

    assert tool.max_active == 1


def test_write_call_is_a_barrier_between_read_batches():
    events = []
    lock = threading.Lock()
    first_reads = threading.Barrier(2)

    class ReadTool(Tool):
        name = "ordered_read"
        description = "Read with an observable test order."
        effect = ToolEffect.READ
        parameters: ClassVar[dict] = {"type": "object", "properties": {}}

        def execute(self, label: str) -> str:
            with lock:
                events.append(f"start:{label}")
            if label.startswith("before"):
                first_reads.wait(timeout=2)
                time.sleep(0.02)
            with lock:
                events.append(f"end:{label}")
            return label

    class WriteTool(Tool):
        name = "ordered_write"
        description = "Write with an observable test order."
        effect = ToolEffect.WRITE
        parameters: ClassVar[dict] = {"type": "object", "properties": {}}

        def execute(self, label: str) -> str:
            with lock:
                events.extend([f"start:{label}", f"end:{label}"])
            return label

    _run_batch([ReadTool(), WriteTool()], [
        _call("c1", "ordered_read", "before-one"),
        _call("c2", "ordered_read", "before-two"),
        _call("c3", "ordered_write", "write"),
        _call("c4", "ordered_read", "after"),
    ])

    write_start = events.index("start:write")
    write_end = events.index("end:write")
    assert events.index("end:before-one") < write_start
    assert events.index("end:before-two") < write_start
    assert write_end < events.index("start:after")


def test_builtin_tools_declare_conservative_effects():
    effects = {tool.name: tool.effect for tool in build_tools()}

    assert {name for name, effect in effects.items() if effect in {
        ToolEffect.PURE, ToolEffect.READ,
    }} == {"read_file", "glob", "grep", "now", "agent_status"}
    assert effects["write_file"] == ToolEffect.WRITE
    assert effects["edit_file"] == ToolEffect.WRITE
    assert effects["todo_write"] == ToolEffect.WRITE
    assert effects["memory_update"] == ToolEffect.WRITE
    assert effects["bash"] == ToolEffect.EXTERNAL
    assert effects["fetch_url"] == ToolEffect.EXTERNAL
    assert effects["agent"] == ToolEffect.EXTERNAL
    assert effects["agent_resume"] == ToolEffect.EXTERNAL


def test_progress_events_identify_a_real_parallel_batch():
    tool = _BarrierReadTool()
    events = []
    event_lock = threading.Lock()

    def on_progress(event, payload):
        with event_lock:
            events.append((event, payload))

    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[
                _call("c1", tool.name, "one"),
                _call("c2", tool.name, "two"),
            ]),
            LLMResponse(content="done"),
        ]),
        tools=[tool],
    )

    assert agent.chat("go", on_tool_progress=on_progress) == "done"
    assert events[0][0] == "batch_started"
    assert events[0][1]["parallel"] is True
    assert events[0][1]["total"] == 2
    names = [event for event, _ in events]
    assert names.index("tool_started", 1) < names.index("tool_completed")
    assert names.index("tool_started", names.index("tool_started") + 1) < names.index(
        "tool_completed"
    )
    completions = [payload for event, payload in events if event == "tool_completed"]
    assert {item["outcome"] for item in completions} == {"success"}
    assert events[-1][0] == "batch_completed"


def test_progress_callback_failure_does_not_break_agent():
    tool = _TrackedWriteTool()
    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[_call("c1", tool.name, "one")]),
            LLMResponse(content="done"),
        ]),
        tools=[tool],
    )

    def broken_progress(_event, _payload):
        raise RuntimeError("display failed")

    assert agent.chat("go", on_tool_progress=broken_progress) == "done"


def test_cli_progress_renders_parallel_status_and_completion():
    output = io.StringIO()
    display = _ToolProgressDisplay(Console(
        file=output,
        force_terminal=False,
        color_system=None,
        width=120,
    ))
    display("batch_started", {
        "total": 2,
        "parallel": True,
        "tools": [
            {"tool_call_id": "c1", "tool_name": "read_file", "arguments": {"file_path": "README.md"}},
            {"tool_call_id": "c2", "tool_name": "grep", "arguments": {"pattern": "Agent"}},
        ],
    })
    for call_id, name in (("c1", "read_file"), ("c2", "grep")):
        display("tool_started", {"tool_call_id": call_id, "tool_name": name})
        display("tool_completed", {
            "tool_call_id": call_id,
            "tool_name": name,
            "outcome": "success",
            "duration_ms": 12.5,
        })
    display("batch_completed", {"total": 2, "parallel": True})

    rendered = output.getvalue()
    assert "Parallel tools" in rendered
    assert "read_file" in rendered
    assert "grep" in rendered
    assert "2/2 done" in rendered
