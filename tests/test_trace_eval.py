"""Structured tracing and the repeatable eval runner."""

import json
import threading
from pathlib import Path
from unittest import mock

import pytest
from openai import APIConnectionError

from corecoder import Agent
from corecoder.demo import ScriptedLLM
from corecoder.eval import EvalCase, compare_reports, load_eval_cases, run_eval_suite
from corecoder.hooks import Hooks
from corecoder.llm import LLM, LLMResponse, ToolCall
from corecoder.permissions import Permission
from corecoder.sandbox import WorkspacePathPolicy
from corecoder.tools.agent import AgentTool
from corecoder.tools.read import ReadFileTool
from corecoder.tools.write import WriteFileTool
from corecoder.trace import CompositeTrace, JsonlTrace, MemoryTrace, TraceSink


def _request():
    try:
        import httpx
    except ModuleNotFoundError:
        import httpx2 as httpx
    return httpx.Request("POST", "https://example.test/v1/chat/completions")


def test_agent_trace_covers_round_tool_permission_and_result_without_content(tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("hello trace", encoding="utf-8")
    trace = MemoryTrace()
    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall(
                id="c1",
                name="read_file",
                arguments={"file_path": "note.txt"},
            )]),
            LLMResponse(content="done"),
        ]),
        tools=[ReadFileTool(path_policy=WorkspacePathPolicy(tmp_path))],
        permission=Permission(),
        workspace=tmp_path,
        trace=trace,
    )

    assert agent.chat("read it") == "done"
    names = [event["event"] for event in trace.events]
    assert names[0] == "agent.run.started"
    assert names[-1] == "agent.run.completed"
    assert names.count("llm.request.completed") == 2
    assert "tool.requested" in names
    assert "tool.started" in names
    assert "tool.completed" in names

    requested = next(event for event in trace.events if event["event"] == "tool.requested")
    permission = next(
        event for event in trace.events if event["event"] == "tool.permission_decided"
    )
    assert requested["round"] == 1
    assert requested["argument_keys"] == ["file_path"]
    assert "arguments" not in requested
    assert permission["decision"] == "read_only"
    assert permission["allowed"] is True
    assert "user_input" not in trace.events[0]
    assert "result" not in trace.events[-1]
    assert len({event["run_id"] for event in trace.events}) == 1


def test_trace_content_is_explicit_opt_in(tmp_path):
    trace = MemoryTrace(capture_content=True)
    agent = Agent(
        llm=ScriptedLLM([LLMResponse(content="secret answer")]),
        tools=[],
        trace=trace,
        workspace=tmp_path,
    )

    assert agent.chat("secret prompt") == "secret answer"
    assert trace.events[0]["user_input"] == "secret prompt"
    assert trace.events[-1]["result"] == "secret answer"


def test_trace_sink_failure_never_fails_agent(tmp_path, caplog):
    class BrokenTrace(TraceSink):
        capture_content = True

        def emit(self, event: str, **fields):
            raise RuntimeError("collector down")

        def content(self, **fields):
            raise RuntimeError("policy down")

    agent = Agent(
        llm=ScriptedLLM([LLMResponse(content="still works")]),
        tools=[],
        trace=BrokenTrace(),
        workspace=tmp_path,
    )

    assert agent.chat("go") == "still works"
    assert any("trace sink failed" in record.getMessage() for record in caplog.records)
    assert any("trace content policy failed" in record.getMessage() for record in caplog.records)


def test_composite_trace_isolates_broken_destinations():
    class BrokenTrace(TraceSink):
        def emit(self, event: str, **fields):
            raise RuntimeError("broken")

    memory = MemoryTrace()
    CompositeTrace(BrokenTrace(), memory).emit("example", value=1)
    assert memory.events[0]["event"] == "example"


def test_composite_trace_only_sends_content_to_opted_in_sinks():
    metadata = MemoryTrace()
    content = MemoryTrace(capture_content=True)
    trace = CompositeTrace(metadata, content)

    trace.emit("example", result="secret", duration_ms=2)
    assert "result" not in metadata.events[0]
    assert content.events[0]["result"] == "secret"
    assert metadata.events[0]["duration_ms"] == 2


def test_jsonl_trace_writes_valid_records_from_multiple_threads(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text("stale\n", encoding="utf-8")
    trace = JsonlTrace(path, append=False)

    def write(worker):
        for index in range(20):
            trace.emit("test.event", worker=worker, index=index)

    threads = [threading.Thread(target=write, args=(worker,)) for worker in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 80
    assert [record["sequence"] for record in records] == list(range(1, 81))
    assert len({record["session_id"] for record in records}) == 1


def test_hooks_and_permission_decisions_are_traceable(tmp_path):
    trace = MemoryTrace()
    target = tmp_path / "written.txt"
    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall(
                id="c1",
                name="write_file",
                arguments={"file_path": str(target), "content": "ok"},
            )]),
            LLMResponse(content="done"),
        ]),
        tools=[WriteFileTool()],
        hooks=Hooks(pre=[{"matcher": "*", "command": "true"}], post=[]),
        permission=Permission(ask=lambda name, arguments: "once"),
        trace=trace,
    )

    assert agent.chat("write") == "done"
    pre = next(event for event in trace.events if event["event"] == "hook.pre.completed")
    permission = next(
        event for event in trace.events if event["event"] == "tool.permission_decided"
    )
    post = next(event for event in trace.events if event["event"] == "hook.post.completed")
    assert pre["configured"] is True and pre["blocked"] is False
    assert permission["decision"] == "once" and permission["allowed"] is True
    assert post["configured"] is True


def test_context_trace_reports_compression_and_opt_in_before_after(tmp_path):
    trace = MemoryTrace(capture_content=True)
    agent = Agent(
        llm=ScriptedLLM([]),
        tools=[],
        max_context_tokens=1_000,
        trace=trace,
        workspace=tmp_path,
    )
    agent.messages = [{
        "role": "tool",
        "tool_call_id": "c1",
        "content": "line\n" * 1_000,
    }]

    compressed, before, after = agent.compress_context()
    assert compressed and after < before
    event = trace.events[-1]
    assert event["event"] == "context.compressed"
    assert "tool_snip" in event["actions"]
    assert event["reclaimed_tokens"] == before - after
    assert len(event["messages_before"][0]["content"]) > len(
        event["messages_after"][0]["content"]
    )


def test_llm_emits_retry_and_fallback_events(monkeypatch):
    llm = LLM(
        model="gpt-4o",
        fallback_models=["gpt-4o-mini"],
        api_key="test",
    )
    llm.client = mock.Mock()
    empty_stream = iter([])
    llm.client.chat.completions.create.side_effect = [
        APIConnectionError(request=_request()),
        APIConnectionError(request=_request()),
        APIConnectionError(request=_request()),
        empty_stream,
    ]
    monkeypatch.setattr("corecoder.llm.time.sleep", lambda seconds: None)
    events = []

    llm.chat(
        [{"role": "user", "content": "hello"}],
        on_event=lambda name, fields: events.append((name, fields)),
    )

    assert [name for name, _ in events].count("llm.retry") == 2
    fallback = next(fields for name, fields in events if name == "llm.fallback")
    assert fallback["from_model"] == "gpt-4o"
    assert fallback["to_model"] == "gpt-4o-mini"


def test_subagent_trace_preserves_parent_child_relationship(tmp_path):
    trace = MemoryTrace()
    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall(
                id="parent-call",
                name="agent",
                arguments={"task": "inspect"},
            )]),
            LLMResponse(content="child result"),
            LLMResponse(content="parent result"),
        ]),
        tools=[AgentTool()],
        trace=trace,
        workspace=tmp_path,
    )

    assert agent.chat("delegate") == "parent result"
    started = [event for event in trace.events if event["event"] == "agent.run.started"]
    assert len(started) == 2
    child = next(event for event in started if event["agent_id"] != agent.agent_id)
    assert child["parent_agent_id"] == agent.agent_id
    subagent_started = next(
        event for event in trace.events if event["event"] == "subagent.started"
    )
    assert child["subagent_task_id"] == subagent_started["task_id"]
    assert any(event["event"] == "subagent.completed" for event in trace.events)


def test_eval_suite_isolated_repeated_and_metric_driven(tmp_path):
    source = tmp_path / "fixture"
    source.mkdir()
    (source / "note.txt").write_text("expected value", encoding="utf-8")
    case = EvalCase(
        name="read-note",
        prompt="read the note",
        workspace=source,
        expect={
            "response_contains": ["expected value"],
            "files_exist": ["note.txt"],
            "tool_calls_include": ["read_file"],
            "max_tool_calls": 1,
            "max_llm_rounds": 2,
        },
    )

    def factory(case, workspace, trace):
        return Agent(
            llm=ScriptedLLM([
                LLMResponse(tool_calls=[ToolCall(
                    id="read",
                    name="read_file",
                    arguments={"file_path": "note.txt"},
                )]),
                LLMResponse(content="expected value"),
            ]),
            tools=[ReadFileTool(path_policy=WorkspacePathPolicy(workspace))],
            permission=Permission(),
            workspace=workspace,
            trace=trace,
        )

    report = run_eval_suite([case], factory, repetitions=2)
    assert report["summary"]["cases"] == 1
    assert report["summary"]["runs"] == 2
    assert report["summary"]["passed_runs"] == 2
    assert report["summary"]["success_rate"] == 1.0
    assert report["cases"][0]["pass_at_k"] is True
    assert all(result["metrics"]["tool_calls"] == 1 for result in report["results"])
    assert all(result["metrics"]["llm_rounds"] == 2 for result in report["results"])
    assert all(
        result["metrics"]["models_used"] == ["scripted-demo"]
        for result in report["results"]
    )
    assert all(Path(result["workspace"]) != source for result in report["results"])
    assert (source / "note.txt").read_text(encoding="utf-8") == "expected value"


def test_eval_manifest_validation_and_report_comparison(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest = tmp_path / "eval.json"
    manifest.write_text(json.dumps({
        "cases": [{
            "name": "one",
            "workspace": "workspace",
            "prompt": "do one thing",
            "expect": {"response_contains": ["done"]},
        }],
    }), encoding="utf-8")

    cases = load_eval_cases(manifest)
    assert cases[0].workspace == workspace.resolve()
    comparison = compare_reports(
        {"summary": {"success_rate": 0.9, "mean_tool_calls": 8}},
        {"summary": {"success_rate": 0.7, "mean_tool_calls": 10}},
    )
    assert comparison["success_rate_delta"] == pytest.approx(0.2)
    assert comparison["mean_tool_calls_delta"] == -2
    assert comparison["mean_estimated_cost_usd_delta"] is None
    with pytest.raises(TypeError, match="JSON objects"):
        compare_reports({}, [])
