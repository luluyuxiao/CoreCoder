"""Tests for core modules: config, context, session, imports."""

import copy
import re
from pathlib import Path
from typing import ClassVar
from unittest import mock

import pytest
from openai import BadRequestError

from corecoder import (
    ALL_TOOLS,
    LLM,
    Agent,
    BudgetExceededError,
    Config,
    JsonlTrace,
    MemoryState,
    MemoryTrace,
    SQLiteSessionStore,
    TraceSink,
    __version__,
)
from corecoder import ContextOverflowError as PublicContextOverflowError
from corecoder import session as session_module
from corecoder.context import (
    ContextManager,
    ContextOverflowError,
    estimate_request_tokens,
    estimate_tokens,
)
from corecoder.llm import LLMResponse, ToolCall
from corecoder.session import list_sessions, load_session, save_session
from tests.conftest import get_tool


def test_version():
    # regex instead of tomllib: the latter only exists on 3.11+ and CI runs 3.10
    m = re.search(r'(?m)^version = "([^"]+)"', Path("pyproject.toml").read_text())
    assert m is not None
    assert __version__ == m.group(1)


def test_readme_line_counts_are_current():
    # The LoC numbers are the brand of this repo. If the engine or the
    # package grows, the README has to move with it, and this test is the
    # alarm: update the badge and the prose in README.md and README_CN.md.
    root = Path(__file__).resolve().parent.parent
    engine_files = [
        root / "corecoder" / name
        for name in ("agent.py", "llm.py", "context.py", "session.py")
    ]
    engine_files += sorted((root / "corecoder" / "tools").glob("*.py"))
    package_files = sorted((root / "corecoder").rglob("*.py"))

    def net_lines(path: Path) -> int:
        return sum(
            1
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    engine = sum(net_lines(f) for f in engine_files)
    physical = sum(
        len(f.read_text(encoding="utf-8").splitlines()) for f in package_files
    )
    package_net = sum(net_lines(f) for f in package_files)

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert f"engine-{engine}_LoC" in readme
    assert f"{len(package_files)} files" in readme
    assert f"{physical:,} physical lines" in readme
    assert f"{package_net:,} net" in readme


def test_public_api_exports():
    """Users should be able to import key classes from the top-level package."""
    assert Agent is not None
    assert LLM is not None
    assert BudgetExceededError is not None
    assert Config is not None
    assert PublicContextOverflowError is ContextOverflowError
    assert TraceSink is not None
    assert JsonlTrace is not None
    assert MemoryState is not None
    assert MemoryTrace is not None
    assert SQLiteSessionStore is not None
    assert len(ALL_TOOLS) == 12


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("CORECODER_MODEL", "test-model")
    c = Config.from_env()
    assert c.model == "test-model"


def test_config_defaults(monkeypatch):
    # clear relevant env vars without leaking the change into other tests
    # and isolate the default contract from a developer's real local .env
    monkeypatch.setattr("corecoder.config._load_dotenv", lambda: None)
    monkeypatch.delenv("CORECODER_MODEL", raising=False)
    monkeypatch.delenv("CORECODER_MAX_TOKENS", raising=False)
    monkeypatch.delenv("CORECODER_FALLBACK_MODELS", raising=False)
    monkeypatch.delenv("CORECODER_MAX_COST_USD", raising=False)
    monkeypatch.delenv("CORECODER_TRACE", raising=False)
    monkeypatch.delenv("CORECODER_TRACE_CONTENT", raising=False)
    monkeypatch.delenv("CORECODER_STORAGE_PATH", raising=False)
    monkeypatch.delenv("CORECODER_AUTOSAVE", raising=False)

    c = Config.from_env()
    assert c.model == "gpt-5.5"
    assert c.max_tokens == 4096
    assert c.temperature == 0.0
    assert c.fallback_models == []
    assert c.max_cost_usd is None
    assert c.trace_path is None
    assert c.trace_content is False
    assert c.storage_path is None
    assert c.autosave is True


def test_config_reads_trace_settings(monkeypatch, tmp_path):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("CORECODER_TRACE", str(path))
    monkeypatch.setenv("CORECODER_TRACE_CONTENT", "true")

    config = Config.from_env()
    assert config.trace_path == str(path)
    assert config.trace_content is True


def test_config_reads_storage_settings(monkeypatch, tmp_path):
    path = tmp_path / "sessions.db"
    monkeypatch.setenv("CORECODER_STORAGE_PATH", str(path))
    monkeypatch.setenv("CORECODER_AUTOSAVE", "false")

    config = Config.from_env()
    assert config.storage_path == str(path)
    assert config.autosave is False


# --- Context ---

def test_estimate_tokens():
    msgs = [{"role": "user", "content": "hello world"}]
    t = estimate_tokens(msgs)
    assert t > 0
    assert t < 100


def test_estimate_tokens_tiered_by_content():
    from corecoder.context import _approx_tokens

    prose = "The quick brown fox jumps over the lazy dog. " * 10  # 460 prose chars
    cjk = "你好世界，这是一段中文。" * 20  # 260 hanzi-ish chars
    code = 'def f(x):\n    return {"k": [1, 2, 3], "s": "{}"}\n' * 10  # symbol-dense

    # CJK reads near 1.5 chars/token, roughly double the old flat rate
    assert _approx_tokens(cjk) == pytest.approx(len(cjk) / 1.5, rel=0.1)
    # prose sits under the old 3-chars rate, symbol soup sits above it
    assert _approx_tokens(prose) == pytest.approx(len(prose) / 3.4, rel=0.1)
    assert _approx_tokens(code) == pytest.approx(len(code) / 2.8, rel=0.1)
    # and every tier still beats the "half the real size" failure the old
    # estimator had on CJK: a 260-char Chinese chat must not read as 86 tokens
    assert _approx_tokens(cjk) > 150


def test_context_snip():
    ctx = ContextManager(max_tokens=3000)
    msgs = [
        {"role": "tool", "tool_call_id": "t1", "content": "x\n" * 1000},
    ]
    before = estimate_tokens(msgs)
    ctx._snip_tool_outputs(msgs)
    after = estimate_tokens(msgs)
    assert after < before


def test_context_snips_giant_single_line_tool_output():
    msgs = [{"role": "tool", "tool_call_id": "t1", "content": "x" * 10_000}]

    assert ContextManager._snip_tool_outputs(msgs)
    assert len(msgs[0]["content"]) <= 1500
    assert "snipped to save context" in msgs[0]["content"]


def test_request_estimate_includes_tool_schemas():
    messages = [{"role": "user", "content": "hello"}]
    small = estimate_request_tokens(messages, [])
    large = estimate_request_tokens(messages, [{
        "type": "function",
        "function": {
            "name": "large_tool",
            "description": "x" * 4000,
            "parameters": {"type": "object", "properties": {}},
        },
    }])

    assert large > small + 1000


def test_context_protects_fresh_tool_result_from_history_snip():
    ctx = ContextManager(max_tokens=8_000)
    old = "old\n" * 2_000
    fresh = "fresh\n" * 1_400
    msgs = [
        {"role": "tool", "tool_call_id": "old", "content": old},
        {"role": "tool", "tool_call_id": "fresh", "content": fresh},
    ]

    ctx.maybe_compress(msgs, protected_tool_call_ids={"fresh"})

    assert len(msgs[0]["content"]) < len(old)
    assert msgs[1]["content"] == fresh
    assert "tool_snip" in ctx.last_actions


def test_context_budget_fit_is_a_hard_postcondition():
    ctx = ContextManager(max_tokens=2_200)
    fixed_tokens = 400
    output_reserve = 200
    huge_args = '{"content":"' + "a" * 10_000 + '"}'
    msgs = [
        {"role": "user", "content": "write it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "fresh",
                "type": "function",
                "function": {"name": "write_file", "arguments": huge_args},
            }],
        },
        {"role": "tool", "tool_call_id": "fresh", "content": "z" * 10_000},
    ]

    ctx.maybe_compress(
        msgs,
        fixed_tokens=fixed_tokens,
        output_reserve=output_reserve,
        protected_tool_call_ids={"fresh"},
    )

    reserved = (
        fixed_tokens
        + output_reserve
        + ctx.safety_margin_tokens
        + estimate_tokens(msgs)
    )
    assert reserved <= ctx.max_tokens
    assert "budget_fit" in ctx.last_actions


def test_context_rejects_irreducible_fixed_overhead():
    ctx = ContextManager(max_tokens=1_000)

    with pytest.raises(ContextOverflowError, match="system prompt, tool schemas"):
        ctx.maybe_compress([], fixed_tokens=800, output_reserve=200)


def test_agent_sends_fresh_tool_observation_before_snipping(tmp_path):
    from corecoder.tools.base import Tool, ToolEffect

    observation = "important middle content\n" * 240

    class LargeRead(Tool):
        name = "large_read"
        description = "Return one important bounded observation."
        effect = ToolEffect.READ
        parameters: ClassVar[dict] = {
            "type": "object", "properties": {}, "required": [],
        }

        def execute(self):
            return observation

    class RecordingLLM:
        model = "recording"
        total_prompt_tokens = 0
        total_completion_tokens = 0
        extra: ClassVar[dict] = {"max_tokens": 128}

        def __init__(self):
            self.requests = []

        def chat(self, messages, tools=None, on_token=None, on_event=None):
            self.requests.append(copy.deepcopy(messages))
            if len(self.requests) == 1:
                return LLMResponse(tool_calls=[ToolCall(
                    id="fresh", name="large_read", arguments={},
                )])
            return LLMResponse(content="done")

    llm = RecordingLLM()
    agent = Agent(
        llm=llm,
        tools=[LargeRead()],
        max_context_tokens=4_000,
        workspace=tmp_path,
    )

    assert agent.chat("read it") == "done"
    seen = next(
        message["content"]
        for message in llm.requests[1]
        if message.get("role") == "tool"
    )
    assert seen == observation
    assert not agent._unconsumed_tool_call_ids


def test_context_compress():
    ctx = ContextManager(max_tokens=2000)
    msgs = []
    for i in range(20):
        msgs.append({"role": "user", "content": f"msg {i} " + "a" * 200})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "b" * 2000})
    before = estimate_tokens(msgs)
    ctx.maybe_compress(msgs, None)
    after = estimate_tokens(msgs)
    assert after < before
    assert len(msgs) < 40  # should be compressed


def test_safe_split_never_orphans_a_tool_message():
    """The kept tail must not begin with a 'tool' message - it would be severed
    from the assistant tool_calls that produced it, which the API rejects."""
    ctx = ContextManager(max_tokens=1000)
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
        {"role": "tool", "tool_call_id": "c2", "content": "result2"},
    ]
    split = ctx._safe_split(messages, keep_recent=1)
    assert messages[split].get("role") != "tool"


def test_compress_never_leaves_an_orphan_tool_reply():
    """After summarisation every tool reply must still follow its tool_calls."""
    ctx = ContextManager(max_tokens=2000)
    msgs = []
    for i in range(20):
        msgs.append({"role": "user", "content": f"msg {i} " + "a" * 200})
        msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}"}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "b" * 800})
    ctx.maybe_compress(msgs, None)
    for i, m in enumerate(msgs):
        if m.get("role") == "tool":
            prev = msgs[i - 1]
            assert prev.get("role") == "tool" or prev.get("tool_calls"), f"orphan tool at {i}"


# --- Session ---

def test_session_save_load(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    msgs = [{"role": "user", "content": "test message"}]
    save_session(msgs, "test-model", "pytest_test_session")
    loaded = load_session("pytest_test_session")
    assert loaded is not None
    assert loaded[0] == msgs
    assert loaded[1] == "test-model"


def test_session_name_is_sanitized(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    msgs = [{"role": "user", "content": "test message"}]
    sid = save_session(msgs, "test-model", "../Research Notes!")

    assert sid == "Research-Notes"
    assert (tmp_path / "sessions.db").exists()
    assert load_session("../Research Notes!") is not None


def test_session_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    assert load_session("nonexistent_session_id") is None


def test_list_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    sessions = list_sessions()
    assert isinstance(sessions, list)


# --- Cost estimation ---

def test_cost_estimation_known_model():
    from corecoder.llm import LLM
    llm = LLM.__new__(LLM)
    llm.model = "gpt-5.4"
    llm.total_prompt_tokens = 1_000_000
    llm.total_completion_tokens = 500_000
    cost = llm.estimated_cost
    assert cost is not None
    assert cost == 2.5 + 7.5  # $2.5/M in + $15/M out * 0.5M

def test_cost_estimation_unknown_model():
    from corecoder.llm import LLM
    llm = LLM.__new__(LLM)
    llm.model = "some-custom-model"
    llm.total_prompt_tokens = 1000
    llm.total_completion_tokens = 500
    assert llm.estimated_cost is None


# --- Changed files tracking ---

def test_edit_tracks_changed_files(tmp_path):
    from corecoder.tools.edit import _changed_files
    _changed_files.clear()
    edit = get_tool("edit_file")
    path = tmp_path / "sample.py"
    path.write_text("aaa\nbbb\n")
    edit.execute(file_path=str(path), old_string="aaa", new_string="zzz")
    assert any(str(path) in p for p in _changed_files)
    _changed_files.clear()


def test_write_tracks_changed_files(tmp_path):
    from corecoder.tools.edit import _changed_files
    _changed_files.clear()
    write = get_tool("write_file")
    path = tmp_path / "tracked.txt"
    write.execute(file_path=str(path), content="tracked\n")
    assert any(path.name in p for p in _changed_files)
    _changed_files.clear()


# --- Agent tool execution ---

def test_agent_tool_scope_is_per_instance():
    """An Agent restricted to a subset of tools must not resolve tools outside it."""
    only_read = [get_tool("read_file")]
    agent = Agent(llm=LLM.__new__(LLM), tools=only_read)
    assert set(agent._tool_by_name) == {"read_file"}

    class _TC:
        name = "bash"  # a real, registered tool - but not in this agent's set
        id = "x"
        arguments: ClassVar[dict] = {"command": "echo hi"}

    assert "unknown tool 'bash'" in agent._exec_tool(_TC())


def test_exec_tool_distinguishes_bad_args_from_internal_error():
    """A TypeError raised inside a tool must not be reported as bad arguments."""
    from corecoder.tools.base import Tool

    class _Boom(Tool):
        name = "boom"
        description = "raises TypeError internally"
        parameters: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}

        def execute(self):
            raise TypeError("internal explosion")

    agent = Agent(llm=LLM.__new__(LLM), tools=[_Boom()])

    class _BadArgs:
        name, id, arguments = "boom", "1", {"unexpected": 1}

    class _Good:
        name, id, arguments = "boom", "2", {}

    assert "bad arguments" in agent._exec_tool(_BadArgs())
    assert "Error executing boom" in agent._exec_tool(_Good())
    assert "bad arguments" not in agent._exec_tool(_Good())


def test_interrupt_backfills_missing_tool_replies():
    """A half-finished tool round must be repaired so history stays valid."""
    agent = Agent(llm=LLM.__new__(LLM), tools=[])
    agent.messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "tool_call_id": "a", "content": "done"},
    ]

    class _TC:
        def __init__(self, i):
            self.id = i

    agent._answer_pending_tool_calls([_TC("a"), _TC("b")])
    replies = [m for m in agent.messages if m.get("role") == "tool"]
    ids = [m["tool_call_id"] for m in replies]
    assert sorted(ids) == ["a", "b"]
    assert ids.count("a") == 1  # the already-answered call wasn't duplicated


# --- Task list injection ---

def test_todo_list_is_injected_into_system_context():
    """After a todo_write call, the next request must carry the list in the system message."""
    from corecoder.tools.todo import TodoWriteTool
    todo = TodoWriteTool()
    agent = Agent(llm=LLM.__new__(LLM), tools=[todo])

    todo.execute(tasks=[
        {"content": "fix the bug", "status": "in_progress"},
        {"content": "add a test", "status": "pending"},
    ])
    system = agent._full_messages()[0]["content"]
    assert "# Current task list" in system
    assert "1. [in_progress] fix the bug" in system
    assert "2. [pending] add a test" in system


def test_todo_injection_tracks_updates():
    """The injection is rebuilt every round: updates show, an empty list injects nothing."""
    from corecoder.tools.todo import TodoWriteTool
    todo = TodoWriteTool()
    agent = Agent(llm=LLM.__new__(LLM), tools=[todo])

    todo.execute(tasks=[{"content": "only task", "status": "in_progress"}])
    todo.execute(tasks=[{"content": "only task", "status": "done"}])
    system = agent._full_messages()[0]["content"]
    assert "[done] only task" in system
    assert "[in_progress] only task" not in system

    todo.execute(tasks=[])
    assert "# Current task list" not in agent._full_messages()[0]["content"]


def test_agent_without_todo_tool_injects_nothing():
    agent = Agent(llm=LLM.__new__(LLM), tools=[get_tool("read_file")])
    assert "# Current task list" not in agent._full_messages()[0]["content"]


# ---------------------------------------------------------------------------
# LLM.chat() provider-dialect fallback (400 param adaptation)
# ---------------------------------------------------------------------------


class TestLLMParamFallback:
    """Newer OpenAI models reject max_tokens / non-default temperature with a
    400 naming the parameter; chat() adapts that one param and retries."""

    @staticmethod
    def _make_llm():
        llm = LLM(model="gpt-5", api_key="sk-test", max_tokens=1024, temperature=0.7)
        llm.client = mock.Mock()
        return llm

    @staticmethod
    def _bad_request(msg):
        try:
            import httpx
        except ModuleNotFoundError:
            # openai>=3.14 moved its HTTP layer from httpx to httpx2; the
            # SDK's error types take a Response from whichever is installed.
            import httpx2 as httpx

        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        return BadRequestError(msg, response=httpx.Response(400, request=req), body=None)

    @staticmethod
    def _stream():
        from types import SimpleNamespace

        return [
            SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(delta=SimpleNamespace(content="ok", tool_calls=None))],
            )
        ]

    def _chat_with_failures(self, llm, failures):
        create = llm.client.chat.completions.create
        create.side_effect = [*failures, self._stream()]
        result = llm.chat(messages=[{"role": "user", "content": "hi"}])
        assert result.content == "ok"
        return create

    def test_max_tokens_translated_when_rejected(self):
        llm = self._make_llm()
        create = self._chat_with_failures(
            llm,
            [self._bad_request("Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.")],
        )
        retry_kwargs = create.call_args_list[-1][1]
        assert retry_kwargs["max_completion_tokens"] == 1024
        assert "max_tokens" not in retry_kwargs

    def test_temperature_dropped_when_rejected(self):
        llm = self._make_llm()
        create = self._chat_with_failures(
            llm,
            [self._bad_request("Unsupported value: 'temperature' does not support 0.7 with this model.")],
        )
        retry_kwargs = create.call_args_list[-1][1]
        assert "temperature" not in retry_kwargs
        assert retry_kwargs["max_tokens"] == 1024

    def test_stream_options_still_dropped_on_first_400(self):
        llm = self._make_llm()
        create = self._chat_with_failures(llm, [self._bad_request("Bad request")])
        retry_kwargs = create.call_args_list[-1][1]
        assert "stream_options" not in retry_kwargs
        assert retry_kwargs["max_tokens"] == 1024

    def test_unrecognized_400_reraises(self):
        llm = self._make_llm()
        with pytest.raises(BadRequestError):
            self._chat_with_failures(
                llm,
                [self._bad_request("Unsupported parameter: 'response_format' here"),
                 self._bad_request("Unsupported parameter: 'response_format' here")],
            )

    def test_rejected_max_completion_tokens_does_not_loop(self):
        """The max_tokens error message names max_completion_tokens, and the
        substring must not re-trigger translation once translated."""
        llm = self._make_llm()
        create = self._chat_with_failures(
            llm,
            [self._bad_request("Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens'."),
             self._bad_request("Unsupported parameter: 'max_completion_tokens' is unknown here")],
        )
        # second failure surfaces because neither retry looped nor re-adapted
        retry_kwargs = create.call_args_list[-1][1]
        assert "max_completion_tokens" in retry_kwargs
