"""MCP stdio servers: handshake, tool registration, calls, dying servers.

The fake server is a stdlib-only Python script run via sys.executable, so
these tests need no shell and run the same on Windows.
"""

import concurrent.futures
import json
import logging
import sys
import urllib.error
from unittest import mock

import pytest

from corecoder import mcp
from corecoder.agent import Agent
from corecoder.capabilities import (
    FILESYSTEM_READ,
    FILESYSTEM_WRITE,
    MCP,
    NETWORK,
    PROCESS,
    CapabilityPolicy,
    CapabilityRule,
)
from corecoder.demo import ScriptedLLM
from corecoder.hooks import Hooks
from corecoder.llm import LLMResponse, ToolCall
from corecoder.mcp import MCPError, load_mcp_tools, refresh_mcp_tools
from corecoder.permissions import Permission
from corecoder.tools.base import ToolEffect

FAKE_SERVER = """
import json, os, sys, time

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8", newline="\\n")

TOOLS = [
    {"name": "echo", "description": "Echo text back",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "crash", "description": "Take the server down mid-call",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "stall", "description": "Answer too slowly",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "fail", "description": "Report a tool-level error",
     "inputSchema": {"type": "object", "properties": {}}},
]

for line in sys.stdin:
    try:
        req = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "id" not in req:
        continue  # notification, nothing to answer
    method, params = req.get("method"), req.get("params") or {}
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "fake", "version": "0.1"}}
    elif method == "tools/list":
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/message", "params": {}}) + "\\n")
        result = {"tools": TOOLS}
    elif method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        if name == "crash":
            os._exit(1)
        if name == "stall":
            time.sleep(30)
        content = [{"type": "text", "text": "echo: " + args["text"] if name == "echo" else "bad input near 42"}]
        result = {"content": content, "isError": name == "fail"}
    else:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req["id"], "error": {"code": -32601, "message": "no such method"}}) + "\\n")
        continue
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result}) + "\\n")
    sys.stdout.flush()
"""


@pytest.fixture
def server_script(tmp_path):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    return script


@pytest.fixture
def mcp_config(tmp_path, server_script):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"fake": {
        "command": sys.executable, "args": [str(server_script)]}}}), encoding="utf-8")
    return cfg


@pytest.fixture(autouse=True)
def _close_clients():
    yield
    for client in mcp._live_clients:
        client.close()
    mcp._live_clients.clear()


def _echo_call(call_id="c1", text="hi"):
    return ToolCall(id=call_id, name="mcp__fake__echo", arguments={"text": text})


def _agent(script, tools, **kwargs):
    return Agent(llm=ScriptedLLM(script), tools=tools, **kwargs)


def test_handshake_registers_each_remote_tool(mcp_config):
    tools = load_mcp_tools(mcp_config)
    assert {t.name for t in tools} == {
        "mcp__fake__echo", "mcp__fake__crash", "mcp__fake__stall", "mcp__fake__fail"}
    echo = next(t for t in tools if t.name == "mcp__fake__echo")
    assert echo.description == "Echo text back"
    assert echo.effect == ToolEffect.UNKNOWN
    assert echo.capabilities == frozenset({
        MCP, PROCESS, NETWORK, FILESYSTEM_READ, FILESYSTEM_WRITE,
    })
    assert echo.schema()["function"]["parameters"]["properties"]["text"] == {"type": "string"}


def test_call_round_trip_returns_text_content(mcp_config):
    agent = _agent(
        [LLMResponse(tool_calls=[_echo_call()]), LLMResponse(content="done")],
        load_mcp_tools(mcp_config),
    )

    assert agent.chat("go") == "done"
    result = agent.messages[2]
    assert result["role"] == "tool" and result["content"] == "echo: hi"


def test_parallel_client_calls_to_one_server_dont_cross_wires(mcp_config):
    echo = next(t for t in load_mcp_tools(mcp_config) if t.name == "mcp__fake__echo")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(echo.execute, text="one")
        two = pool.submit(echo.execute, text="two")

    assert one.result() == "echo: one"
    assert two.result() == "echo: two"


def test_server_crash_mid_call_fails_without_killing_the_loop(mcp_config):
    agent = _agent(
        [LLMResponse(tool_calls=[ToolCall(id="c1", name="mcp__fake__crash", arguments={})]),
         LLMResponse(tool_calls=[_echo_call("c2")]),
         LLMResponse(content="still alive")],
        load_mcp_tools(mcp_config),
    )

    assert agent.chat("go") == "still alive"
    crash_result = agent.messages[2]
    assert "Error executing mcp__fake__crash" in crash_result["content"]
    assert "exited" in crash_result["content"]
    # A later call reconnects and re-handshakes. The failed tool call itself is
    # never replayed because it may already have produced a side effect.
    assert agent.messages[4]["content"] == "echo: hi"


def test_a_slow_server_times_out_the_call(mcp_config):
    stall = next(t for t in load_mcp_tools(mcp_config) if t.name == "mcp__fake__stall")
    stall._client.call_timeout = 0.2
    with pytest.raises(MCPError, match="no answer"):
        stall.execute()


def test_a_tool_level_error_surfaces_its_message(mcp_config):
    fail = next(t for t in load_mcp_tools(mcp_config) if t.name == "mcp__fake__fail")
    with pytest.raises(MCPError, match="bad input near 42"):
        fail.execute()


def test_missing_config_means_no_mcp(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert load_mcp_tools(tmp_path / "nope.json") == []
    assert caplog.records == []


def test_broken_config_is_ignored_with_one_warning(tmp_path, caplog):
    bad = tmp_path / "mcp.json"
    bad.write_text("{not json")
    with caplog.at_level(logging.WARNING):
        assert load_mcp_tools(bad) == []
    assert len(caplog.records) == 1
    assert "ignoring" in caplog.records[0].getMessage()


def test_an_unstartable_server_is_skipped_with_one_warning(tmp_path, caplog):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"ghost": {"command": "not-a-real-binary-xyz"}}}))
    with caplog.at_level(logging.WARNING):
        assert load_mcp_tools(cfg) == []
    assert len(caplog.records) == 1
    assert "ghost" in caplog.records[0].getMessage()


def test_mcp_config_passes_per_server_docker_sandbox(tmp_path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({
        "defaults": {
            "sandbox": {
                "mode": "docker",
                "image": "mcp-base:1",
                "network": "none",
                "workspace": "none",
            },
        },
        "mcpServers": {
            "weather": {
                "command": "weather-server",
                "sandbox": {"network": "bridge"},
            },
        },
    }), encoding="utf-8")
    class StubClient:
        def __init__(self, name, command, args=(), env=None, **kwargs):
            self.name = name
            self.tools = []
            self.kwargs = kwargs

        def close(self):
            pass

    with pytest.MonkeyPatch.context() as monkeypatch:
        created = []

        def factory(*args, **kwargs):
            client = StubClient(*args, **kwargs)
            created.append(client)
            return client

        monkeypatch.setattr(mcp, "MCPClient", factory)
        assert load_mcp_tools(cfg, workspace=tmp_path) == []

    assert created[0].kwargs["sandbox"] == {
        "mode": "docker",
        "image": "mcp-base:1",
        "network": "bridge",
        "workspace": "none",
    }
    assert created[0].kwargs["workspace"] == tmp_path


def test_docker_mcp_capabilities_match_enforced_boundary(tmp_path):
    process = mock.Mock()
    process.stdin = mock.Mock()
    process.stdout = []
    process.poll.return_value = None
    process.wait.return_value = 0
    with (
        mock.patch(
            "corecoder.mcp.DockerCommandExecutor.start_stdio_process",
            return_value=(process, "corecoder-mcp-test"),
        ),
        mock.patch.object(
            mcp.MCPClient,
            "_request",
            side_effect=[{}, {"tools": []}],
        ),
        mock.patch.object(mcp.MCPClient, "_notify"),
        mock.patch("corecoder.mcp.DockerCommandExecutor.remove_container") as remove,
    ):
        client = mcp.MCPClient(
            "weather",
            "weather-server",
            sandbox={
                "mode": "docker",
                "network": "bridge",
                "workspace": "ro",
            },
            workspace=tmp_path,
        )
        assert client.capabilities == frozenset({
            MCP, PROCESS, NETWORK, FILESYSTEM_READ,
        })
        assert client.sandbox_mode == "docker"
        client.close()

    remove.assert_called_once_with("corecoder-mcp-test")


def test_mcp_tools_sit_behind_the_consent_gate():
    # not in READ_ONLY, so with nobody to ask the call is refused, never run
    assert Permission().check("mcp__fake__echo", {}) is not None


def test_hooks_match_mcp_tool_names(mcp_config):
    blocker = f'"{sys.executable}" -c "import sys; sys.stderr.write(\'mcp frozen\'); sys.exit(2)"'
    agent = _agent(
        [LLMResponse(tool_calls=[_echo_call()]), LLMResponse(content="done")],
        load_mcp_tools(mcp_config),
        hooks=Hooks(pre=[{"matcher": "mcp__fake__echo", "command": blocker}], post=[]),
    )

    assert agent.chat("go") == "done"
    assert "mcp frozen" in agent.messages[2]["content"]


class _HTTPResponse:
    def __init__(self, payload=None, *, headers=None):
        self.payload = payload
        self.headers = headers or {"Content-Type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b"" if self.payload is None else json.dumps(self.payload).encode()

    def close(self):
        pass


def test_streamable_http_handshake_session_and_tool_call(tmp_path, monkeypatch):
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        if request.method == "DELETE":
            return _HTTPResponse()
        message = json.loads(request.data)
        method = message["method"]
        if "id" not in message:
            return _HTTPResponse()
        if method == "initialize":
            return _HTTPResponse(
                {"jsonrpc": "2.0", "id": message["id"], "result": {
                    "protocolVersion": "2025-06-18", "capabilities": {}
                }},
                headers={
                    "Content-Type": "application/json",
                    "Mcp-Session-Id": "session-1",
                },
            )
        if method == "tools/list":
            result = {"tools": [{
                "name": "weather",
                "description": "Current weather",
                "inputSchema": {"type": "object", "properties": {
                    "city": {"type": "string"},
                }},
            }]}
        else:
            result = {"content": [{"type": "text", "text": "sunny"}]}
        return _HTTPResponse({
            "jsonrpc": "2.0", "id": message["id"], "result": result,
        })

    monkeypatch.setattr("corecoder.mcp.urllib.request.urlopen", urlopen)
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "remote": {"url": "https://mcp.example.test", "headers": {"X-Test": "yes"}},
    }}))

    tools = load_mcp_tools(cfg)
    assert [tool.name for tool in tools] == ["mcp__remote__weather"]
    assert tools[0].execute(city="Shanghai") == "sunny"
    assert tools[0].capabilities == frozenset({MCP, NETWORK})
    tool_request = requests[-1]
    assert tool_request.get_header("Mcp-session-id") == "session-1"
    assert tool_request.get_header("X-test") == "yes"
    tools[0]._client.close()


def test_mcp_circuit_opens_after_repeated_http_transport_failures(monkeypatch):
    failing = {"enabled": False}

    def urlopen(request, timeout):
        if failing["enabled"]:
            raise urllib.error.URLError("offline")
        message = json.loads(request.data)
        if "id" not in message:
            return _HTTPResponse()
        result = {"tools": []} if message["method"] == "tools/list" else {}
        return _HTTPResponse({
            "jsonrpc": "2.0", "id": message["id"], "result": result,
        })

    monkeypatch.setattr("corecoder.mcp.urllib.request.urlopen", urlopen)
    client = mcp.MCPClient(
        "remote",
        url="https://mcp.example.test",
        circuit_failures=2,
        circuit_cooldown=60,
    )
    failing["enabled"] = True

    with pytest.raises(MCPError, match="offline"):
        client.call_tool("weather", {})
    with pytest.raises(MCPError, match="offline"):
        client.call_tool("weather", {})
    with pytest.raises(MCPError, match="circuit is open"):
        client.call_tool("weather", {})
    client.close()


def test_capability_policy_denies_mcp_before_process_start(tmp_path, monkeypatch, caplog):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "blocked": {"command": "must-not-start"},
    }}))
    policy = CapabilityPolicy(
        default="deny",
        rules=[CapabilityRule("mcp__blocked__*", frozenset({MCP}))],
    )
    factory = mock.Mock()
    monkeypatch.setattr(mcp, "MCPClient", factory)

    with caplog.at_level(logging.WARNING):
        assert load_mcp_tools(cfg, capability_policy=policy) == []

    factory.assert_not_called()
    assert "before startup" in caplog.text


def test_dynamic_refresh_replaces_schema_without_touching_builtin():
    client = mock.Mock()
    client.name = "dynamic"
    client.capabilities = frozenset({MCP, NETWORK})
    old = mcp.MCPTool(client, {
        "name": "old", "inputSchema": {"type": "object", "properties": {}},
    })
    new = mcp.MCPTool(client, {
        "name": "new", "inputSchema": {"type": "object", "properties": {}},
    })
    client.refresh_tools.return_value = [new]
    builtin = mock.Mock(spec=[])  # any non-MCP object is retained by identity

    tools, health = refresh_mcp_tools([builtin, old])

    assert tools == [builtin, new]
    assert health == [{
        "name": "dynamic", "transport": client.transport,
        "status": "healthy", "tool_count": 1,
    }]
