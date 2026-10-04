"""MCP client, distilled to the slice of the protocol an agent uses.

Servers are configured in ~/.corecoder/mcp.json:

    {"mcpServers": {"fs": {"command": "npx", "args": ["-y", "some-fs-server", "/tmp"]}}}

Each server may opt into a persistent hardened Docker stdio container:

    {"sandbox": {"mode": "docker", "network": "none", "workspace": "ro"}}

Servers may use newline-delimited JSON-RPC over stdio or Streamable HTTP. At
startup CoreCoder performs capability admission before launching a process or
opening a remote session, then handshakes (``initialize``), pulls ``tools/list``
and registers every remote tool as ``mcp__<server>__<tool>``. Transport health,
reconnection, a circuit breaker and explicit tool-list refresh are kept here,
outside the Agent loop. A failed ``tools/call`` is never automatically replayed
because the remote side effect may already have happened.
"""

import atexit
import json
import logging
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import ClassVar

from . import __version__
from .capabilities import (
    FILESYSTEM_READ,
    FILESYSTEM_WRITE,
    MCP,
    NETWORK,
    PROCESS,
    CapabilityPolicy,
)
from .resources import ResourceClaim
from .sandbox import DockerCommandExecutor
from .tools.base import Tool, ToolEffect

log = logging.getLogger(__name__)

CONFIG_FILE = Path.home() / ".corecoder" / "mcp.json"
PROTOCOL_VERSION = "2025-06-18"
INIT_TIMEOUT = 15  # seconds for initialize + tools/list at startup
CALL_TIMEOUT = 60  # seconds for one tools/call
CIRCUIT_FAILURES = 3
CIRCUIT_COOLDOWN = 30.0


class MCPError(RuntimeError):
    """Transport or protocol failure talking to one server."""


class MCPClient:
    """One stdio or Streamable HTTP server with bounded recovery."""

    def __init__(
        self,
        name: str,
        command: str | None = None,
        args: list = (),
        env: dict | None = None,
        *,
        url: str | None = None,
        headers: dict | None = None,
        sandbox: dict | str | None = None,
        workspace: str | Path | None = None,
        reconnect: bool = True,
        circuit_failures: int = CIRCUIT_FAILURES,
        circuit_cooldown: float = CIRCUIT_COOLDOWN,
    ):
        if bool(command) == bool(url):
            raise ValueError("MCP server needs exactly one of 'command' or 'url'")
        if circuit_failures <= 0 or circuit_cooldown < 0:
            raise ValueError("MCP circuit settings must be positive")
        self.name = name
        self.call_timeout = CALL_TIMEOUT
        self.transport = "http" if url else "stdio"
        self._command = command
        self._args = _string_args(args)
        self._env = _string_env(env)
        self._url = str(url) if url else None
        self._headers = _string_headers(headers)
        self._session_id: str | None = None
        self._workspace = Path(workspace or Path.cwd()).expanduser().resolve()
        self._sandbox_spec = _normalize_sandbox(sandbox)
        if self.transport == "http" and self._sandbox_spec["mode"] != "host":
            raise ValueError("HTTP MCP servers cannot use the local process sandbox")
        self.reconnect_enabled = bool(reconnect)
        self.circuit_failures = int(circuit_failures)
        self.circuit_cooldown = float(circuit_cooldown)
        self._consecutive_failures = 0
        self._circuit_opened_at: float | None = None
        self._reconnect_lock = threading.Lock()
        self._recovering = False
        self._closed = False
        self._docker_executor: DockerCommandExecutor | None = None
        self._container_name: str | None = None
        self.sandbox_mode = self._sandbox_spec["mode"] if self.transport == "stdio" else "remote"
        self.workspace_access = self._sandbox_spec.get("workspace", "host")
        self.capabilities = _server_capabilities(
            transport=self.transport,
            sandbox=self._sandbox_spec,
        )
        self._next_id = 0
        self._dead: MCPError | None = None
        self._responses: dict[int, dict] = {}
        self._cond = threading.Condition()
        self._write_lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        try:
            self._start_transport()
            self._initialize()
        except BaseException:
            self.close()
            raise

    def _start_transport(self):
        if self.transport == "http":
            self._dead = None
            return
        self._closed = False
        if self.sandbox_mode == "docker":
            self._docker_executor = DockerCommandExecutor(
                self._workspace,
                image=self._sandbox_spec.get("image", "corecoder-sandbox:latest"),
                network=self._sandbox_spec.get("network", "none"),
                memory=self._sandbox_spec.get("memory", "512m"),
                cpus=float(self._sandbox_spec.get("cpus", 0.5)),
                pids_limit=int(self._sandbox_spec.get("pids", 64)),
                docker_binary=self._sandbox_spec.get("docker_binary", "docker"),
            )
            self._proc, self._container_name = self._docker_executor.start_stdio_process(
                self._command,
                self._args,
                env=self._env,
                workspace_access=self.workspace_access,
            )
        else:
            self.workspace_access = "host"
            self._proc = subprocess.Popen(
                [self._command, *self._args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                env={**os.environ, **self._env},
                encoding="utf-8",
                errors="replace",
                bufsize=1,  # line buffered: the transport is newline-delimited JSON
            )
        self._dead = None
        self._responses.clear()
        proc = self._proc
        threading.Thread(target=self._read_loop, args=(proc,), daemon=True).start()

    def _initialize(self):
        self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "corecoder", "version": __version__},
        }, INIT_TIMEOUT, allow_reconnect=False)
        self._notify("notifications/initialized")
        self.refresh_tools(allow_reconnect=False)

    def refresh_tools(self, *, allow_reconnect: bool = True) -> list["MCPTool"]:
        """Refresh a server's dynamic registry and return fresh wrappers."""
        listed = self._request(
            "tools/list", {}, INIT_TIMEOUT, allow_reconnect=allow_reconnect
        )
        specs = listed.get("tools", [])
        if not isinstance(specs, list):
            raise MCPError(f"MCP server {self.name!r} returned invalid tools/list")
        self.tools = [MCPTool(self, spec) for spec in specs if isinstance(spec, dict)]
        return list(self.tools)

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Run one remote tool without replaying an ambiguous failed call."""
        result = self._request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
            self.call_timeout,
            allow_reconnect=True,
        )
        text = "\n".join(
            part.get("text", "")
            for part in result.get("content", [])
            if part.get("type") == "text"
        )
        if result.get("isError"):
            raise MCPError(text or f"{tool_name} reported an error")
        return text or json.dumps(result)  # non-text content: hand the model the raw result

    def health(self) -> dict:
        """Return transport/circuit state; ping when the circuit permits it."""
        try:
            self._request("ping", {}, min(self.call_timeout, 5), allow_reconnect=True)
        except MCPError as e:
            return {
                "name": self.name,
                "transport": self.transport,
                "status": "unhealthy",
                "error": str(e),
                "failures": self._consecutive_failures,
            }
        return {
            "name": self.name,
            "transport": self.transport,
            "status": "healthy",
            "failures": self._consecutive_failures,
        }

    def close(self):
        """Shut the server down. Safe to call twice."""
        self._closed = True
        if self.transport == "http":
            self._close_http_session()
            return
        if self._proc is None or self._proc.poll() is not None:
            self._remove_container()
            return
        try:
            self._proc.stdin.close()
        except (OSError, ValueError):
            pass
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        finally:
            self._remove_container()

    def _remove_container(self):
        if self._docker_executor is not None and self._container_name is not None:
            self._docker_executor.remove_container(self._container_name)
            self._container_name = None

    def _notify(self, method: str):
        try:
            message = {"jsonrpc": "2.0", "method": method, "params": {}}
            if self.transport == "http":
                self._http_exchange(message, INIT_TIMEOUT, expect_response=False)
            else:
                self._write(message)
        except MCPError:
            pass  # a notification has no reply to lose; the next request meets the dead server

    def _request(
        self,
        method: str,
        params: dict,
        timeout: float,
        *,
        allow_reconnect: bool = True,
    ) -> dict:
        self._check_circuit()
        if allow_reconnect:
            self._ensure_connected()
        try:
            result = self._request_once(method, params, timeout)
        except MCPError:
            self._record_failure()
            raise
        self._record_success()
        return result

    def _request_once(self, method: str, params: dict, timeout: float) -> dict:
        with self._cond:
            if self.transport == "stdio" and self._dead is not None:
                raise self._dead
            self._next_id += 1
            req_id = self._next_id
        message = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        if self.transport == "http":
            msg = self._http_exchange(message, timeout, expect_response=True)
            return self._response_result(method, msg)
        self._write(message)
        deadline = time.monotonic() + timeout
        with self._cond:
            while req_id not in self._responses:
                if self._dead is not None:
                    raise self._dead
                left = deadline - time.monotonic()
                if left <= 0:
                    raise MCPError(f"MCP server {self.name!r} gave no answer to {method} in {timeout:g}s")
                self._cond.wait(left)
            msg = self._responses.pop(req_id)
        return self._response_result(method, msg)

    def _response_result(self, method: str, msg: dict) -> dict:
        if "error" in msg:
            error = msg["error"]
            detail = error.get("message", error) if isinstance(error, dict) else error
            raise MCPError(f"MCP server {self.name!r} rejected {method}: {detail}")
        result = msg.get("result", {})
        if not isinstance(result, dict):
            raise MCPError(f"MCP server {self.name!r} returned an invalid {method} result")
        return result

    def _write(self, msg: dict):
        try:
            with self._write_lock:
                if self._proc is None or self._proc.stdin is None:
                    raise OSError("stdin is unavailable")
                self._proc.stdin.write(json.dumps(msg) + "\n")
                self._proc.stdin.flush()
        except OSError as e:
            raise MCPError(f"MCP server {self.name!r} is not writable: {e}") from e

    def _read_loop(self, proc: subprocess.Popen):
        try:
            if proc.stdout is None:
                raise OSError("stdout is unavailable")
            for line in proc.stdout:
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("MCP server %r sent a non-JSON line, skipped", self.name)
                    continue
                if "id" not in msg:
                    continue  # server notification: nothing here needs an answer
                with self._cond:
                    self._responses[msg["id"]] = msg
                    self._cond.notify_all()
        except (OSError, ValueError):
            pass
        with self._cond:
            # An old reader must not mark a newly reconnected process dead.
            if self._proc is proc and self._dead is None and not self._closed:
                self._dead = MCPError(f"MCP server {self.name!r} exited")
            self._cond.notify_all()

    def _ensure_connected(self):
        if self.transport == "http" or self._dead is None:
            return
        if not self.reconnect_enabled:
            raise self._dead
        with self._reconnect_lock:
            if self._dead is None:
                return
            self._stop_process()
            self._start_transport()
            try:
                self._recovering = True
                self._initialize()
            except Exception:
                self._stop_process()
                raise
            finally:
                self._recovering = False

    def _stop_process(self):
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except (OSError, ValueError):
                pass
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._remove_container()

    def _check_circuit(self):
        opened = self._circuit_opened_at
        if opened is None:
            return
        if time.monotonic() - opened >= self.circuit_cooldown:
            self._circuit_opened_at = None
            self._consecutive_failures = 0
            return
        left = self.circuit_cooldown - (time.monotonic() - opened)
        raise MCPError(
            f"MCP server {self.name!r} circuit is open; retry after {left:.1f}s"
        )

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.circuit_failures:
            self._circuit_opened_at = time.monotonic()

    def _record_success(self):
        if self._recovering:
            return
        self._consecutive_failures = 0
        self._circuit_opened_at = None

    def _http_exchange(self, message: dict, timeout: float, *, expect_response: bool) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self._headers,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        request = urllib.request.Request(
            self._url,
            data=json.dumps(message).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                session = response.headers.get("Mcp-Session-Id")
                if session:
                    self._session_id = session
                body = response.read().decode("utf-8", errors="replace")
                content_type = response.headers.get("Content-Type", "")
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as e:
            detail = getattr(e, "reason", None) or str(e)
            raise MCPError(f"MCP HTTP server {self.name!r} request failed: {detail}") from e
        if not expect_response and not body.strip():
            return {}
        try:
            if "text/event-stream" in content_type:
                messages = [
                    json.loads(line[5:].strip())
                    for line in body.splitlines()
                    if line.startswith("data:") and line[5:].strip()
                ]
                payload = next(
                    (item for item in reversed(messages) if "id" in item),
                    messages[-1] if messages else {},
                )
            else:
                payload = json.loads(body)
        except (json.JSONDecodeError, IndexError) as e:
            raise MCPError(f"MCP HTTP server {self.name!r} returned invalid JSON") from e
        if not isinstance(payload, dict):
            raise MCPError(f"MCP HTTP server {self.name!r} returned a non-object response")
        return payload

    def _close_http_session(self):
        if not self._session_id or not self._url:
            return
        request = urllib.request.Request(
            self._url,
            headers={**self._headers, "Mcp-Session-Id": self._session_id},
            method="DELETE",
        )
        try:
            urllib.request.urlopen(request, timeout=5).close()
        except (OSError, urllib.error.URLError, urllib.error.HTTPError):
            pass
        self._session_id = None


class MCPTool(Tool):
    """A tool living in an MCP server, registered under a mcp__ name so it can
    never collide with a built-in. Side effects are unknown, so it stays out of
    Permission.READ_ONLY and the consent gate asks first, like any mutating tool.
    """

    effect = ToolEffect.UNKNOWN
    resource_parallel = True

    def __init__(self, client: MCPClient, spec: dict):
        self._client = client
        self._remote_name = spec["name"]
        self.name = f"mcp__{client.name}__{spec['name']}"
        self.description = spec.get("description") or ""
        self.parameters = spec.get("inputSchema") or {"type": "object", "properties": {}}
        self.capabilities = client.capabilities

    def execute(self, **kwargs) -> str:
        return self._client.call_tool(self._remote_name, kwargs)

    def resource_claims(self, arguments: dict) -> tuple[ResourceClaim, ...]:
        # A server may keep shared mutable state and stdio is one session. Calls
        # to different servers may overlap, but calls to one server serialize.
        return (ResourceClaim(f"mcp-server:{self._client.name}", "write"),)


class _MCPServerAdmissionTool(Tool):
    """Synthetic Tool used to apply policy before a server is contacted."""

    description = "MCP server startup admission"
    parameters: ClassVar[dict] = {"type": "object", "properties": {}}
    effect = ToolEffect.EXTERNAL

    def __init__(self, name: str, capabilities: frozenset[str]):
        self.name = f"mcp__{name}__*"
        self.capabilities = capabilities

    def execute(self, **kwargs) -> str:  # pragma: no cover - never registered
        raise RuntimeError("MCP admission sentinel is not executable")


_live_clients: list[MCPClient] = []


def load_mcp_tools(
    path: Path = CONFIG_FILE,
    *,
    workspace: str | Path | None = None,
    capability_policy: CapabilityPolicy | None = None,
) -> list[Tool]:
    """Start the configured servers and return their tools. A missing file
    means no MCP; a broken file or a server that won't start gets one clear
    warning and the agent carries on with whatever loaded."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, OSError) as e:
        log.warning("ignoring %s: %s", path, e)
        return []
    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        log.warning("ignoring %s: 'defaults' must be an object", path)
        return []
    default_sandbox = defaults.get("sandbox")
    servers = data.get("mcpServers") or {}
    if not isinstance(servers, dict):
        log.warning("ignoring %s: 'mcpServers' must be an object", path)
        return []
    tools: list[Tool] = []
    for name, spec in servers.items():
        try:
            if not isinstance(name, str) or not name:
                raise ValueError("MCP server names must be non-empty strings")
            if not isinstance(spec, dict):
                raise TypeError("MCP server configuration must be an object")
            server_sandbox = _merge_sandbox(default_sandbox, spec.get("sandbox"))
            transport = "http" if spec.get("url") else "stdio"
            normalized_sandbox = _normalize_sandbox(server_sandbox)
            capabilities = _server_capabilities(
                transport=transport,
                sandbox=normalized_sandbox,
            )
            if capability_policy is not None:
                decision = capability_policy.decide(
                    _MCPServerAdmissionTool(name, capabilities), {}
                )
                if not decision.allowed:
                    log.warning(
                        "MCP server %r skipped before startup: %s",
                        name,
                        decision.reason,
                    )
                    continue
            common = {
                "sandbox": normalized_sandbox,
                "workspace": workspace,
                "reconnect": spec.get("reconnect", defaults.get("reconnect", True)),
                "circuit_failures": spec.get(
                    "circuit_failures",
                    defaults.get("circuit_failures", CIRCUIT_FAILURES),
                ),
                "circuit_cooldown": spec.get(
                    "circuit_cooldown",
                    defaults.get("circuit_cooldown", CIRCUIT_COOLDOWN),
                ),
            }
            if transport == "http":
                client = MCPClient(
                    name,
                    url=spec["url"],
                    headers=spec.get("headers"),
                    **common,
                )
            else:
                client = MCPClient(
                    name,
                    spec["command"],
                    _string_args(spec.get("args", [])),
                    spec.get("env"),
                    **common,
                )
        except (MCPError, OSError, KeyError, TypeError, ValueError) as e:
            log.warning("MCP server %r skipped: %s", name, e)
            continue
        _live_clients.append(client)
        tools.extend(client.tools)
    return tools


def refresh_mcp_tools(tools: list[Tool]) -> tuple[list[Tool], list[dict]]:
    """Refresh each live server once and rebuild only the MCP portion.

    A broken server keeps its previous wrappers so a transient refresh failure
    does not make capabilities disappear mid-conversation.
    """
    ordinary = [tool for tool in tools if not isinstance(tool, MCPTool)]
    clients: list[MCPClient] = []
    prior_by_client: dict[int, list[MCPTool]] = {}
    for tool in tools:
        if not isinstance(tool, MCPTool):
            continue
        prior_by_client.setdefault(id(tool._client), []).append(tool)
        if tool._client not in clients:
            clients.append(tool._client)
    refreshed: list[Tool] = []
    health: list[dict] = []
    for client in clients:
        try:
            server_tools = client.refresh_tools()
        except MCPError as e:
            server_tools = prior_by_client.get(id(client), [])
            health.append({
                "name": client.name,
                "transport": client.transport,
                "status": "refresh_failed",
                "error": str(e),
                "tool_count": len(server_tools),
            })
        else:
            health.append({
                "name": client.name,
                "transport": client.transport,
                "status": "healthy",
                "tool_count": len(server_tools),
            })
        refreshed.extend(server_tools)
    return ordinary + refreshed, health


def _string_env(env: dict | None) -> dict[str, str]:
    if env is None:
        return {}
    if not isinstance(env, dict):
        raise TypeError("MCP server env must be an object")
    return {str(key): str(value) for key, value in env.items()}


def _string_headers(headers: dict | None) -> dict[str, str]:
    if headers is None:
        return {}
    if not isinstance(headers, dict):
        raise TypeError("MCP HTTP headers must be an object")
    return {str(key): str(value) for key, value in headers.items()}


def _string_args(args) -> list[str]:
    if not isinstance(args, (list, tuple)) or not all(
        isinstance(item, (str, int, float)) for item in args
    ):
        raise TypeError("MCP server args must be a list of strings")
    return [str(item) for item in args]


def _normalize_sandbox(sandbox: dict | str | None) -> dict:
    if sandbox is None:
        return {"mode": "host"}
    if isinstance(sandbox, str):
        sandbox = {"mode": sandbox}
    if not isinstance(sandbox, dict):
        raise TypeError("MCP sandbox must be 'host', 'docker', or an object")
    normalized = dict(sandbox)
    mode = normalized.get("mode", "docker")
    if mode not in {"host", "docker"}:
        raise ValueError("MCP sandbox mode must be 'host' or 'docker'")
    normalized["mode"] = mode
    if mode == "docker":
        workspace_access = normalized.get("workspace", "none")
        if workspace_access not in {"none", "ro", "rw"}:
            raise ValueError("MCP sandbox workspace must be 'none', 'ro', or 'rw'")
        normalized["workspace"] = workspace_access
    return normalized


def _server_capabilities(*, transport: str, sandbox: dict) -> frozenset[str]:
    if transport == "http":
        return frozenset({MCP, NETWORK})
    if sandbox["mode"] == "host":
        return frozenset({
            MCP,
            PROCESS,
            NETWORK,
            FILESYSTEM_READ,
            FILESYSTEM_WRITE,
        })
    capabilities = {MCP, PROCESS}
    if sandbox.get("network", "none") != "none":
        capabilities.add(NETWORK)
    workspace_access = sandbox.get("workspace", "none")
    if workspace_access in {"ro", "rw"}:
        capabilities.add(FILESYSTEM_READ)
    if workspace_access == "rw":
        capabilities.add(FILESYSTEM_WRITE)
    return frozenset(capabilities)


def _merge_sandbox(default, override):
    if override is None:
        return default
    if isinstance(default, dict) and isinstance(override, dict):
        return {**default, **override}
    return override


@atexit.register
def _shutdown():
    for client in _live_clients:
        client.close()
