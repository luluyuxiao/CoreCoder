"""Per-tool capability authorization and network authority metadata."""

import json
from typing import ClassVar

import pytest

from corecoder.agent import Agent
from corecoder.capabilities import (
    FILESYSTEM_READ,
    NETWORK,
    UNKNOWN,
    load_capability_policy,
)
from corecoder.config import Config
from corecoder.demo import ScriptedLLM
from corecoder.llm import LLMResponse, ToolCall
from corecoder.permissions import Permission
from corecoder.sandbox import DockerCommandExecutor, LocalCommandExecutor
from corecoder.tools.base import Tool, ToolEffect
from corecoder.tools.bash import BashTool


class _NetworkTool(Tool):
    name = "network_probe"
    effect = ToolEffect.EXTERNAL
    capabilities = frozenset({NETWORK})
    description = "test"
    parameters: ClassVar[dict] = {"type": "object", "properties": {}}

    def __init__(self):
        self.executed = False

    def execute(self) -> str:
        self.executed = True
        return "network used"


class _UnknownTool(Tool):
    name = "custom_unknown"
    description = "test"
    parameters: ClassVar[dict] = {"type": "object", "properties": {}}

    def execute(self) -> str:
        return "ran"


def _write_policy(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_capability_policy(path)


def test_missing_default_policy_preserves_backwards_compatible_allow(tmp_path, monkeypatch):
    monkeypatch.setattr("corecoder.capabilities.DEFAULT_POLICY_FILE", tmp_path / "missing.json")
    policy = load_capability_policy()

    decision, result = policy.decide(_UnknownTool(), {})

    assert policy.enabled is False
    assert decision == "capability_allow_all"
    assert result is None


def test_default_deny_blocks_unknown_custom_tools(tmp_path):
    policy = _write_policy(tmp_path / "policy.json", {"default": "deny"})

    decision, result = policy.decide(_UnknownTool(), {})

    assert decision == "capability_deny"
    assert UNKNOWN in result


def test_exact_rule_beats_wildcard_and_allows_declared_capability(tmp_path):
    policy = _write_policy(tmp_path / "policy.json", {
        "default": "deny",
        "tools": {
            "network_*": {"allow": []},
            "network_probe": {"allow": [NETWORK]},
        },
    })

    assert policy.decide(_NetworkTool(), {}) == ("capability_allow", None)


def test_capability_denial_happens_before_permission_and_execution(tmp_path):
    policy = _write_policy(tmp_path / "policy.json", {
        "default": "allow",
        "tools": {"network_probe": {"allow": []}},
    })
    tool = _NetworkTool()
    asked = []
    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall(
                id="network-1",
                name="network_probe",
                arguments={},
            )]),
            LLMResponse(content="handled"),
        ]),
        tools=[tool],
        permission=Permission(ask=lambda name, arguments: asked.append(name) or "once"),
        capability_policy=policy,
    )

    assert agent.chat("probe") == "handled"
    assert tool.executed is False
    assert asked == []
    assert "Capability denied" in agent.messages[2]["content"]


def test_policy_validation_rejects_unknown_capability(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({
        "tools": {"fetch_url": {"allow": ["teleport"]}},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown values"):
        load_capability_policy(path)


def test_bash_network_capability_reflects_enforced_executor_mode(tmp_path):
    local = BashTool(LocalCommandExecutor(tmp_path))
    offline = BashTool(DockerCommandExecutor(tmp_path, network="none"))
    online = BashTool(DockerCommandExecutor(tmp_path, network="bridge"))

    assert NETWORK in local.capabilities
    assert NETWORK not in offline.capabilities
    assert NETWORK in online.capabilities
    assert FILESYSTEM_READ in offline.capabilities


def test_capability_policy_config_from_env(monkeypatch):
    monkeypatch.setenv("CORECODER_CAPABILITY_POLICY", "policy/capabilities.json")
    assert Config.from_env().capability_policy_path == "policy/capabilities.json"


def test_capability_policy_cli_flag(monkeypatch):
    from corecoder.cli import _parse_args

    monkeypatch.setattr(
        "sys.argv",
        ["corecoder", "--capability-policy", "policy.json"],
    )
    assert _parse_args().capability_policy == "policy.json"
