"""Named model profiles: non-secret provider presets and REPL switching."""

import json

import pytest

from corecoder import LLM, Agent, Config, cli
from corecoder.config import (
    ModelProfile,
    ModelProfiles,
    apply_model_profile,
    load_model_profiles,
)
from corecoder.demo import ScriptedLLM


def _write_profiles(tmp_path, data):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_model_profiles_load_and_apply_a_complete_route(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    path = _write_profiles(tmp_path, {
        "default": "deepseek",
        "models": {
            "deepseek": {
                "description": "fast coding preset",
                "provider": "openai",
                "model": "deepseek-chat",
                "base_url": "https://api.deepseek.example/v1",
                "api_key_env": "DEEPSEEK_API_KEY",
                "fallback_models": ["deepseek-reasoner"],
                "fallback_routes": [{"name": "cheap", "model": "small"}],
                "max_cost_usd": 0.25,
                "max_tokens": 2048,
                "temperature": 0.2,
            },
        },
    })

    profiles = load_model_profiles(path)
    applied = apply_model_profile(
        Config(model="base", api_key="base-secret", base_url="https://base.example"),
        profiles.require("deepseek"),
    )

    assert profiles.default == "deepseek"
    assert applied.model_profile == "deepseek"
    assert applied.model == "deepseek-chat"
    assert applied.provider == "openai"
    assert applied.base_url == "https://api.deepseek.example/v1"
    assert applied.api_key == "deepseek-secret"
    assert applied.fallback_models == ["deepseek-reasoner"]
    assert applied.fallback_routes == [{"name": "cheap", "model": "small"}]
    assert applied.max_cost_usd == 0.25
    assert applied.max_tokens == 2048
    assert applied.temperature == 0.2


def test_model_profile_missing_explicit_credential_fails_closed(tmp_path):
    path = _write_profiles(tmp_path, {
        "models": {
            "private": {
                "model": "private-model",
                "api_key_env": "MISSING_PROFILE_KEY",
            },
        },
    })

    with pytest.raises(ValueError, match="MISSING_PROFILE_KEY"):
        apply_model_profile(Config(api_key="unrelated-key"), load_model_profiles(path).require("private"))


def test_model_profile_rejects_unknown_default(tmp_path):
    path = _write_profiles(tmp_path, {
        "default": "not-configured",
        "models": {"local": {"model": "qwen3"}},
    })

    with pytest.raises(ValueError, match="default must name"):
        load_model_profiles(path)


def test_repl_profile_switch_replaces_provider_and_keeps_usage(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "qwen-secret")
    baseline = Config(
        model="old-model",
        api_key="old-secret",
        base_url="https://old.example/v1",
        fallback_models=["old-fallback"],
    )
    agent = Agent(llm=LLM(model="old-model", api_key="old-secret", base_url=baseline.base_url))
    agent.llm.total_prompt_tokens = 12
    agent.llm.total_completion_tokens = 5
    agent.llm.usage_by_route = {
        "primary": {
            "prompt": 12,
            "completion": 5,
            "model": "old-model",
            "provider": "openai",
            "input_price": 1.0,
            "output_price": 2.0,
        },
    }
    profile = ModelProfile(
        name="qwen",
        model="qwen3-coder",
        provider="openai",
        base_url="https://qwen.example/v1",
        api_key_env="QWEN_API_KEY",
    )

    updated = cli._switch_model_profile(agent, baseline, baseline, profile)

    assert updated.model_profile == "qwen"
    assert agent.model_profile == "qwen"
    assert agent.llm.model == "qwen3-coder"
    assert agent.llm.active_route.route_id == "profile:qwen"
    assert agent.llm.active_route.base_url == "https://qwen.example/v1"
    assert agent.llm.total_prompt_tokens == 12
    assert agent.llm.total_completion_tokens == 5
    assert agent.llm.usage_by_route["primary"]["input_price"] == 1.0


def test_model_profile_name_is_session_metadata():
    agent = Agent(llm=LLM(model="demo", api_key="key"))
    agent.model_profile = "demo-profile"
    restored = Agent(llm=LLM(model="other", api_key="key"))

    restored.restore_state(agent.state_snapshot())

    assert restored.model_profile == "demo-profile"


def test_repl_model_command_selects_a_named_profile(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "qwen-secret")
    agent = Agent(llm=ScriptedLLM([]))
    baseline = Config(model="old", api_key="old-secret", base_url="https://old.example")
    profiles = ModelProfiles(profiles={
        "qwen": ModelProfile(
            name="qwen",
            model="qwen3-coder",
            base_url="https://qwen.example/v1",
            api_key_env="QWEN_API_KEY",
        ),
    })
    inputs = iter(["/model qwen", "quit"])
    monkeypatch.setattr(cli, "pt_prompt", lambda *args, **kwargs: next(inputs))

    cli._repl(
        agent,
        baseline,
        model_profiles=profiles,
        baseline_config=baseline,
    )

    assert agent.model_profile == "qwen"
    assert agent.llm.model == "qwen3-coder"
    assert agent.llm.active_route.base_url == "https://qwen.example/v1"


def test_bare_model_command_uses_interactive_profile_picker(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "qwen-secret")
    agent = Agent(llm=ScriptedLLM([]))
    baseline = Config(model="old", api_key="old-secret")
    profiles = ModelProfiles(profiles={
        "qwen": ModelProfile(
            name="qwen",
            model="qwen3-coder",
            base_url="https://qwen.example/v1",
            api_key_env="QWEN_API_KEY",
        ),
    })
    inputs = iter(["/model", "quit"])
    monkeypatch.setattr(cli, "pt_prompt", lambda *args, **kwargs: next(inputs))
    monkeypatch.setattr(cli, "_select_model_profile", lambda profiles, active: "qwen")

    cli._repl(
        agent,
        baseline,
        model_profiles=profiles,
        baseline_config=baseline,
    )

    assert agent.model_profile == "qwen"
    assert agent.llm.model == "qwen3-coder"
