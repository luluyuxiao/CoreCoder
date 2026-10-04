"""Fallback routing and fail-closed USD budget enforcement."""

import json
import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest
from openai import APIConnectionError, AuthenticationError

from corecoder.config import Config
from corecoder.llm import (
    LLM,
    BudgetExceededError,
    LiteLLM,
    ProviderRoute,
    _request_token_upper_bound,
)


def _request():
    try:
        import httpx
    except ModuleNotFoundError:
        import httpx2 as httpx
    return httpx.Request("POST", "https://example.test/v1/chat/completions")


def _auth_error():
    try:
        import httpx
    except ModuleNotFoundError:
        import httpx2 as httpx
    request = _request()
    response = httpx.Response(401, request=request)
    return AuthenticationError("bad key", response=response, body=None)


def _stream(content="ok", prompt=10, completion=5, include_usage=True):
    chunks = [
        SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=None)
            )],
        )
    ]
    if include_usage:
        chunks.append(SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=prompt,
                completion_tokens=completion,
            ),
            choices=[],
        ))
    return iter(chunks)


def _broken_stream(before_first_chunk: bool):
    if not before_first_chunk:
        yield SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="partial", tool_calls=None)
            )],
        )
    raise APIConnectionError(request=_request())


def test_retryable_failure_switches_to_fallback_and_stays_there():
    llm = LLM(
        model="gpt-4o",
        fallback_models=["gpt-4o-mini"],
        api_key="test",
    )
    call = mock.Mock(side_effect=[
        APIConnectionError(request=_request()),
        _stream("fallback"),
        _stream("sticky"),
    ])
    llm._call_with_retry = call

    assert llm.chat([{"role": "user", "content": "hi"}]).content == "fallback"
    assert llm.model == "gpt-4o-mini"
    assert llm.fallback_history[0][:2] == ("gpt-4o", "gpt-4o-mini")

    assert llm.chat([{"role": "user", "content": "again"}]).content == "sticky"
    assert [entry.args[0]["model"] for entry in call.call_args_list] == [
        "gpt-4o", "gpt-4o-mini", "gpt-4o-mini",
    ]


def test_authentication_error_does_not_fallback():
    llm = LLM(
        model="gpt-4o",
        fallback_models=["gpt-4o-mini"],
        api_key="test",
    )
    call = mock.Mock(side_effect=_auth_error())
    llm._call_with_retry = call

    with pytest.raises(AuthenticationError):
        llm.chat([{"role": "user", "content": "hi"}])
    assert call.call_count == 1
    assert llm.model == "gpt-4o"


def test_stream_failure_before_first_chunk_can_fallback():
    llm = LLM(
        model="gpt-4o",
        fallback_models=["gpt-4o-mini"],
        api_key="test",
    )
    call = mock.Mock(side_effect=[_broken_stream(True), _stream("fallback")])
    llm._call_with_retry = call

    assert llm.chat([{"role": "user", "content": "hi"}]).content == "fallback"
    assert llm.model == "gpt-4o-mini"


def test_stream_failure_after_output_does_not_duplicate_via_fallback():
    llm = LLM(
        model="gpt-4o",
        fallback_models=["gpt-4o-mini"],
        api_key="test",
    )
    call = mock.Mock(return_value=_broken_stream(False))
    llm._call_with_retry = call
    emitted = []

    with pytest.raises(APIConnectionError):
        llm.chat(
            [{"role": "user", "content": "hi"}],
            on_token=emitted.append,
        )
    assert emitted == ["partial"]
    assert call.call_count == 1
    assert llm.model == "gpt-4o"


def test_litellm_uses_same_fallback_policy_for_transient_errors():
    llm = LiteLLM(
        model="openai/gpt-4o",
        fallback_models=["openai/gpt-4o-mini"],
    )
    call = mock.Mock(side_effect=[RuntimeError("provider timeout"), _stream()])
    llm._call_with_retry = call

    assert llm.chat([{"role": "user", "content": "hi"}]).content == "ok"
    assert llm.model == "openai/gpt-4o-mini"


def test_cost_is_accumulated_at_each_models_own_price():
    llm = LLM(model="gpt-4o", api_key="test")
    llm._call_with_retry = mock.Mock(side_effect=[
        _stream(prompt=1_000, completion=500),
        _stream(prompt=2_000, completion=1_000),
    ])

    llm.chat([{"role": "user", "content": "first"}])
    llm.model = "deepseek-chat"
    llm.chat([{"role": "user", "content": "second"}])

    expected = (1_000 * 2.5 + 500 * 10 + 2_000 * 0.27 + 1_000 * 1.10) / 1_000_000
    assert llm.estimated_cost == pytest.approx(expected)
    assert set(llm.usage_by_model) == {"gpt-4o", "deepseek-chat"}


def test_budget_reserves_input_and_clamps_max_output_tokens():
    budget = 0.001
    messages = [{"role": "user", "content": "short request"}]
    llm = LLM(
        model="gpt-4o-mini",
        api_key="test",
        max_tokens=4096,
        max_cost_usd=budget,
    )
    call = mock.Mock(return_value=_stream(prompt=20, completion=10))
    llm._call_with_retry = call

    llm.chat(messages)
    params = call.call_args.args[0]
    assert 0 < params["max_tokens"] < 4096
    input_reserve = _request_token_upper_bound(messages, None) * 0.15 / 1_000_000
    assert input_reserve + params["max_tokens"] * 0.6 / 1_000_000 <= budget


def test_exhausted_budget_refuses_request_before_calling_provider():
    llm = LLM(model="gpt-4o-mini", api_key="test", max_cost_usd=0.01)
    llm.usage_by_model = {
        "gpt-4o-mini": {"prompt": 1_000_000, "completion": 0}
    }
    call = mock.Mock()
    llm._call_with_retry = call

    with pytest.raises(BudgetExceededError, match="exhausted"):
        llm.chat([{"role": "user", "content": "hi"}])
    call.assert_not_called()


def test_budget_rejects_unknown_pricing_before_calling_provider():
    llm = LLM(model="private-model", api_key="test", max_cost_usd=1)
    call = mock.Mock()
    llm._call_with_retry = call

    with pytest.raises(BudgetExceededError, match="no pricing"):
        llm.chat([{"role": "user", "content": "hi"}])
    call.assert_not_called()


def test_budget_stops_when_provider_omits_usage():
    llm = LLM(model="gpt-4o-mini", api_key="test", max_cost_usd=1)
    llm._call_with_retry = mock.Mock(return_value=_stream(include_usage=False))

    with pytest.raises(BudgetExceededError, match="no token usage"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_config_reads_fallback_chain_and_budget(monkeypatch):
    monkeypatch.setenv("CORECODER_FALLBACK_MODELS", "gpt-4o-mini, deepseek-chat")
    monkeypatch.setenv("CORECODER_MAX_COST_USD", "2.5")

    config = Config.from_env()
    assert config.fallback_models == ["gpt-4o-mini", "deepseek-chat"]
    assert config.max_cost_usd == 2.5


def test_cli_accepts_ordered_fallbacks_and_positive_budget(monkeypatch):
    from corecoder.cli import _parse_args

    monkeypatch.setattr("sys.argv", [
        "corecoder",
        "--fallback-model", "gpt-4o-mini",
        "--fallback-model", "deepseek-chat",
        "--max-cost", "0.5",
    ])
    args = _parse_args()

    assert args.fallback_models == ["gpt-4o-mini", "deepseek-chat"]
    assert args.max_cost_usd == 0.5


def test_fallback_route_switches_base_url_credentials_and_is_sticky():
    with mock.patch("corecoder.llm.OpenAI") as factory:
        primary_client = mock.Mock(name="primary-client")
        fallback_client = mock.Mock(name="fallback-client")
        factory.side_effect = [primary_client, fallback_client]
        llm = LLM(
            model="primary-model",
            api_key="primary-key",
            base_url="https://primary.example/v1",
            fallback_routes=[ProviderRoute(
                name="deepseek",
                provider="openai",
                model="deepseek-chat",
                api_key="fallback-key",
                base_url="https://api.deepseek.example/v1",
            )],
        )

        calls = []

        def dispatch(params):
            calls.append((llm.client, params["model"]))
            if params["model"] == "primary-model":
                raise APIConnectionError(request=_request())
            return _stream("fallback")

        llm._call_with_retry = mock.Mock(side_effect=dispatch)

        assert llm.chat([{"role": "user", "content": "hi"}]).content == "fallback"
        assert llm.active_route.route_id == "deepseek"
        assert llm.model == "deepseek-chat"
        assert calls == [
            (primary_client, "primary-model"),
            (fallback_client, "deepseek-chat"),
        ]
        assert factory.call_args_list == [
            mock.call(api_key="primary-key", base_url="https://primary.example/v1"),
            mock.call(api_key="fallback-key", base_url="https://api.deepseek.example/v1"),
        ]


def test_openai_primary_can_fallback_to_litellm_route(monkeypatch):
    fake = types.ModuleType("litellm")
    fake.completion = mock.Mock(return_value=_stream("from litellm", prompt=20, completion=4))
    monkeypatch.setitem(sys.modules, "litellm", fake)
    llm = LLM(
        model="gpt-4o",
        api_key="primary",
        fallback_routes=[{
            "name": "claude",
            "provider": "litellm",
            "model": "anthropic/claude-haiku-4-5",
            "api_key": "anthropic-key",
            "base_url": "https://litellm.example",
        }],
    )
    llm._call_with_retry = mock.Mock(side_effect=APIConnectionError(request=_request()))

    response = llm.chat([{"role": "user", "content": "hi"}])

    assert response.content == "from litellm"
    assert llm.active_route.provider == "litellm"
    kwargs = fake.completion.call_args.kwargs
    assert kwargs["model"] == "anthropic/claude-haiku-4-5"
    assert kwargs["api_key"] == "anthropic-key"
    assert kwargs["api_base"] == "https://litellm.example"


def test_route_specific_price_is_used_and_secret_is_not_public():
    route = ProviderRoute(
        name="private",
        model="private-model",
        api_key="never-persist-me",
        input_price=2.0,
        output_price=6.0,
    )
    llm = LLM(model="primary", api_key="primary", fallback_routes=[route])
    llm._activate_route(route)
    llm._call_with_retry = mock.Mock(return_value=_stream(prompt=1_000, completion=500))

    llm.chat([{"role": "user", "content": "usage"}])

    assert llm.estimated_cost == pytest.approx((2_000 + 3_000) / 1_000_000)
    assert "api_key" not in route.public_dict()


def test_same_model_endpoint_can_fallback_to_a_secondary_credential():
    llm = LLM(
        model="gpt-4o",
        api_key="primary-key",
        base_url="https://gateway.example/v1",
        fallback_routes=[{
            "name": "secondary-account",
            "model": "gpt-4o",
            "base_url": "https://gateway.example/v1",
            "api_key": "secondary-key",
        }],
    )

    assert [route.route_id for route in llm.routes] == ["primary", "secondary-account"]
    assert llm.fallback_routes[0].api_key == "secondary-key"


def test_config_reads_cross_provider_routes_from_json(monkeypatch):
    monkeypatch.setenv("FALLBACK_SECRET", "secret")
    monkeypatch.setenv("CORECODER_FALLBACK_ROUTES", json.dumps([{
        "name": "qwen",
        "provider": "openai",
        "model": "qwen3-plus",
        "base_url": "https://dashscope.example/v1",
        "api_key_env": "FALLBACK_SECRET",
    }]))

    config = Config.from_env()
    llm = LLM(model="gpt-4o", api_key="primary", fallback_routes=config.fallback_routes)

    assert config.fallback_routes[0]["model"] == "qwen3-plus"
    assert llm.fallback_routes[0].api_key == "secret"
