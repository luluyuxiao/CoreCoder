"""Fallback routing and fail-closed USD budget enforcement."""

from types import SimpleNamespace
from unittest import mock

import pytest
from openai import APIConnectionError, AuthenticationError

from corecoder.config import Config
from corecoder.llm import LLM, BudgetExceededError, LiteLLM, _request_token_upper_bound


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
