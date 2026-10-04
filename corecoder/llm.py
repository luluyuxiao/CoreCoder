"""LLM provider layer - thin wrapper over OpenAI-compatible APIs.

Since most providers (DeepSeek, Qwen, Kimi, GLM, Ollama, etc.) expose an
OpenAI-compatible endpoint, we just use the openai SDK directly.  Switch
provider by changing OPENAI_BASE_URL + OPENAI_API_KEY. That's it.

For providers that are NOT OpenAI-compatible (AWS Bedrock, Google Vertex,
etc.), use the LiteLLM backend which routes to 100+ providers through a
single unified interface. Set CORECODER_PROVIDER=litellm.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, replace
from itertools import chain

from openai import APIConnectionError, APIError, APITimeoutError, BadRequestError, OpenAI, RateLimitError

log = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    @property
    def message(self) -> dict:
        """Convert to OpenAI message format for appending to history."""
        msg: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


@dataclass(frozen=True)
class ProviderRoute:
    """One independently configured model/provider fallback destination."""

    model: str
    provider: str = "openai"
    name: str = ""
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    input_price: float | None = None
    output_price: float | None = None

    def __post_init__(self):
        if not self.model.strip():
            raise ValueError("provider route model must not be empty")
        if self.provider not in {"openai", "litellm"}:
            raise ValueError("provider route provider must be 'openai' or 'litellm'")
        supplied = self.input_price is not None or self.output_price is not None
        if supplied and (self.input_price is None or self.output_price is None):
            raise ValueError("provider route pricing requires both input_price and output_price")
        if self.input_price is not None and (self.input_price < 0 or self.output_price < 0):
            raise ValueError("provider route pricing must not be negative")

    @property
    def route_id(self) -> str:
        return self.name or f"{self.provider}:{self.model}@{self.base_url or 'default'}"

    def pricing(self) -> tuple[float, float] | None:
        if self.input_price is not None and self.output_price is not None:
            return self.input_price, self.output_price
        return _pricing_for_model(self.model)

    def public_dict(self) -> dict:
        """Return trace/session-safe metadata with credentials removed."""
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "input_price": self.input_price,
            "output_price": self.output_price,
        }

    @classmethod
    def from_value(
        cls,
        value: "ProviderRoute | dict",
        *,
        default_provider: str,
        default_api_key: str | None,
        default_base_url: str | None,
    ) -> "ProviderRoute":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("fallback route must be a ProviderRoute or object")
        provider = str(value.get("provider") or default_provider).lower()
        api_key_env = value.get("api_key_env")
        if api_key_env is not None and not isinstance(api_key_env, str):
            raise TypeError("provider route api_key_env must be a string")
        api_key = value.get("api_key")
        if api_key is None and api_key_env:
            api_key = os.getenv(api_key_env)
        if api_key is None and provider == default_provider:
            api_key = default_api_key
        base_url = value.get("base_url")
        if base_url is None and provider == default_provider:
            base_url = default_base_url
        return cls(
            name=str(value.get("name") or ""),
            provider=provider,
            model=str(value.get("model") or ""),
            base_url=str(base_url) if base_url else None,
            api_key=str(api_key) if api_key else None,
            input_price=(
                float(value["input_price"])
                if value.get("input_price") is not None else None
            ),
            output_price=(
                float(value["output_price"])
                if value.get("output_price") is not None else None
            ),
        )


# pricing per million tokens: (input, output)
# sources: openai.com/api/pricing, api-docs.deepseek.com, platform.claude.com,
#          platform.moonshot.ai, alibabacloud.com/help/en/model-studio
_PRICING = {
    # OpenAI - current flagships
    "gpt-5.5": (5, 30),
    "gpt-5.4": (2.5, 15),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4-nano": (0.2, 1.25),
    "o4-mini": (1.1, 4.4),
    # OpenAI - previous gen (still widely used)
    "gpt-4.1": (2, 8),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "gpt-4o": (2.5, 10),
    "gpt-4o-mini": (0.15, 0.6),
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    # Anthropic Claude
    "claude-opus-4-6": (5, 25),
    "claude-sonnet-4-6": (3, 15),
    "claude-haiku-4-5": (1, 5),
    # Alibaba Qwen
    "qwen3-max": (0.78, 3.9),
    "qwen3-plus": (0.26, 0.78),
    "qwen-max": (0.78, 3.9),
    # Moonshot Kimi
    "kimi-k2.5": (0.6, 3),
}


class BudgetExceededError(RuntimeError):
    """The configured USD budget cannot safely fund another model request."""


def _pricing_for_model(model: str) -> tuple[float, float] | None:
    """Find pricing for plain and LiteLLM-style provider/model names."""
    return _PRICING.get(model) or _PRICING.get(model.rsplit("/", 1)[-1])


def _request_token_upper_bound(messages: list[dict], tools: list[dict] | None) -> int:
    """Conservative token ceiling for budget reservation.

    A tokenizer token cannot encode less than one UTF-8 byte. Counting the
    serialized request bytes plus protocol headroom therefore intentionally
    overestimates normal chat inputs without adding a tokenizer dependency.
    """
    payload = json.dumps(
        {"messages": messages, "tools": tools or []},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return len(payload.encode("utf-8")) + 256 + 16 * (len(messages) + len(tools or []))


def _adapt_rejected_param(params: dict, exc: BadRequestError) -> bool:
    """Translate or drop one parameter a provider rejected with a 400.

    Newer OpenAI models (gpt-5, o-series) accept only ``max_completion_tokens``
    and the default temperature; the 400 message quotes the offender
    (``'max_tokens'``). Quoted matching is load-bearing: the rejection text
    for ``max_completion_tokens`` must not re-trigger the translation.
    Returns False when nothing recognizable was rejected.
    """
    message = str(exc).lower()
    if "'max_tokens'" in message and "max_tokens" in params:
        params["max_completion_tokens"] = params.pop("max_tokens")
        return True
    if "'temperature'" in message and "temperature" in params:
        params.pop("temperature")
        return True
    return False


class LLM:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        fallback_models: list[str] | tuple[str, ...] | None = None,
        fallback_routes: list[ProviderRoute | dict] | tuple[ProviderRoute | dict, ...] | None = None,
        max_cost_usd: float | None = None,
        **kwargs,
    ):
        self._init_policy(
            model,
            fallback_models,
            max_cost_usd,
            provider="openai",
            api_key=api_key,
            base_url=base_url,
            fallback_routes=fallback_routes,
        )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._openai_clients = {
            (api_key or "", base_url or ""): self.client,
        }
        self.extra = kwargs  # temperature, max_tokens, etc.

    def _init_policy(
        self,
        model: str,
        fallback_models: list[str] | tuple[str, ...] | None,
        max_cost_usd: float | None,
        *,
        provider: str,
        api_key: str | None,
        base_url: str | None,
        fallback_routes: list[ProviderRoute | dict] | tuple[ProviderRoute | dict, ...] | None,
    ):
        if max_cost_usd is not None and max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be greater than zero")
        self.model = model
        self.fallback_models = [
            candidate
            for candidate in dict.fromkeys(fallback_models or [])
            if candidate and candidate != model
        ]
        primary = ProviderRoute(
            name="primary",
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        explicit = [
            ProviderRoute.from_value(
                route,
                default_provider=provider,
                default_api_key=api_key,
                default_base_url=base_url,
            )
            for route in (fallback_routes or [])
        ]
        legacy = [
            ProviderRoute(
                name=f"fallback-{index + 1}",
                provider=provider,
                model=candidate,
                base_url=base_url,
                api_key=api_key,
            )
            for index, candidate in enumerate(self.fallback_models)
        ]
        seen = {primary.route_id}
        routes = [primary]
        for route in [*explicit, *legacy]:
            if route.route_id not in seen and not (
                route.model == primary.model
                and route.provider == primary.provider
                and route.base_url == primary.base_url
                and route.api_key == primary.api_key
            ):
                routes.append(route)
                seen.add(route.route_id)
        self.routes = routes
        self.fallback_routes = routes[1:]
        self._active_route = primary
        self.max_cost_usd = max_cost_usd
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.usage_by_model: dict[str, dict[str, int]] = {}
        self.usage_by_route: dict[str, dict] = {}
        self.fallback_history: list[tuple[str, str, str]] = []
        self._budget_usage_missing = False
        self._event_local = threading.local()

    @property
    def estimated_cost(self) -> float | None:
        """Rough cumulative USD cost, including every model used by fallback."""
        route_usage = getattr(self, "usage_by_route", None)
        if route_usage:
            total = 0.0
            routes = {route.route_id: route for route in self.routes}
            for route_id, tokens in route_usage.items():
                route = routes.get(route_id)
                pricing = route.pricing() if route is not None else None
                if pricing is None and tokens.get("input_price") is not None:
                    pricing = (tokens["input_price"], tokens["output_price"])
                if pricing is None:
                    return None
                input_rate, output_rate = pricing
                total += (
                    tokens["prompt"] * input_rate / 1_000_000
                    + tokens["completion"] * output_rate / 1_000_000
                )
            return total
        usage = getattr(self, "usage_by_model", None)
        if usage:
            total = 0.0
            for model, tokens in usage.items():
                pricing = _pricing_for_model(model)
                if not pricing:
                    return None
                input_rate, output_rate = pricing
                total += (
                    tokens["prompt"] * input_rate / 1_000_000
                    + tokens["completion"] * output_rate / 1_000_000
                )
            return total

        # Compatibility for callers/tests that populated the old counters
        # directly instead of going through chat().
        pricing = _pricing_for_model(self.model)
        if not pricing:
            return None
        input_rate, output_rate = pricing
        return (
            self.total_prompt_tokens * input_rate / 1_000_000
            + self.total_completion_tokens * output_rate / 1_000_000
        )

    @property
    def remaining_budget(self) -> float | None:
        if self.max_cost_usd is None:
            return None
        spent = self.estimated_cost
        if spent is None:
            return None
        return max(0.0, self.max_cost_usd - spent)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
        on_event=None,
    ) -> LLMResponse:
        """Send one request while binding provider events to this call."""
        previous = getattr(self._event_local, "callback", None)
        self._event_local.callback = on_event
        try:
            return self._chat_impl(messages, tools, on_token)
        finally:
            self._event_local.callback = previous

    def _chat_impl(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
    ) -> LLMResponse:
        """Send messages, stream back response, handle tool calls."""
        base_params: dict = {
            "messages": messages,
            "stream": True,
            **self.extra,
        }
        if tools:
            base_params["tools"] = tools

        # Provider-dialect fallbacks. A 400 that quotes a parameter is adapted
        # (newer OpenAI models take max_completion_tokens, default temperature
        # only); a 400 that names nothing drops stream_options once, matching
        # the previous single-fallback behavior for servers that reject the
        # OpenAI extension silently. The loop is bounded by the param set, and
        # an unrecognized 400 re-raises instead of doubling _call_with_retry's
        # exhausted retries. LiteLLM never lands here: drop_params strips
        # unsupported keys on that path.
        candidates = self._candidate_routes()
        starting_model = self.model
        starting_route = candidates[0]
        last_fallback_error: Exception | None = None
        response_model = self.model
        response_route = starting_route
        for position, route in enumerate(candidates):
            candidate = route.model
            params = {**base_params, "model": candidate}
            params["stream_options"] = {"include_usage": True}
            self._apply_budget_limit(params, route)
            try:
                while True:
                    try:
                        stream = self._call_route(params, route)
                        break
                    except BadRequestError as e:
                        if _adapt_rejected_param(params, e):
                            self._emit_event(
                                "llm.provider_adapted",
                                model=candidate,
                                provider=route.provider,
                                route=route.route_id,
                                reason="rejected_parameter",
                            )
                            continue
                        if "stream_options" not in params:
                            raise
                        if self.max_cost_usd is not None:
                            raise BudgetExceededError(
                                "the provider rejected a request while usage reporting was "
                                "enabled; refusing an unmetered compatibility retry"
                            ) from e
                        params.pop("stream_options")
                        self._emit_event(
                            "llm.provider_adapted",
                            model=candidate,
                            provider=route.provider,
                            route=route.route_id,
                            reason="stream_options_removed",
                        )
                # Stream creation can succeed and the iterator can still fail
                # before yielding anything. That is safe to fallback from: no
                # token has reached the callback and no partial answer exists.
                iterator = iter(stream)
                try:
                    first_chunk = next(iterator)
                except StopIteration:
                    stream = iter(())
                else:
                    stream = chain((first_chunk,), iterator)
            except Exception as e:
                has_fallback = position < len(candidates) - 1
                retryable = (
                    self._is_litellm_transient(e)
                    if route.provider == "litellm" else self._is_fallback_error(e)
                )
                if not has_fallback or not retryable:
                    raise
                next_route = candidates[position + 1]
                next_model = next_route.model
                log.warning(
                    "model %r unavailable after retries (%s); trying fallback %r",
                    candidate,
                    e,
                    next_model,
                )
                last_fallback_error = e
                self._emit_event(
                    "llm.fallback",
                    from_model=candidate,
                    to_model=next_model,
                    from_route=route.route_id,
                    to_route=next_route.route_id,
                    from_provider=route.provider,
                    to_provider=next_route.provider,
                    error_type=type(e).__name__,
                    status_code=getattr(e, "status_code", None),
                )
                continue

            response_model = candidate
            response_route = route
            if route.route_id != starting_route.route_id:
                self._activate_route(route)  # sticky: skip a dead primary next round
                self.fallback_history.append(
                    (starting_model, candidate, str(last_fallback_error or "unavailable"))
                )
            else:
                self._activate_route(route)
            break

        content_parts: list[str] = []
        tc_map: dict[int, dict] = {}  # index -> {id, name, arguments_str}
        prompt_tok = 0
        completion_tok = 0

        for chunk in stream:
            # usage info comes in the final chunk; getattr-style reads stay safe
            # across OpenAI SDK objects and litellm's provider-varying shapes
            usage = getattr(chunk, "usage", None)
            if usage:
                # some providers send usage with null fields; coerce to 0 so the
                # running totals below don't blow up on int + None
                prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
                completion_tok = getattr(usage, "completion_tokens", 0) or 0

            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta

            # accumulate text
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                if on_token:
                    on_token(delta.content)

            # accumulate tool calls across chunks
            if getattr(delta, "tool_calls", None):
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_map:
                        tc_map[idx] = {"id": "", "name": "", "args": ""}
                    if tc_delta.id:
                        tc_map[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc_map[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_map[idx]["args"] += tc_delta.function.arguments

        # parse accumulated tool calls
        parsed: list[ToolCall] = []
        for idx in sorted(tc_map):
            raw = tc_map[idx]
            try:
                args = json.loads(raw["args"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            parsed.append(ToolCall(id=raw["id"], name=raw["name"], arguments=args))

        self.total_prompt_tokens += prompt_tok
        self.total_completion_tokens += completion_tok
        usage = self.usage_by_model.setdefault(
            response_model, {"prompt": 0, "completion": 0}
        )
        usage["prompt"] += prompt_tok
        usage["completion"] += completion_tok
        route_usage = self.usage_by_route.setdefault(
            response_route.route_id,
            {
                "prompt": 0,
                "completion": 0,
                "model": response_route.model,
                "provider": response_route.provider,
                "input_price": (
                    response_route.pricing()[0]
                    if response_route.pricing() is not None else None
                ),
                "output_price": (
                    response_route.pricing()[1]
                    if response_route.pricing() is not None else None
                ),
            },
        )
        route_usage["prompt"] += prompt_tok
        route_usage["completion"] += completion_tok

        if self.max_cost_usd is not None:
            if prompt_tok + completion_tok == 0:
                self._budget_usage_missing = True
                raise BudgetExceededError(
                    "the provider returned no token usage; the USD budget can no longer "
                    "be enforced safely"
                )
            spent = self.estimated_cost
            if spent is None or spent > self.max_cost_usd + 1e-12:
                raise BudgetExceededError(
                    f"USD budget exhausted (${spent or 0:.6f} spent / "
                    f"${self.max_cost_usd:.6f} limit)"
                )

        return LLMResponse(
            content="".join(content_parts),
            tool_calls=parsed,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            model=response_model,
        )

    def _emit_event(self, event: str, **fields):
        callback = getattr(self._event_local, "callback", None)
        if callback is None:
            return
        try:
            callback(event, fields)
        except Exception as e:  # noqa: BLE001
            log.warning("LLM event callback failed for %s: %s", event, e)

    def _candidate_routes(self) -> list[ProviderRoute]:
        active = self._active_route
        if self.model != active.model:
            active = replace(active, model=self.model, name="manual")
        matching = next(
            (
                index for index, route in enumerate(self.routes)
                if route.route_id == active.route_id
            ),
            None,
        )
        if matching is None:
            return [active, *[
                route for route in self.fallback_routes
                if route.model != active.model or route.provider != active.provider
            ]]
        return self.routes[matching:]

    def _activate_route(self, route: ProviderRoute):
        previous = self._active_route
        self._active_route = route
        self.model = route.model
        if route.provider == "openai":
            if not (
                previous.provider == "openai"
                and previous.api_key == route.api_key
                and previous.base_url == route.base_url
            ):
                self.client = self._openai_client(route)
        else:
            self.api_key = route.api_key
            self.base_url = route.base_url

    @property
    def active_route(self) -> ProviderRoute:
        return self._active_route

    def _openai_client(self, route: ProviderRoute):
        clients = getattr(self, "_openai_clients", None)
        if clients is None:
            clients = {}
            self._openai_clients = clients
        key = (route.api_key or "", route.base_url or "")
        client = clients.get(key)
        if client is None:
            client = OpenAI(api_key=route.api_key or "not-set", base_url=route.base_url)
            clients[key] = client
        return client

    def _call_route(self, params: dict, route: ProviderRoute):
        if route.provider == "litellm":
            return self._call_litellm_route_with_retry(params, route)
        active = self._active_route
        if not (
            active.provider == "openai"
            and active.api_key == route.api_key
            and active.base_url == route.base_url
        ):
            self.client = self._openai_client(route)
        return self._call_with_retry(params)

    def _call_litellm_route_with_retry(
        self,
        params: dict,
        route: ProviderRoute,
        max_retries: int = 3,
    ):
        import litellm

        params["drop_params"] = True
        if route.api_key:
            params["api_key"] = route.api_key
        if route.base_url:
            params["api_base"] = route.base_url
        for attempt in range(max_retries):
            try:
                return litellm.completion(**params)
            except Exception as error:
                if not self._is_litellm_transient(error) or attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                self._emit_event(
                    "llm.retry",
                    attempt=attempt + 1,
                    wait_seconds=wait,
                    error_type=type(error).__name__,
                    model=params.get("model"),
                    provider=route.provider,
                    route=route.route_id,
                )
                time.sleep(wait)

    @staticmethod
    def _is_litellm_transient(error: Exception) -> bool:
        text = str(error).lower()
        return any(
            marker in text for marker in (
                "rate_limit", "rate limit", "timeout", "connection",
                "internal server", "overloaded", "500", "502", "503", "504", "529",
            )
        )

    def _apply_budget_limit(self, params: dict, model: ProviderRoute | str):
        """Reserve input cost and clamp output tokens to the remaining USD."""
        if self.max_cost_usd is None:
            return
        if self._budget_usage_missing:
            raise BudgetExceededError("token usage is unavailable; refusing more spend")

        route = model if isinstance(model, ProviderRoute) else None
        model_name = route.model if route is not None else model
        pricing = route.pricing() if route is not None else _pricing_for_model(model_name)
        if pricing is None:
            raise BudgetExceededError(
                f"no pricing is configured for model {model_name!r}; refusing an unmetered request"
            )
        spent = self.estimated_cost
        if spent is None:
            raise BudgetExceededError("existing model usage cannot be priced safely")
        remaining = self.max_cost_usd - spent
        if remaining <= 0:
            raise BudgetExceededError(
                f"USD budget exhausted (${spent:.6f} spent / "
                f"${self.max_cost_usd:.6f} limit)"
            )

        input_rate, output_rate = pricing
        input_tokens = _request_token_upper_bound(
            params["messages"], params.get("tools")
        )
        input_reserve = input_tokens * input_rate / 1_000_000
        affordable_output = int(
            max(0.0, remaining - input_reserve) * 1_000_000 / output_rate
        )
        if affordable_output < 1:
            raise BudgetExceededError(
                f"USD budget has ${remaining:.6f} left, below the conservative "
                f"${input_reserve:.6f} input reserve for the next request"
            )

        key = "max_completion_tokens" if "max_completion_tokens" in params else "max_tokens"
        requested = params.get(key)
        params[key] = min(int(requested), affordable_output) if requested else affordable_output

    @staticmethod
    def _is_fallback_error(exc: Exception) -> bool:
        if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
            return True
        return isinstance(exc, APIError) and (getattr(exc, "status_code", 0) or 0) >= 500

    def _call_with_retry(self, params: dict, max_retries: int = 3):
        """Retry on transient errors with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(**params)
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                self._emit_event(
                    "llm.retry",
                    attempt=attempt + 1,
                    wait_seconds=wait,
                    error_type=type(e).__name__,
                    model=params.get("model"),
                )
                time.sleep(wait)
            except APIError as e:
                # retry 5xx server errors but not 4xx; base APIError has no status_code so read it defensively
                status_code = getattr(e, "status_code", None)
                if status_code and status_code >= 500 and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    self._emit_event(
                        "llm.retry",
                        attempt=attempt + 1,
                        wait_seconds=wait,
                        error_type=type(e).__name__,
                        status_code=status_code,
                        model=params.get("model"),
                    )
                    time.sleep(wait)
                else:
                    raise


class LiteLLM(LLM):
    """LLM backend via LiteLLM, supporting 100+ providers.

    Use this when your target provider is NOT OpenAI-compatible
    (AWS Bedrock, Google Vertex, Cohere, etc.) or when you want
    a single interface to switch between any provider by changing
    the model string.

    Set CORECODER_PROVIDER=litellm and use LiteLLM model strings
    like ``anthropic/claude-3-haiku``, ``bedrock/anthropic.claude-v2``,
    ``vertex_ai/gemini-pro``, etc.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        fallback_models: list[str] | tuple[str, ...] | None = None,
        fallback_routes: list[ProviderRoute | dict] | tuple[ProviderRoute | dict, ...] | None = None,
        max_cost_usd: float | None = None,
        **kwargs,
    ):
        # skip LLM.__init__ which creates an OpenAI client
        self._init_policy(
            model,
            fallback_models,
            max_cost_usd,
            provider="litellm",
            api_key=api_key,
            base_url=base_url,
            fallback_routes=fallback_routes,
        )
        self.api_key = api_key
        self.base_url = base_url
        self.extra = kwargs

    def _call_route(self, params: dict, route: ProviderRoute):
        if route.provider == "openai":
            self.client = self._openai_client(route)
            return LLM._call_with_retry(self, params)
        self.api_key = route.api_key
        self.base_url = route.base_url
        return self._call_with_retry(params)

    @staticmethod
    def _is_fallback_error(exc: Exception) -> bool:
        if LLM._is_fallback_error(exc):
            return True
        return LLM._is_litellm_transient(exc)

    def _call_with_retry(self, params: dict, max_retries: int = 3):
        """Retry on transient errors with exponential backoff via litellm."""
        import litellm

        params["drop_params"] = True
        if self.api_key:
            params["api_key"] = self.api_key
        if self.base_url:
            params["api_base"] = self.base_url

        for attempt in range(max_retries):
            try:
                return litellm.completion(**params)
            except Exception as e:
                err = str(e).lower()
                is_transient = any(
                    kw in err
                    for kw in ["rate_limit", "timeout", "connection", "502", "503", "529"]
                )
                is_server = any(kw in err for kw in ["500", "502", "503", "504"])
                if (is_transient or is_server) and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    self._emit_event(
                        "llm.retry",
                        attempt=attempt + 1,
                        wait_seconds=wait,
                        error_type=type(e).__name__,
                        model=params.get("model"),
                    )
                    time.sleep(wait)
                else:
                    raise
