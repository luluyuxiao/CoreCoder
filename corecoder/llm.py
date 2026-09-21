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
import threading
import time
from dataclasses import dataclass, field
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
        max_cost_usd: float | None = None,
        **kwargs,
    ):
        self._init_policy(model, fallback_models, max_cost_usd)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.extra = kwargs  # temperature, max_tokens, etc.

    def _init_policy(
        self,
        model: str,
        fallback_models: list[str] | tuple[str, ...] | None,
        max_cost_usd: float | None,
    ):
        if max_cost_usd is not None and max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be greater than zero")
        self.model = model
        self.fallback_models = [
            candidate
            for candidate in dict.fromkeys(fallback_models or [])
            if candidate and candidate != model
        ]
        self.max_cost_usd = max_cost_usd
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.usage_by_model: dict[str, dict[str, int]] = {}
        self.fallback_history: list[tuple[str, str, str]] = []
        self._budget_usage_missing = False
        self._event_local = threading.local()

    @property
    def estimated_cost(self) -> float | None:
        """Rough cumulative USD cost, including every model used by fallback."""
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
        candidates = list(dict.fromkeys([self.model, *self.fallback_models]))
        starting_model = self.model
        last_fallback_error: Exception | None = None
        response_model = self.model
        for position, candidate in enumerate(candidates):
            params = {**base_params, "model": candidate}
            params["stream_options"] = {"include_usage": True}
            self._apply_budget_limit(params, candidate)
            try:
                while True:
                    try:
                        stream = self._call_with_retry(params)
                        break
                    except BadRequestError as e:
                        if _adapt_rejected_param(params, e):
                            self._emit_event(
                                "llm.provider_adapted",
                                model=candidate,
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
                if not has_fallback or not self._is_fallback_error(e):
                    raise
                next_model = candidates[position + 1]
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
                    error_type=type(e).__name__,
                    status_code=getattr(e, "status_code", None),
                )
                continue

            response_model = candidate
            if candidate != starting_model:
                self.model = candidate  # sticky: do not retry a dead primary every round
                self.fallback_history.append(
                    (starting_model, candidate, str(last_fallback_error or "unavailable"))
                )
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

    def _apply_budget_limit(self, params: dict, model: str):
        """Reserve input cost and clamp output tokens to the remaining USD."""
        if self.max_cost_usd is None:
            return
        if self._budget_usage_missing:
            raise BudgetExceededError("token usage is unavailable; refusing more spend")

        pricing = _pricing_for_model(model)
        if pricing is None:
            raise BudgetExceededError(
                f"no pricing is configured for model {model!r}; refusing an unmetered request"
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
        max_cost_usd: float | None = None,
        **kwargs,
    ):
        # skip LLM.__init__ which creates an OpenAI client
        self._init_policy(model, fallback_models, max_cost_usd)
        self.api_key = api_key
        self.base_url = base_url
        self.extra = kwargs

    @staticmethod
    def _is_fallback_error(exc: Exception) -> bool:
        if LLM._is_fallback_error(exc):
            return True
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "rate_limit",
                "rate limit",
                "timeout",
                "connection",
                "internal server",
                "overloaded",
                "500",
                "502",
                "503",
                "504",
                "529",
            )
        )

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
