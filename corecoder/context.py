"""Budget-aware, multi-layer context compression.

The model limit covers more than conversation history: every request also
contains the system prompt, tool schemas, protocol framing, and space reserved
for the answer. CoreCoder computes a message budget from the whole request,
then applies increasingly lossy tool snipping, summarization, and collapse.

A final deterministic ``budget_fit`` is a correctness boundary. Fresh tool
results are protected from ordinary history snipping until one successful LLM
request consumes them; if a fresh batch alone cannot fit, budget_fit retains a
clearly marked bounded observation instead of sending an impossible request.
"""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLM

_CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿　-〿＀-･]")
_SYMBOL_RE = re.compile(r"[^\w\s]")
_MESSAGE_OVERHEAD_TOKENS = 4
_REQUEST_OVERHEAD_TOKENS = 16
_TOOL_SCHEMA_OVERHEAD_TOKENS = 8


class ContextOverflowError(RuntimeError):
    """The fixed request overhead leaves no usable room for messages."""


def _approx_tokens(text: str) -> int:
    """Estimate CJK, symbol-dense code, and ordinary prose conservatively."""
    cjk = len(_CJK_RE.findall(text))
    rest = len(text) - cjk
    dense = rest > 0 and len(_SYMBOL_RE.findall(text)) / len(text) > 0.25
    return math.ceil(cjk / 1.5) + int(rest / (2.8 if dense else 3.4))


def _json_text(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def estimate_tokens(messages: list[dict]) -> int:
    """Estimate message tokens, including per-message protocol framing."""
    total = 0
    for message in messages:
        total += _MESSAGE_OVERHEAD_TOKENS
        content = message.get("content")
        if content:
            total += _approx_tokens(content if isinstance(content, str) else _json_text(content))
        if message.get("tool_calls"):
            total += _approx_tokens(_json_text(message["tool_calls"]))
        if message.get("tool_call_id"):
            total += _approx_tokens(str(message["tool_call_id"]))
    return total


def estimate_tool_schema_tokens(tools: list[dict] | None) -> int:
    """Estimate the schemas sent on every tool-capable request."""
    if not tools:
        return 0
    return _approx_tokens(_json_text(tools)) + _TOOL_SCHEMA_OVERHEAD_TOKENS * len(tools)


def estimate_request_tokens(messages: list[dict], tools: list[dict] | None = None) -> int:
    """Estimate a complete model input, not just conversation messages."""
    return estimate_tokens(messages) + estimate_tool_schema_tokens(tools) + _REQUEST_OVERHEAD_TOKENS


def _truncate_text(text: str, max_chars: int, label: str) -> str:
    """Keep useful head/tail text within one hard character bound."""
    if len(text) <= max_chars:
        return text
    marker = f"\n... ({len(text)} chars, {label}) ...\n"
    if max_chars <= len(marker) + 16:
        return marker[:max_chars]
    remaining = max_chars - len(marker)
    head = max(8, remaining * 2 // 3)
    tail = max(8, remaining - head)
    return text[:head] + marker + text[-tail:]


class ContextManager:
    def __init__(self, max_tokens: int = 128_000):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        self.max_tokens = max_tokens
        self.last_actions: tuple[str, ...] = ()
        self.last_budget: dict[str, int] = {}

    @property
    def safety_margin_tokens(self) -> int:
        """Estimator slack: small for tests, proportional for real windows."""
        return max(64, int(self.max_tokens * 0.05))

    def available_message_tokens(self, fixed_tokens: int = 0, output_reserve: int = 0) -> int:
        """Tokens left for mutable conversation history in one full request."""
        available = (
            self.max_tokens
            - max(0, fixed_tokens)
            - max(0, output_reserve)
            - self.safety_margin_tokens
        )
        if available < _MESSAGE_OVERHEAD_TOKENS:
            raise ContextOverflowError(
                "system prompt, tool schemas, output reserve, and safety margin "
                f"consume the configured {self.max_tokens}-token context window"
            )
        return available

    def maybe_compress(
        self,
        messages: list[dict],
        llm: LLM | None = None,
        *,
        fixed_tokens: int = 0,
        output_reserve: int = 0,
        protected_tool_call_ids: set[str] | None = None,
    ) -> bool:
        """Compress against the real request budget and guarantee it fits."""
        protected = set(protected_tool_call_ids or ())
        budget = self.available_message_tokens(fixed_tokens, output_reserve)
        current = estimate_tokens(messages)
        compressed = False
        actions: list[str] = []

        snip_at = int(budget * 0.50)
        summarize_at = int(budget * 0.70)
        collapse_at = int(budget * 0.90)

        if current > snip_at and self._snip_tool_outputs(
            messages, protected_tool_call_ids=protected
        ):
            compressed = True
            actions.append("tool_snip")
            current = estimate_tokens(messages)

        if (
            current > summarize_at
            and len(messages) > 10
            and self._summarize_old(messages, llm, keep_recent=8)
        ):
            compressed = True
            actions.append("summarize")
            current = estimate_tokens(messages)

        if current > collapse_at and len(messages) > 4 and self._hard_collapse(messages, llm):
            compressed = True
            actions.append("hard_collapse")
            current = estimate_tokens(messages)

        if current > budget and self._fit_to_budget(messages, budget, protected):
            compressed = True
            actions.append("budget_fit")
            current = estimate_tokens(messages)

        if current > budget:
            raise ContextOverflowError(
                f"context compression could not fit {current} message tokens "
                f"into the {budget}-token message budget"
            )

        self.last_actions = tuple(actions)
        self.last_budget = {
            "max_tokens": self.max_tokens,
            "fixed_tokens": max(0, fixed_tokens),
            "output_reserve": max(0, output_reserve),
            "safety_margin": self.safety_margin_tokens,
            "message_budget": budget,
            "message_tokens": current,
            "protected_tool_results": len(protected),
        }
        return compressed

    @staticmethod
    def _snip_tool_outputs(
        messages: list[dict],
        protected_tool_call_ids: set[str] | None = None,
        *,
        max_chars: int = 1500,
        include_protected: bool = False,
    ) -> bool:
        """Trim large tool results by characters, including giant one-line data."""
        protected = protected_tool_call_ids or set()
        changed = False
        for message in messages:
            if message.get("role") != "tool":
                continue
            if not include_protected and message.get("tool_call_id") in protected:
                continue
            content = message.get("content", "")
            if not isinstance(content, str) or len(content) <= max_chars:
                continue
            lines = content.count("\n") + 1
            message["content"] = _truncate_text(
                content,
                max_chars,
                f"{lines} lines, snipped to save context",
            )
            changed = True
        return changed

    @staticmethod
    def _safe_split(messages: list[dict], keep_recent: int) -> int:
        """Return a tail boundary that never orphans tool result messages."""
        split = max(0, len(messages) - keep_recent)
        while split > 0 and messages[split].get("role") == "tool":
            split -= 1
        return split

    def _summarize_old(
        self,
        messages: list[dict],
        llm: LLM | None,
        keep_recent: int = 8,
    ) -> bool:
        """Layer 2: summarize old conversation, keeping recent messages intact."""
        if len(messages) <= keep_recent:
            return False
        split = self._safe_split(messages, keep_recent)
        if split <= 0:
            return False
        old = messages[:split]
        tail = messages[split:]
        summary = self._get_summary(old, llm)

        messages.clear()
        messages.append({
            "role": "user",
            "content": f"[Context compressed - conversation summary]\n{summary}",
        })
        messages.append({
            "role": "assistant",
            "content": "Got it, I have the context from our earlier conversation.",
        })
        messages.extend(tail)
        return True

    def _hard_collapse(self, messages: list[dict], llm: LLM | None) -> bool:
        """Layer 3: keep a semantic summary and one complete recent group."""
        split = self._safe_split(messages, 4 if len(messages) > 4 else 2)
        if split <= 0:
            return False
        tail = messages[split:]
        summary = self._get_summary(messages[:split], llm)

        messages.clear()
        messages.append({
            "role": "user",
            "content": f"[Hard context reset]\n{summary}",
        })
        messages.append({
            "role": "assistant",
            "content": "Context restored. Continuing from where we left off.",
        })
        messages.extend(tail)
        return True

    def _fit_to_budget(
        self,
        messages: list[dict],
        budget: int,
        protected_tool_call_ids: set[str],
    ) -> bool:
        """Deterministically force history under budget without invalid tool pairs."""
        changed = self._snip_tool_outputs(
            messages,
            protected_tool_call_ids=protected_tool_call_ids,
            max_chars=500,
        )
        if estimate_tokens(messages) <= budget:
            return changed

        if self._compact_tool_arguments(messages, max_chars=600):
            changed = True
        if estimate_tokens(messages) <= budget:
            return changed

        if self._snip_tool_outputs(
            messages,
            protected_tool_call_ids=protected_tool_call_ids,
            max_chars=700,
            include_protected=True,
        ):
            changed = True
        if estimate_tokens(messages) <= budget:
            return changed

        self._emergency_reset(messages, budget, protected_tool_call_ids)
        return True

    @staticmethod
    def _compact_tool_arguments(messages: list[dict], max_chars: int) -> bool:
        """Keep tool ids/names and short scalar args while bounding large payloads."""
        changed = False
        for message in messages:
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                raw = function.get("arguments")
                if not isinstance(raw, str) or len(raw) <= max_chars:
                    continue
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    parsed = None
                if isinstance(parsed, dict):
                    compact = {}
                    for key, value in parsed.items():
                        rendered = _json_text(value)
                        compact[key] = (
                            value
                            if len(rendered) <= 160
                            else f"<context-truncated {len(rendered)} chars>"
                        )
                    compact["_context_note"] = "large tool arguments were compacted"
                    replacement = _json_text(compact)
                else:
                    replacement = _json_text({
                        "_context_note": f"tool arguments truncated from {len(raw)} chars"
                    })
                function["arguments"] = _truncate_text(
                    replacement, max_chars, "tool arguments compacted"
                )
                changed = True
        return changed

    def _emergency_reset(
        self,
        messages: list[dict],
        budget: int,
        protected_tool_call_ids: set[str],
    ):
        """Replace irreducibly large history with one bounded, valid state note."""
        latest_user = next((
            str(message.get("content") or "")
            for message in reversed(messages)
            if message.get("role") == "user"
        ), "")
        observations = []
        for message in messages:
            if (
                message.get("role") == "tool"
                and message.get("tool_call_id") in protected_tool_call_ids
            ):
                observations.append(
                    f"- {message.get('tool_call_id')}: "
                    + _truncate_text(str(message.get("content") or ""), 350, "observation bounded")
                )
        state = self._extract_key_info(messages)
        text = "\n".join(filter(None, [
            "[Emergency context reset to fit the model window]",
            f"Latest user request: {_truncate_text(latest_user, 800, 'request bounded')}",
            f"Recovered state: {state}",
            "Latest tool observations:\n" + "\n".join(observations) if observations else "",
        ]))
        messages[:] = [{"role": "user", "content": text}]

        while estimate_tokens(messages) > budget and len(messages[0]["content"]) > 32:
            current = messages[0]["content"]
            messages[0]["content"] = _truncate_text(
                current,
                max(32, int(len(current) * 0.70)),
                "emergency state bounded",
            )
        if estimate_tokens(messages) > budget:
            messages[0]["content"] = "[Context reset]"

    def _get_summary(self, messages: list[dict], llm: LLM | None) -> str:
        """Generate summary via LLM or fall back to deterministic extraction."""
        flat = self._flatten(messages)

        if llm:
            try:
                resp = llm.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Compress this conversation into a brief summary. "
                                "Preserve: file paths edited, key decisions made, "
                                "errors encountered, current task state. "
                                "Drop: verbose command output, code listings, "
                                "redundant back-and-forth."
                            ),
                        },
                        {"role": "user", "content": flat[:15000]},
                    ],
                )
                return resp.content
            except Exception:  # noqa: BLE001, S110
                pass

        return self._extract_key_info(messages)

    @staticmethod
    def _flatten(messages: list[dict]) -> str:
        parts = []
        for message in messages:
            role = message.get("role", "?")
            text = message.get("content", "") or ""
            if text:
                parts.append(f"[{role}] {str(text)[:400]}")
        return "\n".join(parts)

    @staticmethod
    def _extract_key_info(messages: list[dict]) -> str:
        """Fallback: extract file paths and errors without another model call."""
        files_seen = set()
        errors = []

        for message in messages:
            text = str(message.get("content", "") or "")
            for match in re.finditer(r'[\w./\-]+\.\w{1,5}', text):
                files_seen.add(match.group())
            for line in text.splitlines():
                if "error" in line.lower():
                    errors.append(line.strip()[:150])

        parts = []
        if files_seen:
            parts.append(f"Files touched: {', '.join(sorted(files_seen)[:20])}")
        if errors:
            parts.append(f"Errors seen: {'; '.join(errors[:5])}")
        return "\n".join(parts) or "(no extractable context)"
