"""Structured decisions shared by every tool-execution gate.

The original implementation used a mixture of ``None``, strings and
``(label, result)`` tuples. ``ToolDecision`` keeps one explicit contract while
remaining tuple-compatible for embedders that used the old ``decide()`` APIs.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolDecision:
    """One auditable allow/deny result from a named policy gate."""

    gate: str
    code: str
    allowed: bool
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def result(self) -> str | None:
        """Legacy tool-result representation: ``None`` allows, text denies."""
        return None if self.allowed else self.reason

    @classmethod
    def allow(cls, gate: str, code: str, **metadata: Any) -> ToolDecision:
        return cls(gate=gate, code=code, allowed=True, metadata=metadata)

    @classmethod
    def deny(
        cls,
        gate: str,
        code: str,
        reason: str,
        **metadata: Any,
    ) -> ToolDecision:
        return cls(
            gate=gate,
            code=code,
            allowed=False,
            reason=reason,
            metadata=metadata,
        )

    def __iter__(self) -> Iterator[str | None]:
        """Allow ``label, result = decision`` during the compatibility window."""
        yield self.code
        yield self.result

    def __getitem__(self, index: int) -> str | None:
        return (self.code, self.result)[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, tuple) and len(other) == 2:
            return (self.code, self.result) == other
        if not isinstance(other, ToolDecision):
            return False
        return (
            self.gate,
            self.code,
            self.allowed,
            self.reason,
            self.metadata,
        ) == (
            other.gate,
            other.code,
            other.allowed,
            other.reason,
            other.metadata,
        )


def coerce_decision(
    value: ToolDecision | tuple[str, str | None] | str | None,
    *,
    gate: str,
    allow_code: str,
    deny_code: str,
) -> ToolDecision:
    """Adapt legacy extension return values at the Agent boundary."""
    if isinstance(value, ToolDecision):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        code, result = value
        if result is None:
            return ToolDecision.allow(gate, str(code))
        return ToolDecision.deny(gate, str(code), str(result))
    if value is None:
        return ToolDecision.allow(gate, allow_code)
    return ToolDecision.deny(gate, deny_code, str(value))


__all__ = ["ToolDecision", "coerce_decision"]
