"""Per-tool capability policy, independent from consent and sandboxing.

Permission answers "did the user authorize this call?".  A capability policy
answers "may this Tool exercise this kind of authority at all?".  Policies are
deny-by-capability rather than deny-by-command, so they remain useful for MCP
and custom tools whose implementation is outside the Agent process.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path

FILESYSTEM_READ = "filesystem_read"
FILESYSTEM_WRITE = "filesystem_write"
NETWORK = "network"
PROCESS = "process"
SUBAGENT = "subagent"
MCP = "mcp"
UNKNOWN = "unknown"

KNOWN_CAPABILITIES = frozenset({
    FILESYSTEM_READ,
    FILESYSTEM_WRITE,
    NETWORK,
    PROCESS,
    SUBAGENT,
    MCP,
    UNKNOWN,
})
DEFAULT_POLICY_FILE = Path.home() / ".corecoder" / "capabilities.json"


@dataclass(frozen=True)
class CapabilityRule:
    pattern: str
    allowed: frozenset[str] | None  # None means all capabilities


class CapabilityPolicy:
    """Match Tool names to the capabilities they are allowed to exercise."""

    def __init__(
        self,
        *,
        default: str = "allow",
        rules: list[CapabilityRule] | None = None,
        source: str | Path | None = None,
    ):
        if default not in {"allow", "deny"}:
            raise ValueError("capability policy default must be 'allow' or 'deny'")
        self.default = default
        self.rules = list(rules or [])
        self.source = Path(source).expanduser().resolve() if source else None

    @property
    def enabled(self) -> bool:
        return bool(self.rules) or self.default == "deny"

    @classmethod
    def from_file(cls, path: str | Path) -> CapabilityPolicy:
        source = Path(path).expanduser().resolve()
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("capability policy must be a JSON object")
        default = data.get("default", "allow")
        raw_tools = data.get("tools", {})
        if not isinstance(raw_tools, dict):
            raise TypeError("capability policy 'tools' must be an object")
        rules = []
        for pattern, spec in raw_tools.items():
            if not isinstance(pattern, str) or not pattern:
                raise ValueError("capability policy tool patterns must be non-empty strings")
            if not isinstance(spec, dict):
                raise TypeError(f"capability rule {pattern!r} must be an object")
            raw_allowed = spec.get("allow", [])
            if not isinstance(raw_allowed, list) or not all(
                isinstance(item, str) for item in raw_allowed
            ):
                raise TypeError(f"capability rule {pattern!r} allow must be a string list")
            unknown = set(raw_allowed) - KNOWN_CAPABILITIES - {"*"}
            if unknown:
                raise ValueError(
                    f"capability rule {pattern!r} contains unknown values: "
                    + ", ".join(sorted(unknown))
                )
            allowed = None if "*" in raw_allowed else frozenset(raw_allowed)
            rules.append(CapabilityRule(pattern=pattern, allowed=allowed))
        return cls(default=default, rules=rules, source=source)

    def _allowed_for(self, tool_name: str) -> frozenset[str] | None:
        exact = next((rule for rule in self.rules if rule.pattern == tool_name), None)
        if exact is not None:
            return exact.allowed
        matches = [
            rule for rule in self.rules
            if rule.pattern != tool_name and fnmatch.fnmatchcase(tool_name, rule.pattern)
        ]
        if matches:
            # Prefer the most specific matching pattern, independent of JSON order.
            return max(matches, key=lambda rule: len(rule.pattern)).allowed
        return None if self.default == "allow" else frozenset()

    def decide(self, tool, arguments: dict) -> tuple[str, str | None]:
        required = frozenset(tool.required_capabilities(arguments))
        invalid = required - KNOWN_CAPABILITIES
        if invalid:
            required = required | {UNKNOWN}
        allowed = self._allowed_for(tool.name)
        if allowed is None:
            return "capability_allow_all", None
        denied = required - allowed
        if not denied:
            return "capability_allow", None
        return (
            "capability_deny",
            (
                f"Capability denied: tool {tool.name!r} requires "
                f"{', '.join(sorted(denied))}, which is not allowed by the active "
                "per-tool capability policy. Do not retry unchanged."
            ),
        )

    def as_dict(self) -> dict:
        return {
            "default": self.default,
            "source": str(self.source) if self.source else None,
            "rules": [
                {
                    "pattern": rule.pattern,
                    "allow": ["*"] if rule.allowed is None else sorted(rule.allowed),
                }
                for rule in self.rules
            ],
        }


def load_capability_policy(path: str | Path | None = None) -> CapabilityPolicy:
    """Load an explicit policy, or use the default file when it exists."""
    source = Path(path).expanduser() if path is not None else DEFAULT_POLICY_FILE
    if not source.exists():
        if path is not None:
            raise FileNotFoundError(f"capability policy not found: {source}")
        return CapabilityPolicy()
    return CapabilityPolicy.from_file(source)


__all__ = [
    "DEFAULT_POLICY_FILE",
    "FILESYSTEM_READ",
    "FILESYSTEM_WRITE",
    "KNOWN_CAPABILITIES",
    "MCP",
    "NETWORK",
    "PROCESS",
    "SUBAGENT",
    "UNKNOWN",
    "CapabilityPolicy",
    "CapabilityRule",
    "load_capability_policy",
]
