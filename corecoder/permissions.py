"""User consent for tool calls, distilled from Claude Code's permissions.

The tools split in two. Read-only ones (read_file, glob, grep, todo_write,
now, agent_status, load_skill)
run the moment the model asks; the mutating ones (edit_file, write_file,
bash, and spawning a sub-agent) stop for a yes first. "Always allow" is
remembered per tool for the rest of the session: per tool rather than per
command, because one approved bash prefix says nothing about the next
command anyway.

When there is nobody to ask (one-shot -p mode, or a library embedding with
no callback), a mutating call is refused instead of blocking on input that
can never arrive. The refusal travels back as an ordinary tool result, so
the loop survives and the model can route around it.
"""

import threading

from .decisions import ToolDecision


class Permission:
    """Session-scoped consent state. Pure: no I/O, the CLI hands in `ask`."""

    READ_ONLY = frozenset({
        "read_file", "glob", "grep", "todo_write", "now", "agent_status",
        "load_skill", "memory_update",
    })

    def __init__(self, ask=None, allow_all: bool = False):
        # ask(tool_name, arguments) -> "once" | "always" | "deny"
        self.ask = ask
        self.allow_all = allow_all
        self._always: set[str] = set()
        self._lock = threading.Lock()

    def is_preapproved(self, tool_name: str) -> bool:
        """Whether a call can run without asking a human right now."""
        with self._lock:
            return (
                tool_name in self.READ_ONLY
                or self.allow_all
                or tool_name in self._always
            )

    def for_background(self):
        """Return a live, non-interactive view of this permission state.

        A worker thread must never invoke the terminal's prompt callback. It
        may use ``--yes`` or an existing "always allow" decision, while every
        other stateful call fails closed as an ordinary tool result.
        """
        return _BackgroundPermission(self)

    def check(self, tool_name: str, arguments: dict) -> str | None:
        """Decide one call. None lets it through; a string is the refusal
        the model receives as its tool result."""
        return self.decide(tool_name, arguments).result

    def decide(self, tool_name: str, arguments: dict) -> ToolDecision:
        """Return a traceable decision label and the optional refusal text."""
        with self._lock:
            if tool_name in self.READ_ONLY:
                return ToolDecision.allow("permission", "read_only")
            if self.allow_all:
                return ToolDecision.allow("permission", "allow_all")
            if tool_name in self._always:
                return ToolDecision.allow("permission", "always_cached")
        if self.ask is None:
            return ToolDecision.deny(
                "permission",
                "non_interactive_deny",
                f"Permission denied: {tool_name} mutates state and this session is "
                "non-interactive, so nobody can approve it. Rerun with --yes, or "
                "tell the user the exact step so they can run it themselves.",
            )
        verdict = self.ask(tool_name, arguments)
        if verdict == "always":
            with self._lock:
                self._always.add(tool_name)
            return ToolDecision.allow("permission", "always")
        if verdict == "once":
            return ToolDecision.allow("permission", "once")
        return ToolDecision.deny(
            "permission",
            "deny",
            f"Permission denied: the user refused this {tool_name} call. "
            "Do not retry it unchanged; ask what they would prefer instead.",
        )


class _BackgroundPermission:
    """Permission adapter used by asynchronous sub-agents."""

    def __init__(self, parent: Permission):
        self.parent = parent

    def check(self, tool_name: str, arguments: dict) -> str | None:
        return self.decide(tool_name, arguments).result

    def decide(self, tool_name: str, arguments: dict) -> ToolDecision:
        if self.parent.is_preapproved(tool_name):
            return ToolDecision.allow("permission", "background_preapproved")
        return ToolDecision.deny(
            "permission",
            "background_deny",
            f"Permission denied: background sub-agent cannot prompt for {tool_name}. "
            "Pre-approve this tool with 'always allow' before launching, use --yes, "
            "or run the sub-agent in foreground mode.",
        )
