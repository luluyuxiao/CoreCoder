"""Structured task memory kept outside lossy conversation history.

This state is deliberately small and explicit. It is re-injected on every LLM
request and persisted with the session, while ordinary messages may be
summarized or collapsed. Constraints stored here are semantic instructions;
real enforcement still belongs to hooks, capabilities, permissions, and the
sandbox.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

_PLAN_STATUSES = frozenset({"pending", "in_progress", "completed", "blocked"})
_MAX_TEXT = 2000
_MAX_ITEMS = 100
_MAX_FILES = 200
_MAX_MEMORY_CHARS = 24_000


def _text(value, limit: int = _MAX_TEXT) -> str:
    return " ".join(str(value or "").split())[:limit].rstrip()


def _next_id(prefix: str, items: list[dict]) -> str:
    used = {str(item.get("id") or "") for item in items}
    number = len(items) + 1
    while f"{prefix}-{number}" in used:
        number += 1
    return f"{prefix}-{number}"


@dataclass
class MemoryState:
    """Goal, constraints, plan, decisions, and derived artifacts for one Agent."""

    goal: str = ""
    constraints: list[dict] = field(default_factory=list)
    plan: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)

    def _size(self) -> int:
        return (
            len(self.goal)
            + sum(len(str(value)) for item in self.constraints for value in item.values())
            + sum(len(str(value)) for item in self.plan for value in item.values())
            + sum(len(str(value)) for item in self.decisions for value in item.values())
            + sum(len(path) for path in self.files_modified)
        )

    def _check_size(self):
        if self._size() > _MAX_MEMORY_CHARS:
            raise ValueError(
                f"structured memory exceeds its {_MAX_MEMORY_CHARS}-character budget"
            )

    def set_goal(self, content: str, *, source: str = "user") -> str:
        goal = _text(content)
        if not goal:
            raise ValueError("goal must be non-empty")
        previous = self.goal
        self.goal = goal
        try:
            self._check_size()
        except ValueError:
            self.goal = previous
            raise
        return self.goal

    def clear_goal(self):
        self.goal = ""

    def add_constraint(self, content: str, *, source: str = "user") -> dict:
        text = _text(content)
        if not text:
            raise ValueError("constraint must be non-empty")
        if len(self.constraints) >= _MAX_ITEMS:
            raise ValueError("constraint limit reached")
        item = {
            "id": _next_id("constraint", self.constraints),
            "content": text,
            "source": _text(source, 40) or "user",
            "status": "active",
            "enforcement": "semantic",
        }
        self.constraints.append(item)
        try:
            self._check_size()
        except ValueError:
            self.constraints.pop()
            raise
        return copy.deepcopy(item)

    def revoke_constraint(self, constraint_id: str) -> bool:
        for item in self.constraints:
            if item.get("id") == constraint_id and item.get("status") == "active":
                item["status"] = "revoked"
                return True
        return False

    def add_plan_step(self, content: str, *, source: str = "agent") -> dict:
        text = _text(content)
        if not text:
            raise ValueError("plan step must be non-empty")
        if len(self.plan) >= _MAX_ITEMS:
            raise ValueError("plan step limit reached")
        item = {
            "id": _next_id("step", self.plan),
            "content": text,
            "status": "pending",
            "source": _text(source, 40) or "agent",
        }
        self.plan.append(item)
        try:
            self._check_size()
        except ValueError:
            self.plan.pop()
            raise
        return copy.deepcopy(item)

    def update_plan_step(
        self,
        step_id: str,
        *,
        status: str | None = None,
        content: str | None = None,
    ) -> dict | None:
        if status is not None and status not in _PLAN_STATUSES:
            raise ValueError("invalid plan status")
        for item in self.plan:
            if item.get("id") != step_id:
                continue
            previous = copy.deepcopy(item)
            if status is not None:
                item["status"] = status
            if content is not None:
                text = _text(content)
                if not text:
                    item.clear()
                    item.update(previous)
                    raise ValueError("plan step content must be non-empty")
                item["content"] = text
            try:
                self._check_size()
            except ValueError:
                item.clear()
                item.update(previous)
                raise
            return copy.deepcopy(item)
        return None

    def clear_plan(self):
        self.plan.clear()

    def add_decision(
        self,
        content: str,
        *,
        reason: str = "",
        source: str = "agent",
    ) -> dict:
        text = _text(content)
        if not text:
            raise ValueError("decision must be non-empty")
        if len(self.decisions) >= _MAX_ITEMS:
            raise ValueError("decision limit reached")
        item = {
            "id": _next_id("decision", self.decisions),
            "content": text,
            "reason": _text(reason),
            "source": _text(source, 40) or "agent",
        }
        self.decisions.append(item)
        try:
            self._check_size()
        except ValueError:
            self.decisions.pop()
            raise
        return copy.deepcopy(item)

    def record_file(self, path: str):
        normalized = str(path).strip()[:1000]
        if (
            normalized
            and normalized not in self.files_modified
            and len(self.files_modified) < _MAX_FILES
            and self._size() + len(normalized) <= _MAX_MEMORY_CHARS
        ):
            self.files_modified.append(normalized)

    def clear(self):
        self.goal = ""
        self.constraints.clear()
        self.plan.clear()
        self.decisions.clear()
        self.files_modified.clear()

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "constraints": copy.deepcopy(self.constraints),
            "plan": copy.deepcopy(self.plan),
            "decisions": copy.deepcopy(self.decisions),
            "files_modified": list(self.files_modified),
        }

    @classmethod
    def from_dict(cls, data) -> MemoryState:
        state = cls()
        if not isinstance(data, dict):
            return state
        goal = _text(data.get("goal"))
        if goal:
            state.goal = goal
        for raw in list(data.get("constraints") or [])[:_MAX_ITEMS]:
            if not isinstance(raw, dict) or not _text(raw.get("content")):
                continue
            state.constraints.append({
                "id": _text(raw.get("id"), 80)
                or _next_id("constraint", state.constraints),
                "content": _text(raw.get("content")),
                "source": _text(raw.get("source"), 40) or "user",
                "status": (
                    raw.get("status") if raw.get("status") in {"active", "revoked"}
                    else "active"
                ),
                "enforcement": "semantic",
            })
            if state._size() > _MAX_MEMORY_CHARS:
                state.constraints.pop()
                break
        for raw in list(data.get("plan") or [])[:_MAX_ITEMS]:
            if not isinstance(raw, dict) or not _text(raw.get("content")):
                continue
            state.plan.append({
                "id": _text(raw.get("id"), 80) or _next_id("step", state.plan),
                "content": _text(raw.get("content")),
                "status": (
                    raw.get("status") if raw.get("status") in _PLAN_STATUSES
                    else "pending"
                ),
                "source": _text(raw.get("source"), 40) or "agent",
            })
            if state._size() > _MAX_MEMORY_CHARS:
                state.plan.pop()
                break
        for raw in list(data.get("decisions") or [])[:_MAX_ITEMS]:
            if not isinstance(raw, dict) or not _text(raw.get("content")):
                continue
            state.decisions.append({
                "id": _text(raw.get("id"), 80)
                or _next_id("decision", state.decisions),
                "content": _text(raw.get("content")),
                "reason": _text(raw.get("reason")),
                "source": _text(raw.get("source"), 40) or "agent",
            })
            if state._size() > _MAX_MEMORY_CHARS:
                state.decisions.pop()
                break
        for path in list(data.get("files_modified") or [])[:_MAX_FILES]:
            state.record_file(str(path))
        return state

    def render(self) -> str:
        sections = []
        if self.goal:
            sections.append("## Goal\n" + self.goal)
        active = [item for item in self.constraints if item.get("status") == "active"]
        if active:
            lines = [
                f"- [{item['id']}] {item['content']}"
                for item in active
            ]
            sections.append(
                "## Critical constraints\n"
                "These are persistent semantic instructions. They do not replace "
                "Permission, Hooks, Capability Policy, or Sandbox enforcement.\n"
                + "\n".join(lines)
            )
        if self.plan:
            sections.append("## Current plan\n" + "\n".join(
                f"- [{item['status']}] {item['id']}: {item['content']}"
                for item in self.plan
            ))
        if self.decisions:
            sections.append("## Decisions\n" + "\n".join(
                f"- {item['content']}"
                + (f" — {item['reason']}" if item.get("reason") else "")
                for item in self.decisions
            ))
        if self.files_modified:
            sections.append("## Files modified\n" + "\n".join(
                f"- {path}" for path in self.files_modified
            ))
        if not sections:
            return ""
        return "# Structured task memory\n\n" + "\n\n".join(sections)
