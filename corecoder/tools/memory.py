"""Tool for updating structured task memory without rewriting chat history."""

from typing import ClassVar

from ..memory import MemoryState
from .base import Tool, ToolEffect


class MemoryUpdateTool(Tool):
    name = "memory_update"
    effect = ToolEffect.WRITE
    capabilities = frozenset()
    description = (
        "Update structured task memory that persists across context compression and resume. "
        "Use it for the goal, plan steps, and durable decisions. It cannot create critical "
        "constraints; only the user can do that through the CLI."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "set_goal", "clear_goal", "add_plan_step",
                    "update_plan_step", "clear_plan", "add_decision",
                ],
            },
            "content": {"type": "string"},
            "item_id": {"type": "string"},
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed", "blocked"],
            },
            "reason": {"type": "string"},
        },
        "required": ["operation"],
    }

    def __init__(self):
        self._state: MemoryState | None = None

    def bind(self, state: MemoryState):
        self._state = state

    def execute(
        self,
        operation: str,
        content: str = "",
        item_id: str = "",
        status: str = "pending",
        reason: str = "",
    ) -> str:
        if self._state is None:
            return "Error: memory_update tool is not initialized"
        try:
            if operation == "set_goal":
                return f"Goal updated: {self._state.set_goal(content, source='agent')}"
            if operation == "clear_goal":
                self._state.clear_goal()
                return "Goal cleared."
            if operation == "add_plan_step":
                item = self._state.add_plan_step(content)
                return f"Plan step added: {item['id']}"
            if operation == "update_plan_step":
                item = self._state.update_plan_step(
                    item_id,
                    status=status,
                    content=content or None,
                )
                return (
                    f"Plan step updated: {item['id']} [{item['status']}]"
                    if item else f"Error: plan step {item_id!r} not found"
                )
            if operation == "clear_plan":
                self._state.clear_plan()
                return "Plan cleared."
            if operation == "add_decision":
                item = self._state.add_decision(content, reason=reason)
                return f"Decision recorded: {item['id']}"
            return f"Error: unknown memory operation {operation!r}"
        except ValueError as error:
            return f"Error: {error}"

