"""A tool that tells the agent the current time."""

import time
from typing import ClassVar

from .base import Tool, ToolEffect


class NowTool(Tool):
    name = "now"
    effect = ToolEffect.PURE
    description = "Get the current local date and time. Use this when the user asks about the current time or you need a timestamp."
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")
