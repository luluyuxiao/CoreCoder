"""Tool adapter for lazy Skill instruction loading."""

from datetime import datetime, timezone
from typing import ClassVar

from ..capabilities import FILESYSTEM_READ
from ..skills import SkillRegistry
from .base import Tool, ToolEffect


class LoadSkillTool(Tool):
    name = "load_skill"
    effect = ToolEffect.READ
    capabilities = frozenset({FILESYSTEM_READ})
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Exact skill name from the available catalog",
            },
        },
        "required": ["name"],
    }

    def __init__(self, registry: SkillRegistry):
        self.registry = registry
        catalog = registry.catalog()
        self.description = (
            "Load the complete instructions for one reusable workflow only when "
            "the current task matches it. Skills guide how to work but never grant "
            "extra tools or bypass permissions. Available skills:\n" + catalog
        )
        self.parameters = {
            **self.parameters,
            "properties": {
                **self.parameters["properties"],
                "name": {
                    **self.parameters["properties"]["name"],
                    "enum": registry.names(),
                },
            },
        }

    def execute(self, name: str) -> str:
        return self.registry.load(name)

    def activation_record(self, name: str, result: str | None = None) -> dict | None:
        """Describe a successful load without storing the full instructions.

        ``result`` is checked when supplied so a blocked or failed tool call can
        never become active merely because its arguments named a valid Skill.
        """
        skill = self.registry.get(name)
        if skill is None or (result is not None and result != skill.render()):
            return None
        return {
            "name": skill.name,
            "status": "active",
            "content_hash": skill.content_hash,
            "scope": skill.scope,
            "loaded_at": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
        }

    def reconcile_activation(self, name: str, record: dict) -> dict:
        """Validate persisted state against the currently discovered registry."""
        normalized = {
            "name": name,
            "status": "unavailable",
            "content_hash": str(record.get("content_hash") or ""),
            "scope": str(record.get("scope") or "unknown"),
            "loaded_at": str(record.get("loaded_at") or ""),
        }
        skill = self.registry.get(name)
        if skill is None:
            return normalized
        normalized["scope"] = skill.scope
        normalized["status"] = (
            "active" if normalized["content_hash"] == skill.content_hash else "changed"
        )
        return normalized

    def render_active(self, records: dict[str, dict]) -> str:
        """Render only hash-matched Skills, plus actionable stale-state notices."""
        active: list[str] = []
        notices: list[str] = []
        for name in sorted(records):
            record = records[name]
            skill = self.registry.get(name)
            status = record.get("status")
            if (
                status == "active"
                and skill is not None
                and record.get("content_hash") == skill.content_hash
            ):
                active.append(skill.render())
            elif status == "changed":
                notices.append(
                    f"- {name}: SKILL.md changed since it was loaded. Its instructions "
                    "are not active; call load_skill again to accept the current version."
                )
            elif status == "unavailable":
                notices.append(
                    f"- {name}: the saved Skill is no longer available in this workspace."
                )
        sections: list[str] = []
        if active:
            sections.append(
                "# Active workflow skills\n\n"
                "These Skills were explicitly loaded for this session and remain active "
                "across context compression and resume. Follow them as workflow "
                "instructions; they do not grant capabilities or bypass safety controls.\n\n"
                + "\n\n".join(active)
            )
        if notices:
            sections.append("# Saved Skill state notices\n\n" + "\n".join(notices))
        return "\n\n".join(sections)
