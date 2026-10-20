"""Budgeted Tool discovery and lazy schema activation.

Execution registration and model visibility are deliberately separate.  Every
Tool remains available to the runtime, but once the complete schema set exceeds
the configured budget only core Tools plus two tiny discovery Tools are sent to
the model.  A new user task first retrieves and exposes a small relevant set;
the model can still search the catalog and activate different schemas when that
automatic selection is insufficient.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import ClassVar

from .context import estimate_tool_schema_tokens
from .tools.base import Tool, ToolEffect

SEARCH_TOOL_NAME = "search_tools"
ACTIVATE_TOOL_NAME = "activate_tools"
DISCOVERY_TOOL_NAMES = frozenset({SEARCH_TOOL_NAME, ACTIVATE_TOOL_NAME})

# These are the small, general-purpose capabilities that let a coding agent
# inspect a task and recover additional schemas.  They are preferred, not
# blindly forced: an unusually tiny explicit budget may omit lower-priority
# entries while always retaining the two discovery Tools.
CORE_TOOL_NAMES = (
    "read_file",
    "grep",
    "glob",
    "memory_update",
    "todo_write",
    "now",
    "load_skill",
    "edit_file",
    "write_file",
    "bash",
    "agent_status",
    "agent",
    "agent_resume",
    "fetch_url",
)


def default_tool_schema_budget(max_context_tokens: int) -> int:
    """Use at most ten percent of context, capped for very large windows."""
    return min(12_000, max(1_024, int(max_context_tokens * 0.10)))


class ToolCatalog:
    """All executable Tools plus a bounded model-visible active subset."""

    def __init__(self, tools: Iterable[Tool], budget_tokens: int):
        if budget_tokens <= 0:
            raise ValueError("tool schema budget must be greater than zero")
        self.budget_tokens = int(budget_tokens)
        self._tools: dict[str, Tool] = {}
        self._order: list[str] = []
        self._active: set[str] = set()
        self.enabled = False
        self.search_tool = ToolSearchTool(self)
        self.activate_tool = ToolActivateTool(self)
        self.replace(tools)

    @property
    def discovery_tools(self) -> tuple[Tool, Tool]:
        return self.search_tool, self.activate_tool

    @property
    def all_tools(self) -> list[Tool]:
        return [self._tools[name] for name in self._order]

    @property
    def active_names(self) -> list[str]:
        return [tool.name for tool in self.schema_tools()]

    @property
    def hidden_names(self) -> list[str]:
        return [name for name in self._order if name not in self._active]

    @property
    def active_actual_names(self) -> list[str]:
        return [name for name in self._order if name in self._active]

    @property
    def active_schema_tokens(self) -> int:
        return estimate_tool_schema_tokens([tool.schema() for tool in self.schema_tools()])

    @property
    def total_schema_tokens(self) -> int:
        return estimate_tool_schema_tokens([tool.schema() for tool in self.all_tools])

    def get(self, name: str) -> Tool | None:
        if name == SEARCH_TOOL_NAME and self.enabled:
            return self.search_tool
        if name == ACTIVATE_TOOL_NAME and self.enabled:
            return self.activate_tool
        return self._tools.get(name)

    def is_active(self, name: str) -> bool:
        if not self.enabled:
            return name in self._tools
        return name in DISCOVERY_TOOL_NAMES or name in self._active

    def schema_tools(self) -> list[Tool]:
        if not self.enabled:
            return self.all_tools
        active = [self._tools[name] for name in self._order if name in self._active]
        return [*active, *self.discovery_tools]

    def replace(self, tools: Iterable[Tool]):
        """Replace the catalog while preserving valid on-demand selections."""
        self.last_retrieval: dict | None = None
        previous = set(self._active)
        updated = list(tools)
        names = [tool.name for tool in updated]
        if len(set(names)) != len(names):
            raise ValueError("tool names must be unique")
        enabled = (
            estimate_tool_schema_tokens([tool.schema() for tool in updated])
            > self.budget_tokens
        )
        reserved = DISCOVERY_TOOL_NAMES.intersection(names)
        if enabled and reserved:
            joined = ", ".join(sorted(reserved))
            raise ValueError(f"tool names are reserved for lazy discovery: {joined}")
        discovery_tokens = estimate_tool_schema_tokens([
            tool.schema() for tool in self.discovery_tools
        ])
        if enabled and discovery_tokens > self.budget_tokens:
            raise ValueError(
                "tool schema budget is too small for discovery tools: "
                f"need at least {discovery_tokens} tokens"
            )
        self._tools = {tool.name: tool for tool in updated}
        self._order = names
        self.enabled = enabled
        if not self.enabled:
            self._active = set(names)
            return

        self._active = set()
        # The discovery schemas are the non-negotiable recovery path.  Add
        # general-purpose Tools greedily, but reserve forty percent for the
        # on-demand schema the model is expected to discover next.
        core_budget = max(discovery_tokens, int(self.budget_tokens * 0.60))
        for name in CORE_TOOL_NAMES:
            if name in self._tools:
                self._try_add(name, limit=core_budget)
        for name in self._order:
            if name in previous and name not in self._active:
                self._try_add(name)

    def _schema_tokens_for(self, names: set[str]) -> int:
        selected = [self._tools[name] for name in self._order if name in names]
        if self.enabled:
            selected.extend(self.discovery_tools)
        return estimate_tool_schema_tokens([tool.schema() for tool in selected])

    def _try_add(self, name: str, *, limit: int | None = None) -> bool:
        candidate = set(self._active)
        candidate.add(name)
        if self._schema_tokens_for(candidate) > (limit or self.budget_tokens):
            return False
        self._active = candidate
        return True

    def search(self, query: str, limit: int = 10, include_active: bool = False) -> dict:
        """Return a bounded, schema-free catalog view ranked lexically."""
        query = str(query or "").strip().lower()
        limit = max(1, min(20, int(limit or 10)))
        terms = [term for term in re.findall(r"[\w.-]+", query) if term]
        ranked = []
        for index, name in enumerate(self._order):
            active = name in self._active or not self.enabled
            if active and not include_active:
                continue
            tool = self._tools[name]
            description = str(tool.description or "")
            parameters = tool.parameters if isinstance(tool.parameters, dict) else {}
            properties = parameters.get("properties") or {}
            capabilities = sorted(tool.capabilities)
            haystack = " ".join([
                name,
                description,
                " ".join(map(str, properties)),
                " ".join(capabilities),
            ]).lower()
            score = 0
            if query and query in name.lower():
                score += 100
            if query and query in description.lower():
                score += 50
            for term in terms:
                if term in name.lower():
                    score += 20
                if term in description.lower():
                    score += 8
                if term in haystack:
                    score += 2
            if query and score == 0:
                continue
            ranked.append((score, -index, tool, active, properties, parameters))

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        results = []
        for score, _index, tool, active, properties, parameters in ranked[:limit]:
            results.append({
                "name": tool.name,
                "description": str(tool.description or "")[:240],
                "parameters": list(properties)[:20],
                "required": list(parameters.get("required") or [])[:20],
                "capabilities": sorted(tool.capabilities),
                "effect": tool.effect,
                "active": active,
                "score": score,
            })
        return {
            "query": query,
            "matches": len(ranked),
            "returned": len(results),
            "results": results,
        }

    def retrieve(self, query: str, limit: int = 5) -> dict:
        """Expose a budget-fitting Top-K set for one new user task.

        Retrieval considers active and hidden Tools together so a relevant
        on-demand selection can survive the next task.  Empty retrievals and
        selections whose schemas cannot fit are non-destructive: the previous
        visible set remains available and the model can use ``search_tools`` as
        a recovery path.
        """
        limit = max(1, min(20, int(limit or 5)))
        search = self.search(query, limit=limit, include_active=True)
        candidates = [item["name"] for item in search["results"]]
        previous = set(self._active)
        if not self.enabled or not candidates:
            result = {
                "query": search["query"],
                "matches": search["matches"],
                "candidates": candidates,
                "activated": [],
                "already_active": [],
                "deactivated": [],
                "rejected_over_budget": [],
                "retained_previous": bool(self.enabled and not candidates),
                "active_tools": self.active_actual_names,
                "active_schema_tokens": self.active_schema_tokens,
                "schema_budget_tokens": self.budget_tokens,
            }
            self.last_retrieval = result
            return result

        activation = self.activate(candidates, replace=True)
        selected = [name for name in candidates if name in self._active]
        retained_previous = False
        if not selected:
            # A single unusually large Tool schema must not erase a working
            # selection merely because it could not fit beside discovery.
            self._active = previous
            retained_previous = True
            activation["deactivated"] = []
            activation["active_tools"] = self.active_actual_names
            activation["active_schema_tokens"] = self.active_schema_tokens

        result = {
            "query": search["query"],
            "matches": search["matches"],
            "candidates": candidates,
            "activated": activation["activated"],
            "already_active": activation["already_active"],
            "deactivated": activation["deactivated"],
            "rejected_over_budget": activation["rejected_over_budget"],
            "retained_previous": retained_previous,
            "active_tools": activation["active_tools"],
            "active_schema_tokens": activation["active_schema_tokens"],
            "schema_budget_tokens": activation["schema_budget_tokens"],
        }
        self.last_retrieval = result
        return result

    def activate(self, names: Iterable[str], *, replace: bool = True) -> dict:
        """Expose requested schemas, optionally replacing prior on-demand Tools."""
        requested = []
        for raw in names:
            name = str(raw or "").strip()
            if name and name not in requested:
                requested.append(name)

        core = {name for name in CORE_TOOL_NAMES if name in self._active}
        previous_dynamic = self._active - core
        if replace:
            self._active = set(core)

        activated: list[str] = []
        already_active: list[str] = []
        unknown: list[str] = []
        rejected: list[str] = []
        for name in requested:
            if name not in self._tools:
                unknown.append(name)
            elif name in self._active:
                already_active.append(name)
            elif self._try_add(name):
                activated.append(name)
            else:
                rejected.append(name)

        return {
            "activated": activated,
            "already_active": already_active,
            "deactivated": sorted(previous_dynamic - self._active) if replace else [],
            "unknown": unknown,
            "rejected_over_budget": rejected,
            "active_tools": self.active_actual_names,
            "active_schema_tokens": self.active_schema_tokens,
            "schema_budget_tokens": self.budget_tokens,
        }

    def status(self) -> dict:
        return {
            "lazy_discovery": self.enabled,
            "catalog_tools": len(self._tools),
            "active_tools": self.active_names,
            "hidden_tools": self.hidden_names,
            "active_schema_tokens": self.active_schema_tokens,
            "total_schema_tokens": self.total_schema_tokens,
            "schema_budget_tokens": self.budget_tokens,
            "last_retrieval": self.last_retrieval,
        }


class ToolSearchTool(Tool):
    name = SEARCH_TOOL_NAME
    effect = ToolEffect.PURE
    capabilities = frozenset()
    description = (
        "Search inactive Tool capabilities without loading their full schemas. "
        "Use this when the visible tools cannot complete the task, then call activate_tools."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Task, capability, service, or Tool name to search for",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "description": "Maximum compact results (default 10)",
            },
            "include_active": {
                "type": "boolean",
                "description": "Also include schemas that are already active",
            },
        },
        "required": ["query"],
    }

    def __init__(self, catalog: ToolCatalog):
        self.catalog = catalog

    def execute(self, query: str, limit: int = 10, include_active: bool = False) -> str:
        return json.dumps(
            self.catalog.search(query, limit, include_active),
            ensure_ascii=False,
            separators=(",", ":"),
        )


class ToolActivateTool(Tool):
    name = ACTIVATE_TOOL_NAME
    effect = ToolEffect.WRITE
    capabilities = frozenset()
    description = (
        "Activate full schemas returned by search_tools. By default replaces prior "
        "on-demand schemas while retaining core tools, keeping the prompt within budget."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "tool_names": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 20,
                "description": "Exact Tool names returned by search_tools",
            },
            "replace": {
                "type": "boolean",
                "description": "Replace previous on-demand schemas (default true)",
            },
        },
        "required": ["tool_names"],
    }

    def __init__(self, catalog: ToolCatalog):
        self.catalog = catalog

    def execute(self, tool_names: list[str], replace: bool = True) -> str:
        if not isinstance(tool_names, list) or not all(
            isinstance(name, str) for name in tool_names
        ):
            return "Error: tool_names must be an array of strings"
        return json.dumps(
            self.catalog.activate(tool_names, replace=replace),
            ensure_ascii=False,
            separators=(",", ":"),
        )
