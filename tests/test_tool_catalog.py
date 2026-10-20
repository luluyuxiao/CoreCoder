"""Budgeted Tool Schema discovery and activation."""

import json

from corecoder import Agent, Config
from corecoder.demo import ScriptedLLM
from corecoder.llm import LLMResponse, ToolCall
from corecoder.permissions import Permission
from corecoder.tools import build_tools
from corecoder.tools.base import Tool, ToolEffect


class _CatalogTool(Tool):
    effect = ToolEffect.PURE
    capabilities = frozenset()

    def __init__(self, name: str, purpose: str):
        self.name = name
        self.description = purpose + " " + ("detail " * 140)
        self.parameters = {
            "type": "object",
            "properties": {
                "value": {"type": "string", "description": purpose},
            },
            "required": ["value"],
        }
        self.calls = 0

    def execute(self, value: str) -> str:
        self.calls += 1
        return f"{self.name}: {value}"


def _large_tool_set():
    return [
        _CatalogTool("weather_forecast", "weather forecast by city"),
        _CatalogTool("calendar_lookup", "calendar events and schedules"),
        _CatalogTool("issue_search", "repository issue search"),
        _CatalogTool("database_query", "read database records"),
    ]


def _schema_names(agent: Agent) -> set[str]:
    return {schema["function"]["name"] for schema in agent._tool_schemas()}


def test_small_registry_keeps_legacy_all_tools_behavior():
    tool = _CatalogTool("weather_forecast", "weather forecast")
    agent = Agent(
        llm=ScriptedLLM([]),
        tools=[tool],
        tool_schema_budget_tokens=10_000,
    )

    assert agent.tool_catalog.enabled is False
    assert set(agent._tool_by_name) == {"weather_forecast"}
    assert _schema_names(agent) == {"weather_forecast"}


def test_large_registry_exposes_discovery_not_hidden_schemas():
    agent = Agent(
        llm=ScriptedLLM([]),
        tools=_large_tool_set(),
        tool_schema_budget_tokens=1_200,
    )

    assert agent.tool_catalog.enabled is True
    assert _schema_names(agent) == {"search_tools", "activate_tools"}
    system = agent._full_messages()[0]["content"]
    assert "search_tools" in system
    assert "weather_forecast" not in system

    result = json.loads(agent._tool_by_name["search_tools"].execute("weather city"))
    assert result["results"][0]["name"] == "weather_forecast"
    assert len(result["results"][0]["description"]) <= 240


def test_activation_adds_full_schema_and_replaces_previous_selection():
    agent = Agent(
        llm=ScriptedLLM([]),
        tools=_large_tool_set(),
        tool_schema_budget_tokens=1_200,
    )
    activate = agent._tool_by_name["activate_tools"]

    first = json.loads(activate.execute(["weather_forecast"]))
    assert first["activated"] == ["weather_forecast"]
    assert "weather_forecast" in _schema_names(agent)

    second = json.loads(activate.execute(["calendar_lookup"]))
    assert second["activated"] == ["calendar_lookup"]
    assert second["deactivated"] == ["weather_forecast"]
    assert "calendar_lookup" in _schema_names(agent)
    assert "weather_forecast" not in _schema_names(agent)
    assert second["active_schema_tokens"] <= second["schema_budget_tokens"]


def test_core_selection_reserves_budget_for_one_discovered_tool():
    extra = _CatalogTool("weather_forecast", "weather forecast by city")
    agent = Agent(
        llm=ScriptedLLM([]),
        tools=[*build_tools(), extra],
        tool_schema_budget_tokens=1_024,
    )

    result = agent.tool_catalog.activate(["weather_forecast"])

    assert result["activated"] == ["weather_forecast"]
    assert result["active_schema_tokens"] <= 1_024


def test_hidden_tool_call_is_observation_not_execution():
    tools = _large_tool_set()
    weather = tools[0]
    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall(
                id="c1",
                name="weather_forecast",
                arguments={"value": "Shanghai"},
            )]),
            LLMResponse(content="I need to activate it first."),
        ]),
        tools=tools,
        permission=Permission(allow_all=True),
        tool_schema_budget_tokens=1_200,
        tool_retrieval_limit=0,
    )

    assert agent.chat("check weather") == "I need to activate it first."
    assert weather.calls == 0
    assert "schema is not active" in agent.messages[2]["content"]


def test_discover_activate_then_call_uses_schema_on_next_round():
    tools = _large_tool_set()
    weather = tools[0]

    class RecordingLLM(ScriptedLLM):
        def __init__(self):
            super().__init__([
                LLMResponse(tool_calls=[ToolCall(
                    id="search",
                    name="search_tools",
                    arguments={"query": "weather"},
                )]),
                LLMResponse(tool_calls=[ToolCall(
                    id="activate",
                    name="activate_tools",
                    arguments={"tool_names": ["weather_forecast"]},
                )]),
                LLMResponse(tool_calls=[ToolCall(
                    id="weather",
                    name="weather_forecast",
                    arguments={"value": "Shanghai"},
                )]),
                LLMResponse(content="sunny"),
            ])
            self.schemas = []

        def chat(self, messages, tools=None, on_token=None, on_event=None):
            self.schemas.append({item["function"]["name"] for item in tools or []})
            return super().chat(messages, tools, on_token, on_event)

    llm = RecordingLLM()
    agent = Agent(
        llm=llm,
        tools=tools,
        tool_schema_budget_tokens=1_200,
        tool_retrieval_limit=0,
    )

    assert agent.chat("weather in Shanghai") == "sunny"
    assert weather.calls == 1
    assert llm.schemas[0] == {"search_tools", "activate_tools"}
    assert "weather_forecast" in llm.schemas[2]


def test_active_selection_survives_session_restore():
    tools = _large_tool_set()
    original = Agent(
        llm=ScriptedLLM([]),
        tools=tools,
        tool_schema_budget_tokens=1_200,
    )
    original.tool_catalog.activate(["issue_search"])

    restored = Agent(
        llm=ScriptedLLM([]),
        tools=_large_tool_set(),
        tool_schema_budget_tokens=1_200,
    )
    restored.restore_state(original.state_snapshot())

    assert "issue_search" in _schema_names(restored)
    assert "weather_forecast" not in _schema_names(restored)


def test_registry_refresh_preserves_valid_active_name_with_new_wrapper():
    original_tools = _large_tool_set()
    agent = Agent(
        llm=ScriptedLLM([]),
        tools=original_tools,
        tool_schema_budget_tokens=1_200,
    )
    agent.tool_catalog.activate(["issue_search"])
    refreshed = _large_tool_set()

    agent.replace_tools(refreshed)

    assert agent._tool_by_name["issue_search"] is refreshed[2]
    assert "issue_search" in _schema_names(agent)
    assert "weather_forecast" not in _schema_names(agent)


def test_tool_schema_budget_loads_from_environment(monkeypatch):
    monkeypatch.setenv("CORECODER_TOOL_SCHEMA_BUDGET", "2500")
    monkeypatch.setenv("CORECODER_TOOL_RETRIEVAL_LIMIT", "7")

    assert Config.from_env().tool_schema_budget_tokens == 2500
    assert Config.from_env().tool_retrieval_limit == 7
    assert Permission().decide("search_tools", {}).allowed
    assert Permission().decide("activate_tools", {}).allowed


def test_auto_retrieval_exposes_relevant_schema_on_first_llm_round():
    tools = _large_tool_set()
    weather = tools[0]

    class RecordingLLM(ScriptedLLM):
        def __init__(self):
            super().__init__([
                LLMResponse(tool_calls=[ToolCall(
                    id="weather",
                    name="weather_forecast",
                    arguments={"value": "Shanghai"},
                )]),
                LLMResponse(content="sunny"),
            ])
            self.schemas = []

        def chat(self, messages, tools=None, on_token=None, on_event=None):
            self.schemas.append({item["function"]["name"] for item in tools or []})
            return super().chat(messages, tools, on_token, on_event)

    llm = RecordingLLM()
    agent = Agent(
        llm=llm,
        tools=tools,
        tool_schema_budget_tokens=1_200,
        tool_retrieval_limit=5,
    )

    assert agent.chat("weather forecast for Shanghai") == "sunny"
    assert weather.calls == 1
    assert "weather_forecast" in llm.schemas[0]
    assert "search_tools" in llm.schemas[0]
    assert agent.tool_catalog.last_retrieval["candidates"][0] == "weather_forecast"


def test_auto_retrieval_replaces_previous_dynamic_selection():
    agent = Agent(
        llm=ScriptedLLM([LLMResponse(content="done")]),
        tools=_large_tool_set(),
        tool_schema_budget_tokens=1_200,
    )
    agent.tool_catalog.activate(["weather_forecast"])

    assert agent.chat("search repository issues") == "done"
    assert "issue_search" in _schema_names(agent)
    assert "weather_forecast" not in _schema_names(agent)


def test_auto_retrieval_zero_matches_keeps_previous_selection():
    agent = Agent(
        llm=ScriptedLLM([LLMResponse(content="done")]),
        tools=_large_tool_set(),
        tool_schema_budget_tokens=1_200,
    )
    agent.tool_catalog.activate(["weather_forecast"])

    assert agent.chat("unrelated capability with no catalog vocabulary") == "done"
    assert "weather_forecast" in _schema_names(agent)
    assert agent.tool_catalog.last_retrieval["retained_previous"] is True


def test_hundred_tool_catalog_retrieves_a_relevant_first_round_schema():
    target = _CatalogTool("weather_forecast", "special weather forecast by city")
    tools = [
        *[
            _CatalogTool(f"service_{index:03d}", f"generic service capability {index}")
            for index in range(99)
        ],
        target,
    ]
    llm = ScriptedLLM([LLMResponse(content="ready")])
    agent = Agent(
        llm=llm,
        tools=tools,
        tool_schema_budget_tokens=1_200,
        tool_retrieval_limit=5,
    )

    assert agent.chat("special weather forecast for Shanghai") == "ready"
    retrieval = agent.tool_catalog.last_retrieval
    assert len(agent.tools) == 100
    assert retrieval["candidates"][0] == "weather_forecast"
    assert len(retrieval["candidates"]) <= 5
    assert "weather_forecast" in _schema_names(agent)
    assert retrieval["active_schema_tokens"] <= retrieval["schema_budget_tokens"]
