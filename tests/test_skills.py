"""Skill discovery and lazy instruction loading."""

import json
from pathlib import Path

import pytest

from corecoder import Agent
from corecoder.demo import ScriptedLLM
from corecoder.llm import LLMResponse, ToolCall
from corecoder.permissions import Permission
from corecoder.session import save_snapshot
from corecoder.skills import SkillRegistry
from corecoder.storage import SQLiteSessionStore
from corecoder.tools import build_tools
from corecoder.tools.skill import LoadSkillTool


def _write_skill(root: Path, directory: str, name: str, description: str, body: str):
    target = root / directory
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )


def test_project_skill_overrides_same_named_global_skill(tmp_path):
    global_dir = tmp_path / "global"
    project_dir = tmp_path / "project"
    _write_skill(global_dir, "review", "review", "global description", "global body")
    _write_skill(project_dir, "review", "review", "project description", "project body")

    registry = SkillRegistry.discover(
        tmp_path,
        global_dir=global_dir,
        project_dir=project_dir,
    )

    assert registry.names() == ["review"]
    assert registry.get("review").scope == "project"
    assert "project body" in registry.load("review")
    assert "global body" not in registry.load("review")


def test_invalid_skill_is_skipped_with_a_warning(tmp_path, caplog):
    project_dir = tmp_path / "skills"
    _write_skill(project_dir, "bad", "Bad Name", "description", "body")

    registry = SkillRegistry.discover(
        tmp_path,
        global_dir=tmp_path / "missing",
        project_dir=project_dir,
    )

    assert not registry
    assert "ignoring invalid skill" in caplog.text


def test_skill_symlink_cannot_escape_discovery_root(tmp_path, caplog):
    outside = tmp_path / "outside"
    _write_skill(outside, "secret", "secret", "outside", "do not load")
    root = tmp_path / "skills"
    root.mkdir()
    try:
        (root / "linked").symlink_to(outside / "secret", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    registry = SkillRegistry.discover(
        tmp_path,
        global_dir=tmp_path / "missing",
        project_dir=root,
    )

    assert not registry
    assert "outside its skill root" in caplog.text


def test_load_skill_is_read_only_and_uses_an_enum_schema(tmp_path):
    project_dir = tmp_path / "skills"
    _write_skill(project_dir, "review", "review", "Review carefully", "Read before judging.")
    registry = SkillRegistry.discover(
        tmp_path,
        global_dir=tmp_path / "missing",
        project_dir=project_dir,
    )
    tool = LoadSkillTool(registry)

    assert tool.is_concurrency_safe()
    assert tool.schema()["function"]["parameters"]["properties"]["name"]["enum"] == ["review"]
    assert Permission().check("load_skill", {"name": "review"}) is None
    assert "Read before judging." in tool.execute("review")
    assert "does not grant tools" in tool.execute("review")
    assert tool.activation_record("review", "blocked by hook") is None


def test_agent_can_lazy_load_a_skill_as_an_ordinary_tool_result(tmp_path):
    project_dir = tmp_path / "skills"
    _write_skill(project_dir, "review", "review", "Review carefully", "Inspect the tests.")
    registry = SkillRegistry.discover(
        tmp_path,
        global_dir=tmp_path / "missing",
        project_dir=project_dir,
    )
    skill_tool = LoadSkillTool(registry)
    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall(
                id="skill-1",
                name="load_skill",
                arguments={"name": "review"},
            )]),
            LLMResponse(content="review ready"),
        ]),
        tools=[skill_tool],
        permission=Permission(),
    )

    assert agent.chat("review this") == "review ready"
    assert agent.messages[2]["role"] == "tool"
    assert "Inspect the tests." in agent.messages[2]["content"]
    assert agent.active_skills["review"]["status"] == "active"
    assert "Inspect the tests." in agent._full_messages()[0]["content"]


def test_active_skill_survives_context_loss_and_sqlite_resume(tmp_path):
    project_dir = tmp_path / "skills"
    _write_skill(
        project_dir,
        "review",
        "review",
        "Review carefully",
        "Persistent review instructions.",
    )
    registry = SkillRegistry.discover(
        tmp_path,
        global_dir=tmp_path / "missing",
        project_dir=project_dir,
    )
    original = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall(
                id="skill-persist",
                name="load_skill",
                arguments={"name": "review"},
            )]),
            LLMResponse(content="loaded"),
        ]),
        tools=[LoadSkillTool(registry)],
        workspace=tmp_path,
    )
    assert original.chat("load it") == "loaded"

    # Model the result of destructive context compaction: the original Tool
    # Result is gone, while structured Skill state remains independently saved.
    original.messages = [{"role": "user", "content": "continue reviewing"}]
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    snapshot = json.loads(json.dumps(original.state_snapshot("saved")))
    save_snapshot("skill-resume", snapshot, store=store)
    record = store.load("skill-resume")
    assert record is not None
    assert record.metadata["active_skills"]["review"]["content_hash"]
    assert "Persistent review instructions." not in json.dumps(record.metadata)

    restored = Agent(
        llm=ScriptedLLM([]),
        tools=[LoadSkillTool(registry)],
        workspace=tmp_path,
    )
    restored.restore_state(record.to_snapshot())

    system = restored._full_messages()[0]["content"]
    assert restored.active_skill_states()["review"]["status"] == "active"
    assert "Persistent review instructions." in system
    assert "across context compression and resume" in system


def test_changed_skill_requires_an_explicit_reload_after_resume(tmp_path):
    project_dir = tmp_path / "skills"
    _write_skill(
        project_dir,
        "review",
        "review",
        "Review carefully",
        "Original instructions.",
    )
    original_registry = SkillRegistry.discover(
        tmp_path,
        global_dir=tmp_path / "missing",
        project_dir=project_dir,
    )
    original = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall(
                id="skill-original",
                name="load_skill",
                arguments={"name": "review"},
            )]),
            LLMResponse(content="loaded"),
        ]),
        tools=[LoadSkillTool(original_registry)],
        workspace=tmp_path,
    )
    original.chat("load it")
    snapshot = json.loads(json.dumps(original.state_snapshot()))

    (project_dir / "review" / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review carefully\n---\nChanged instructions.\n",
        encoding="utf-8",
    )
    changed_registry = SkillRegistry.discover(
        tmp_path,
        global_dir=tmp_path / "missing",
        project_dir=project_dir,
    )
    restored = Agent(
        llm=ScriptedLLM([]),
        tools=[LoadSkillTool(changed_registry)],
        workspace=tmp_path,
    )
    restored.restore_state(snapshot)

    system = restored._full_messages()[0]["content"]
    assert restored.active_skill_states()["review"]["status"] == "changed"
    assert "changed since it was loaded" in system
    assert "Changed instructions." not in system


def test_reset_clears_active_skills(tmp_path):
    project_dir = tmp_path / "skills"
    _write_skill(project_dir, "review", "review", "Review", "Instructions.")
    registry = SkillRegistry.discover(
        tmp_path,
        global_dir=tmp_path / "missing",
        project_dir=project_dir,
    )
    tool = LoadSkillTool(registry)
    agent = Agent(llm=ScriptedLLM([]), tools=[tool], workspace=tmp_path)
    agent._activate_skill_from_result(
        ToolCall(id="skill-reset", name="load_skill", arguments={"name": "review"}),
        tool.execute("review"),
    )

    agent.reset()

    assert agent.active_skill_states() == {}
    assert "# Active workflow skills" not in agent._full_messages()[0]["content"]


def test_build_tools_adds_skill_tool_only_for_nonempty_registry(tmp_path):
    empty = SkillRegistry()
    assert "load_skill" not in {tool.name for tool in build_tools(skill_registry=empty)}

    project_dir = tmp_path / "skills"
    _write_skill(project_dir, "review", "review", "Review carefully", "Inspect it.")
    registry = SkillRegistry.discover(
        tmp_path,
        global_dir=tmp_path / "missing",
        project_dir=project_dir,
    )

    assert "load_skill" in {tool.name for tool in build_tools(skill_registry=registry)}


def test_repository_corecoder_review_skill_is_discoverable(tmp_path):
    root = Path(__file__).resolve().parent.parent
    registry = SkillRegistry.discover(root, global_dir=tmp_path / "missing")
    skill = registry.get("corecoder-review")

    assert skill is not None
    assert skill.scope == "project"
    assert "tool_calls" in skill.instructions
    assert "Transcript Events remain append-only" in skill.instructions
