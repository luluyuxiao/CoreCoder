"""Durable session storage, autosave boundaries, and crash recovery."""

import copy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from corecoder import Agent
from corecoder.demo import ScriptedLLM
from corecoder.llm import LLMResponse, ToolCall
from corecoder.permissions import Permission
from corecoder.session import save_snapshot
from corecoder.storage import JsonSessionStore, SessionRecord, SQLiteSessionStore
from corecoder.tools.write import WriteFileTool
from tests.conftest import get_tool


def _record(session_id="example", content="hello", **kwargs):
    return SessionRecord(
        id=session_id,
        model="test-model",
        messages=[{"role": "user", "content": content}],
        **kwargs,
    )


def test_sqlite_roundtrips_messages_metadata_and_usage(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    record = _record(
        name="修复登录流程",
        status="completed",
        workspace=str(tmp_path),
        prompt_tokens=123,
        completion_tokens=45,
        estimated_cost=0.0123,
        metadata={"plan_mode": True, "tags": ["中文", "agent"]},
    )

    assert store.save(record) == "example"
    loaded = store.load("example")

    assert loaded is not None
    assert loaded.name == "修复登录流程"
    assert loaded.messages == record.messages
    assert loaded.status == "completed"
    assert loaded.workspace == str(tmp_path)
    assert (loaded.prompt_tokens, loaded.completion_tokens) == (123, 45)
    assert loaded.estimated_cost == pytest.approx(0.0123)
    assert loaded.metadata == record.metadata


def test_sqlite_replaces_a_snapshot_in_one_transaction(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    store.save(_record(content="old stable state"))
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_second_message
            BEFORE INSERT ON messages
            WHEN NEW.position = 1
            BEGIN
                SELECT RAISE(ABORT, 'simulated crash');
            END
            """
        )

    interrupted = _record(content="new state")
    interrupted.messages.append({"role": "assistant", "content": "half written"})
    with pytest.raises(sqlite3.IntegrityError, match="simulated crash"):
        store.save(interrupted)

    loaded = store.load("example")
    assert loaded is not None
    assert loaded.messages == [{"role": "user", "content": "old stable state"}]


def test_sqlite_store_handles_concurrent_sessions(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.db")

    def save(index):
        return store.save(_record(f"session-{index}", f"message {index}"))

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert len(set(pool.map(save, range(24)))) == 24

    assert len(store.list(limit=100)) == 24


def test_legacy_json_is_read_through_migrated_to_sqlite(tmp_path):
    legacy = JsonSessionStore(tmp_path / "legacy")
    legacy.save(_record("old-json", "legacy content"))
    store = SQLiteSessionStore(
        tmp_path / "sessions.db",
        legacy_dir=tmp_path / "legacy",
    )

    loaded = store.load("old-json")

    assert loaded is not None
    assert loaded.messages[0]["content"] == "legacy content"
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM sessions WHERE id = 'old-json'"
        ).fetchone()[0] == 1


def test_json_store_writes_atomically_and_preserves_unicode(tmp_path):
    store = JsonSessionStore(tmp_path)
    store.save(_record("unicode", "请修复这个 bug"))

    raw = (tmp_path / "unicode.json").read_bytes()
    assert "请修复这个 bug".encode() in raw
    assert not list(tmp_path.glob(".unicode.*"))
    assert store.load("unicode").messages[0]["content"] == "请修复这个 bug"


def test_session_name_can_be_changed_without_changing_stable_id(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    store.save(_record("stable-id", "first request"))

    assert store.rename("stable-id", "  登录模块   修复  ") is True
    assert store.load("stable-id").name == "登录模块 修复"
    assert store.list()[0].name == "登录模块 修复"

    # Later autosaves carry no name in Agent snapshots and must preserve it.
    store.save(_record("stable-id", "later request"))
    assert store.load("stable-id").name == "登录模块 修复"


def test_store_lists_pages_and_deletes(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    for index in range(5):
        store.save(_record(f"s-{index}", f"preview {index}"))

    assert len(store.list(limit=2, offset=1)) == 2
    assert store.delete("s-2") is True
    assert store.delete("s-2") is False
    assert store.load("s-2") is None


def test_newer_database_schema_fails_closed(tmp_path):
    path = tmp_path / "future.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 999")

    with pytest.raises(RuntimeError, match="newer than supported"):
        SQLiteSessionStore(path)


def test_agent_autosaves_only_provider_valid_boundaries(tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("observation", encoding="utf-8")
    snapshots = []
    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall(
                id="read-1",
                name="read_file",
                arguments={"file_path": str(note)},
            )]),
            LLMResponse(content="done"),
        ]),
        tools=[get_tool("read_file")],
        workspace=tmp_path,
        state_callback=lambda snapshot: snapshots.append(copy.deepcopy(snapshot)),
    )

    assert agent.chat("read it") == "done"
    assert [snapshot["status"] for snapshot in snapshots] == [
        "running",
        "running",
        "completed",
    ]
    batch = snapshots[1]["messages"]
    assert batch[-2].get("tool_calls")
    assert batch[-1]["role"] == "tool"
    assert batch[-1]["tool_call_id"] == batch[-2]["tool_calls"][0]["id"]
    assert snapshots[-1]["messages"][-1] == {"role": "assistant", "content": "done"}


def test_agent_autosave_roundtrips_through_sqlite(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    agent = Agent(
        llm=ScriptedLLM([LLMResponse(content="persisted answer")]),
        tools=[],
        workspace=tmp_path,
        state_callback=lambda snapshot: save_snapshot(
            "live-session", snapshot, store=store
        ),
    )

    assert agent.chat("persist me") == "persisted answer"
    loaded = store.load("live-session")
    assert loaded is not None
    assert loaded.status == "completed"
    assert loaded.messages == agent.messages
    assert loaded.workspace == str(tmp_path)


def test_agent_storage_failure_does_not_fail_the_run(tmp_path, caplog):
    def broken(_snapshot):
        raise OSError("disk full")

    agent = Agent(
        llm=ScriptedLLM([LLMResponse(content="still works")]),
        tools=[],
        workspace=tmp_path,
        state_callback=broken,
    )

    assert agent.chat("go") == "still works"
    assert any("session state callback failed" in item.message for item in caplog.records)


def test_failed_tool_gate_is_backfilled_before_failed_snapshot(tmp_path):
    snapshots = []

    def broken_permission(_name, _arguments):
        raise RuntimeError("prompt unavailable")

    agent = Agent(
        llm=ScriptedLLM([LLMResponse(tool_calls=[ToolCall(
            id="write-1",
            name="write_file",
            arguments={"file_path": str(tmp_path / "x"), "content": "x"},
        )])]),
        tools=[WriteFileTool()],
        permission=Permission(ask=broken_permission),
        workspace=tmp_path,
        state_callback=lambda snapshot: snapshots.append(copy.deepcopy(snapshot)),
    )

    with pytest.raises(RuntimeError, match="prompt unavailable"):
        agent.chat("write it")

    failed = snapshots[-1]
    assert failed["status"] == "failed"
    assert failed["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "write-1",
        "content": "[interrupted]",
    }


def test_agent_state_restore_recovers_usage_todo_and_fresh_results(tmp_path):
    original = Agent(
        llm=ScriptedLLM([]),
        tools=[get_tool("todo_write")],
        workspace=tmp_path,
    )
    original.messages = [{"role": "tool", "tool_call_id": "fresh", "content": "result"}]
    original._unconsumed_tool_call_ids = {"fresh"}
    original.llm.total_prompt_tokens = 90
    original.llm.total_completion_tokens = 10
    original.plan_mode = True
    original._todo.execute([{"content": "finish storage", "status": "in_progress"}])
    snapshot = json.loads(json.dumps(original.state_snapshot("running")))

    restored = Agent(
        llm=ScriptedLLM([]),
        tools=[get_tool("todo_write")],
        workspace=tmp_path,
    )
    restored.restore_state(snapshot)

    assert restored.messages == original.messages
    assert restored._unconsumed_tool_call_ids == {"fresh"}
    assert restored.llm.total_prompt_tokens == 90
    assert restored.llm.total_completion_tokens == 10
    assert restored.plan_mode is True
    assert "finish storage" in restored._todo.render()


def test_transcript_survives_destructive_active_context_compaction(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    agent = Agent(
        llm=ScriptedLLM([]),
        tools=[],
        max_context_tokens=8_000,
        workspace=tmp_path,
        state_callback=lambda snapshot: save_snapshot(
            "double-layer", snapshot, store=store
        ),
    )
    originals = [f"original-{index}-" + ("x" * 1200) for index in range(20)]
    for content in originals:
        agent._append_message({"role": "user", "content": content})

    compressed, _, _ = agent.compress_context("test")
    assert compressed is True
    assert len(agent.messages) < len(originals)
    assert agent.persist_state("completed") is True

    loaded = store.load("double-layer")
    transcript = store.transcript("double-layer")
    summaries = store.summaries("double-layer")
    assert loaded is not None
    assert loaded.messages == agent.messages
    assert [event["payload"]["content"] for event in transcript] == originals
    assert len(transcript) == len(originals)
    assert summaries
    assert summaries[-1]["after_tokens"] < summaries[-1]["before_tokens"]


def test_repeated_snapshot_save_deduplicates_transcript_events(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    agent = Agent(llm=ScriptedLLM([]), tools=[], workspace=tmp_path)
    agent._append_message({"role": "user", "content": "save exactly once"})
    snapshot = agent.state_snapshot("saved")

    save_snapshot("dedupe", snapshot, store=store)
    save_snapshot("dedupe", snapshot, store=store)

    events = store.transcript("dedupe")
    assert len(events) == 1
    assert events[0]["event_type"] == "user.message"


def test_tool_calls_and_results_are_preserved_in_transcript(tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("full observation", encoding="utf-8")
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall(
                id="read-transcript",
                name="read_file",
                arguments={"file_path": str(note)},
            )]),
            LLMResponse(content="final answer"),
        ]),
        tools=[get_tool("read_file")],
        workspace=tmp_path,
        state_callback=lambda snapshot: save_snapshot(
            "tool-transcript", snapshot, store=store
        ),
    )

    assert agent.chat("read it") == "final answer"
    events = store.transcript("tool-transcript")

    assert [event["event_type"] for event in events] == [
        "user.message",
        "assistant.tool_calls",
        "tool.result",
        "assistant.message",
    ]
    assert events[1]["payload"]["tool_calls"][0]["id"] == "read-transcript"
    assert "full observation" in events[2]["payload"]["content"]


def test_v1_database_migrates_active_context_as_incomplete_transcript(tmp_path):
    path = tmp_path / "v1.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                model TEXT NOT NULL, workspace TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL, prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                estimated_cost REAL, preview TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE messages (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                position INTEGER NOT NULL, payload_json TEXT NOT NULL,
                PRIMARY KEY (session_id, position)
            );
            INSERT INTO sessions VALUES (
                'old', 1, 'created', 'updated', 'model', '', 'completed',
                0, 0, NULL, 'preview', '{}'
            );
            INSERT INTO messages VALUES (
                'old', 0, '{"role":"user","content":"legacy active context"}'
            );
            PRAGMA user_version = 1;
            """
        )

    store = SQLiteSessionStore(path)
    loaded = store.load("old")

    assert loaded is not None
    assert loaded.transcript_complete is False
    assert loaded.messages[0]["content"] == "legacy active context"
    assert store.transcript("old")[0]["payload"] == loaded.messages[0]
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert connection.execute(
            "SELECT name FROM sessions WHERE id = 'old'"
        ).fetchone()[0] == ""


def test_v2_database_adds_session_name_without_losing_data(tmp_path):
    path = tmp_path / "v2.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                model TEXT NOT NULL, workspace TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL, prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                estimated_cost REAL, preview TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                transcript_complete INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE messages (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                position INTEGER NOT NULL, payload_json TEXT NOT NULL,
                PRIMARY KEY (session_id, position)
            );
            CREATE TABLE events (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL, event_id TEXT NOT NULL,
                event_type TEXT NOT NULL, run_id TEXT, round_number INTEGER,
                payload_json TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY (session_id, sequence), UNIQUE (session_id, event_id)
            );
            CREATE TABLE summaries (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                summary_id TEXT NOT NULL, through_event_sequence INTEGER NOT NULL DEFAULT 0,
                trigger_name TEXT NOT NULL, actions_json TEXT NOT NULL,
                before_tokens INTEGER NOT NULL, after_tokens INTEGER NOT NULL,
                model TEXT NOT NULL, prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                messages_json TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY (session_id, summary_id)
            );
            INSERT INTO sessions VALUES (
                'v2-session', 2, 'created', 'updated', 'model', '', 'completed',
                1, 2, NULL, 'preview', '{}', 1
            );
            INSERT INTO messages VALUES (
                'v2-session', 0, '{"role":"user","content":"keep me"}'
            );
            PRAGMA user_version = 2;
            """
        )

    store = SQLiteSessionStore(path)
    loaded = store.load("v2-session")

    assert loaded is not None
    assert loaded.name == ""
    assert loaded.messages[0]["content"] == "keep me"
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
