"""Undo checkpoints for edit_file / write_file."""

from __future__ import annotations

import json

from corecoder import checkpoints
from corecoder.agent import Agent
from corecoder.checkpoints import CheckpointManager
from corecoder.demo import ScriptedLLM
from corecoder.tools.edit import EditFileTool
from corecoder.tools.write import WriteFileTool


def setup_function():
    checkpoints.clear()


def test_edit_then_undo_restores_previous_bytes(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("v1\n", encoding="utf-8")
    assert EditFileTool().execute(str(f), "v1", "v2").startswith("Edited")
    assert f.read_text() == "v2\n"

    assert checkpoints.undo() == f"Restored {f}."
    assert f.read_text() == "v1\n"


def test_undo_removes_file_created_by_write(tmp_path):
    f = tmp_path / "new.py"
    assert not f.exists()
    WriteFileTool().execute(str(f), "print(1)\n")
    assert f.exists()

    assert checkpoints.undo() == f"Removed {f} (created this session)."
    assert not f.exists()


def test_undo_pops_one_mutation_at_a_time(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("v1\n", encoding="utf-8")
    EditFileTool().execute(str(f), "v1", "v2")
    EditFileTool().execute(str(f), "v2", "v3")

    assert checkpoints.pending() == 2
    checkpoints.undo()
    assert f.read_text() == "v2\n"
    checkpoints.undo()
    assert f.read_text() == "v1\n"


def test_failed_edit_leaves_no_checkpoint(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("v1\n", encoding="utf-8")
    # old_string absent -> error before any write
    result = EditFileTool().execute(str(f), "missing", "v2")
    assert result.startswith("Error:")
    assert checkpoints.pending() == 0
    assert checkpoints.undo() == "Nothing to undo."


def test_undo_recreates_a_deleted_parent_tree(tmp_path):
    # bash side effects are untracked: if the parent dir is removed between
    # the checkpoint and /undo, restore recreates it instead of dying.
    f = tmp_path / "deep" / "nested" / "a.py"
    f.parent.mkdir(parents=True)
    f.write_text("v1\n", encoding="utf-8")
    EditFileTool().execute(str(f), "v1", "v2")
    import shutil

    shutil.rmtree(tmp_path / "deep")
    assert not f.parent.exists()

    msg = checkpoints.undo()
    assert msg == f"Restored {f} (recreated missing parent directories)."
    assert f.read_text() == "v1\n"


def test_agent_scoped_checkpoint_survives_session_snapshot_restore(tmp_path):
    target = tmp_path / "durable.txt"
    original_manager = CheckpointManager()
    original_tool = WriteFileTool(checkpoint_manager=original_manager)
    original = Agent(llm=ScriptedLLM([]), tools=[original_tool], workspace=tmp_path)
    original_tool.execute(str(target), "created\n")
    snapshot = json.loads(json.dumps(original.state_snapshot()))

    restored_manager = CheckpointManager()
    restored = Agent(
        llm=ScriptedLLM([]),
        tools=[WriteFileTool(checkpoint_manager=restored_manager)],
        workspace=tmp_path,
    )
    restored.restore_state(snapshot)

    assert restored.checkpoints.pending() == 1
    assert restored.checkpoints.undo() == f"Removed {target} (created this session)."
    assert not target.exists()
